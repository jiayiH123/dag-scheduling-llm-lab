#!/usr/bin/env python3
"""Portable original200BC/300PPO recipe. New outputs only; original checkpoints stay frozen."""
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import argparse,json,math,os,random,subprocess,sys,time
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'vendor'))
from budget_feedback_actor import BudgetFeedbackActor

def rpc(p,q):
    p.stdin.write(json.dumps(q)+'\n');p.stdin.flush();line=p.stdout.readline()
    if not line:raise RuntimeError('simulator exited')
    return json.loads(line)

class Prior:
    @staticmethod
    def load_prior(seed,stage):
        torch.manual_seed(seed);model=BudgetFeedbackActor()
        saved=torch.load(ROOT/f'checkpoints/warmstart_bc_seed{seed}.pt',map_location='cpu',weights_only=True)
        missing,extra=model.load_state_dict(saved['state_dict'],strict=False)
        if extra or any(not k.startswith('budget_head.') for k in missing):raise AssertionError('checkpoint mismatch')
        return model
    @staticmethod
    def evaluate(p,model,mode,ids):
        rows=[];model.eval()
        for offset in range(0,len(ids),40):
            gs=rpc(p,{'command':'get','ids':ids[offset:offset+40]});items=[]
            with torch.no_grad():
                for g in gs:
                    before=model.forward_calls;x=model(g['graph'])
                    if model.forward_calls-before!=1:raise AssertionError('forward count')
                    items.append({'id':g['id'],'logits':x.tolist(),'seed':0})
            result=rpc(p,{'command':'rollout','mode':mode,'sample':False,'items':items})
            for g,r in zip(gs,result):r['reference']=g['reference'];rows.append(r)
        return rows
prior=Prior()


def conditionals(x, decisions, mode):
    """Vectorized equivalent of independent per-decision categorical policies."""
    width = max(len(d['available']) for d in decisions)
    indices = torch.tensor([d['available']+[0]*(width-len(d['available'])) for d in decisions])
    mask = torch.tensor([[True]*len(d['available'])+[False]*(width-len(d['available'])) for d in decisions])
    offsets = torch.tensor([d['offsets']+[0.]*(width-len(d['available'])) for d in decisions])
    phi = torch.tensor([d['budget_'+mode]+[0.]*(width-len(d['available'])) for d in decisions])
    pairs = x[indices]
    score = pairs[:, :, 0]+pairs[:, :, 1]*phi+offsets
    logs = torch.log_softmax(score.masked_fill(~mask, -1e9), dim=1)
    selected = torch.tensor([d['selected'] for d in decisions])
    return logs.gather(1, selected[:, None]).squeeze(1), -(logs.exp()*logs).sum(1)

def evaluate(p, model, mode, ids):
    return prior.evaluate(p, model, mode, ids)

def validation(p, model, mode, ids):
    return sum(r['makespan']/r['reference'] for r in evaluate(p, model, mode, ids))/len(ids)

def shared_bc(p, init, seed, out):
    path = out/f'bc_seed{seed}.pt'
    model = prior.load_prior(seed, 'bc')
    if path.exists():
        saved = torch.load(path, weights_only=True)
        model.load_state_dict(saved['state_dict'])
        return model, {k: v for k, v in saved.items() if k != 'state_dict'}, []
    for parameter in model.budget_head.parameters():
        parameter.requires_grad_(False)
    rng = random.Random(seed)
    optimizer = torch.optim.Adam([v for v in model.parameters() if v.requires_grad], lr=.001)
    best = validation(p, model, 'initial', init['val_ids'])
    best_state, best_step, curves = deepcopy(model.state_dict()), 0, []
    started = time.perf_counter()
    for update in range(1, 201):
        gs = rpc(p, {'command': 'get', 'ids': rng.sample(init['train_ids'], 8)})
        losses = []
        for g in gs:
            x = model(g['graph'])
            logs, _ = conditionals(x, g['teacher_decisions'], 'initial')
            losses.append(-logs.mean())
        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss):
            raise ValueError('nonfinite BC loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1))
        if not math.isfinite(grad):
            raise ValueError('nonfinite BC gradient')
        optimizer.step()
        entry = {'seed': seed, 'stage': 'BC-large', 'update': update, 'loss': float(loss.detach()),
                 'gradient_norm': grad, 'wall_seconds': time.perf_counter()-started}
        if update % 50 == 0:
            value = validation(p, model, 'initial', init['val_ids'])
            entry['val_C_over_HEFT'] = value
            if value < best:
                best, best_state, best_step = value, deepcopy(model.state_dict()), update
            print(f'large BC seed{seed} {update} val {value:.4f} best {best:.4f}', flush=True)
        curves.append(entry)
    model.load_state_dict(best_state)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    detail = {'seed': seed, 'stage': 'BC-large', 'mode': 'initial', 'selected_step': best_step,
              'val_ratio': best, 'updates_done': 200, 'training_seconds': time.perf_counter()-started}
    torch.save({'state_dict': best_state, **detail}, path)
    return model, detail, curves

def ppo(p, init, seed, mode, bc, out):
    model = deepcopy(bc)
    path = out/f'{mode}_seed{seed}.pt'
    if path.exists():
        saved = torch.load(path, weights_only=True)
        model.load_state_dict(saved['state_dict'])
        return model, {k: v for k, v in saved.items() if k != 'state_dict'}, []
    rng = random.Random(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=.0003)
    best = validation(p, model, mode, init['val_ids'])
    best_state, best_step, curves = deepcopy(model.state_dict()), 0, []
    started, updates = time.perf_counter(), 0
    for update in range(1, 301):
        if time.perf_counter()-started >= 180:
            break
        gs = rpc(p, {'command': 'get', 'ids': rng.sample(init['train_ids'], 8)})
        with torch.no_grad():
            xs = [model(g['graph']) for g in gs]
        items = [{'id': g['id'], 'logits': x.tolist(), 'seed': seed*1000000+update*100+j}
                 for j, (g, x) in enumerate(zip(gs, xs))]
        sampled = rpc(p, {'command': 'rollout', 'mode': mode, 'sample': True, 'items': items})
        greedy = rpc(p, {'command': 'rollout', 'mode': mode, 'sample': False, 'items': items})
        oldlogs = [torch.tensor([d['selected_log_probability'] for d in r['decisions']]) for r in sampled]
        advantages = [(b['makespan']-r['makespan'])/g['reference'] for g, r, b in zip(gs, sampled, greedy)]
        lr = .0003*min(1, update/20) if update <= 20 else .00003+.5*.00027*(1+math.cos(math.pi*(update-20)/280))
        for group in optimizer.param_groups:
            group['lr'] = lr
        last_kl, epochs = 0., 0
        for epoch in range(4):
            losses, kls = [], []
            for g, r, old, adv in zip(gs, sampled, oldlogs, advantages):
                x = model(g['graph'])
                logs, entropy = conditionals(x, r['decisions'], mode)
                delta, ratio = logs-old, (logs-old).exp()
                pg = -torch.minimum(ratio*adv, ratio.clamp(.8, 1.2)*adv).sum()
                teacher, _ = conditionals(x, g['teacher_decisions'], mode)
                losses.append(pg-.02*teacher.mean()-.001*(1-update/300)*entropy.sum())
                kls.append(((ratio-1)-delta).mean().detach())
            last_kl = float(torch.stack(kls).mean())
            if last_kl > .03:
                break
            loss = torch.stack(losses).mean()
            if not torch.isfinite(loss):
                raise ValueError('nonfinite PPO loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1))
            if not math.isfinite(grad):
                raise ValueError('nonfinite PPO gradient')
            optimizer.step()
            epochs += 1
        updates = update
        entry = {'seed': seed, 'stage': 'PPO-'+mode, 'update': update, 'epochs': epochs,
                 'approx_kl': last_kl, 'lr': lr, 'wall_seconds': time.perf_counter()-started}
        if update % 50 == 0:
            value = validation(p, model, mode, init['val_ids'])
            entry['val_C_over_HEFT'] = value
            if value < best:
                best, best_state, best_step = value, deepcopy(model.state_dict()), update
            print(f'large PPO {mode} seed{seed} {update} val {value:.4f} best {best:.4f} {time.perf_counter()-started:.1f}s', flush=True)
        curves.append(entry)
    if updates % 50:
        value = validation(p, model, mode, init['val_ids'])
        if value < best:
            best, best_state, best_step = value, deepcopy(model.state_dict()), updates
    model.load_state_dict(best_state)
    detail = {'seed': seed, 'stage': 'PPO-'+mode, 'mode': mode, 'selected_step': best_step,
              'val_ratio': best, 'updates_done': updates, 'training_seconds': time.perf_counter()-started,
              'bc_anchor': .02}
    torch.save({'state_dict': best_state, **detail}, path)
    return model, detail, curves

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default=str(ROOT/'data/reference_dataset.jsonl.gz'));ap.add_argument('--seed',type=int,choices=[11,22,33],default=11);ap.add_argument('--sim-python',default=os.getenv('SIM_PYTHON',sys.executable));ap.add_argument('--output',required=True);args=ap.parse_args();out=Path(args.output)
    if out.exists():raise FileExistsError('choose a fresh output directory')
    out.mkdir(parents=True);torch.set_num_threads(1)
    p=subprocess.Popen([args.sim_python,'-m','daglab.sim_server'],cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
    try:
        init=rpc(p,{'command':'init','data':str(Path(args.data).resolve())})
        bc,bcd,curves=shared_bc(p,init,args.seed,out)
        policy,detail,rlcurves=ppo(p,init,args.seed,'local',bc,out)
        rows={stage:evaluate(p,model,'local',init['test_ids']) for stage,model in [('BC-local',bc),('PPO-local',policy)]}
        (out/'training_curves.json').write_text(json.dumps(curves+rlcurves))
        (out/'evaluation.json').write_text(json.dumps(rows))
        (out/'metadata.json').write_text(json.dumps({'BC':bcd,'PPO':detail,'seed':args.seed,'data':str(Path(args.data).name),'device':'cpu','threads':1,'classification':'portable reference replay on known reference split; not a fresh method gain'},indent=2))
        print(json.dumps({'output':str(out),'BC':bcd,'PPO':detail},indent=2))
    finally:
        if p.poll() is None:
            rpc(p,{'command':'close'});p.wait(timeout=15)
if __name__=='__main__':main()
