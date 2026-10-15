#!/usr/bin/env python3
"""Shared BC warmstart, then paired initial/local budget-score PPO arms."""
from copy import deepcopy
from datetime import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time
from zoneinfo import ZoneInfo
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
spec = importlib.util.spec_from_file_location('prior_train', ROOT/'experiments/budget_ppo/train.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
rpc = prior.rpc


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
    out = Path(__file__).parent/'results/run001'
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    paths = [Path(__file__), Path(__file__).with_name('server.py'), Path(__file__).with_name('protocol.md'),
             ROOT/'src/cached_budget_decoder.py', ROOT/'src/budget_feedback_actor.py',
             ROOT/'src/single_forward_actor.py', ROOT/'src/budget_core.py', ROOT/'src/prepared_sgs.py']
    paths += [ROOT/f'experiments/gnn_metric_pilot/results/run001/bc_seed{seed}.pt' for seed in [11, 22, 33]]
    hashes = {str(v.relative_to(ROOT)): hashlib.sha256(v.read_bytes()).hexdigest() for v in paths}
    metapath = out/'metadata.json'
    meta = {'classification': 'EXPLORATORY larger-DAG controlled pilot',
            'started': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
            'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'hashes': hashes, 'seeds': [11, 22, 33], 'arms': ['initial', 'local'],
            'bc_updates': 200, 'ppo_updates_max': 300, 'ppo_epochs_max': 4, 'batch_size': 8,
            'ppo_arm_wall_cap_seconds': 180, 'device': 'cpu', 'threads': 1, 'torch': torch.__version__,
            'status': 'running'}
    if metapath.exists():
        meta = json.loads(metapath.read_text())
        if meta['hashes'] != hashes or meta.get('status') == 'completed':
            raise RuntimeError('cannot overwrite a finished run or resume changed code')
        meta['resumed'] = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    metapath.write_text(json.dumps(meta, indent=2))
    p = subprocess.Popen(['/opt/miniconda3/envs/dag-research/bin/python', str(Path(__file__).with_name('server.py'))],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    started = time.perf_counter()
    try:
        init = rpc(p, {'command': 'init', 'out': str(out)})
        meta['latent_counts'] = init['latent_counts']
        models = []
        curvepath = out/'training_curves.json'
        curves = json.loads(curvepath.read_text()) if curvepath.exists() else []
        for seed in meta['seeds']:
            bc, detail, curve = shared_bc(p, init, seed, out)
            models.append((bc, detail))
            curves += curve
            curvepath.write_text(json.dumps(curves, indent=2))
            for mode in meta['arms']:
                model, detail, curve = ppo(p, init, seed, mode, bc, out)
                models.append((model, detail))
                curves += curve
                curvepath.write_text(json.dumps(curves, indent=2))
                meta['completed_models'] = [d for m, d in models]
                metapath.write_text(json.dumps(meta, indent=2))
        # Architecture/arm/checkpoint/seed choices finalized before any test inference.
        meta['deployment_selection'] = min([d for m, d in models if d['stage'].startswith('PPO-')], key=lambda d: d['val_ratio'])
        meta['selection_completed'] = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
        metapath.write_text(json.dumps(meta, indent=2))
        for seed in meta['seeds']:
            model = prior.load_prior(seed, 'bc')
            saved = torch.load(ROOT/f'experiments/budget_ppo/results/run001/initial_seed{seed}.pt', weights_only=True)
            model.load_state_dict(saved['state_dict'])
            models.append((model, {'seed': seed, 'stage': 'Frozen-micro-PPO', 'mode': 'initial',
                                   'selected_step': saved['selected_step'], 'val_ratio': None}))
        with (out/'evaluation.jsonl').open('w') as f:
            for model, detail in models:
                for row in evaluate(p, model, detail['mode'], init['test_ids']):
                    f.write(json.dumps({**detail, 'fold': 'test', **row}, allow_nan=False)+'\n')
                f.flush()
                print('test saved '+detail['stage']+' seed'+str(detail['seed']), flush=True)
        meta.update(models=[d for m, d in models], status='completed',
                    wall_seconds=time.perf_counter()-started,
                    finished=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                    parameters=sum(v.numel() for v in models[0][0].parameters()),
                    sampled_trajectories=sum(d['updates_done']*8 for m, d in models if d['stage'].startswith('PPO-')),
                    greedy_baseline_trajectories=sum(d['updates_done']*8 for m, d in models if d['stage'].startswith('PPO-')))
        metapath.write_text(json.dumps(meta, indent=2))
        print(json.dumps({k: meta[k] for k in ['status', 'wall_seconds', 'sampled_trajectories', 'deployment_selection']}, indent=2), flush=True)
    finally:
        if p.poll() is None:
            try:
                rpc(p, {'command': 'close'})
            except Exception:
                pass
            p.wait(timeout=10)


if __name__ == '__main__':
    main()
