#!/usr/bin/env python3
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
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/"src"))
from budget_feedback_actor import BudgetFeedbackActor

def rpc(p,q):
    p.stdin.write(json.dumps(q)+"\n");p.stdin.flush();line=p.stdout.readline()
    if not line:raise RuntimeError("simulator exited")
    return json.loads(line)

def conditionals(x,decisions,mode):
    logs=[];entropies=[]
    for d in decisions:
        pairs=x[d["available"]];phi=torch.tensor(d["budget_"+mode],dtype=x.dtype)
        score=pairs[:,0]+pairs[:,1]*phi+torch.tensor(d["offsets"],dtype=x.dtype);logp=torch.log_softmax(score,dim=0)
        logs.append(logp[d["selected"]]);entropies.append(-(logp.exp()*logp).sum())
    return torch.stack(logs),torch.stack(entropies)

def evaluate(p,model,mode,ids):
    rows=[];model.eval()
    for offset in range(0,len(ids),40):
        gs=rpc(p,{"command":"get","ids":ids[offset:offset+40]});items=[];timings={}
        with torch.no_grad():
            for g in gs:
                before=model.forward_calls;started=time.perf_counter();x=model(g["graph"]);elapsed=time.perf_counter()-started
                if model.forward_calls-before!=1:raise AssertionError("forward count")
                timings[g["id"]]={"forward_seconds":elapsed,"feature_seconds":g["feature_seconds"],"reference":g["reference"],"coefficient_abs_mean":float(x[:,1].abs().mean())}
                items.append({"id":g["id"],"logits":x.tolist(),"seed":0})
        started=time.perf_counter();rs=rpc(p,{"command":"rollout","mode":mode,"sample":False,"include_schedule":True,"items":items});rpc_time=time.perf_counter()-started
        for r in rs:
            r.update(timings[r["id"]]);r.update(network_calls=1,batch_rpc_seconds=rpc_time,component_total_seconds=r["decoder_seconds"]+r["forward_seconds"]+r["feature_seconds"]);rows.append(r)
    return rows

def validation(p,model,mode,ids):return sum(r["makespan"]/r["reference"] for r in evaluate(p,model,mode,ids))/len(ids)

def load_prior(seed,stage):
    torch.manual_seed(seed);model=BudgetFeedbackActor();path=ROOT/f"experiments/gnn_metric_pilot/results/run001/{stage}_seed{seed}.pt"
    checkpoint=torch.load(path,weights_only=True);missing,extra=model.load_state_dict(checkpoint["state_dict"],strict=False)
    if extra or any(not k.startswith("budget_head.") for k in missing):raise AssertionError("unexpected checkpoint mismatch")
    return model

def train_one(p,init,seed,mode,out,bc_anchor=.02):
    model=load_prior(seed,"bc");rng=random.Random(seed);opt=torch.optim.Adam(model.parameters(),lr=.0003)
    best=validation(p,model,mode,init["val_ids"]);best_state=deepcopy(model.state_dict());best_step=0;curve=[];started=time.perf_counter();updates_done=0
    for update in range(1,401):
        if time.perf_counter()-started>=120:break
        gs=rpc(p,{"command":"get","ids":rng.sample(init["train_ids"],16)});model.train()
        with torch.no_grad():xs=[model(g["graph"]) for g in gs]
        items=[{"id":g["id"],"logits":x.tolist(),"seed":seed*1000000+update*100+j} for j,(g,x) in enumerate(zip(gs,xs))]
        sampled=rpc(p,{"command":"rollout","mode":mode,"sample":True,"items":items});greedy=rpc(p,{"command":"rollout","mode":mode,"sample":False,"items":items})
        oldlogs=[torch.tensor([d["selected_log_probability"] for d in r["decisions"]],dtype=torch.float32) for r in sampled]
        advantages=[(b["makespan"]-r["makespan"])/g["reference"] for g,r,b in zip(gs,sampled,greedy)]
        lr=.0003*min(1,update/20) if update<=20 else .00003+.5*(.0003-.00003)*(1+math.cos(math.pi*(update-20)/380))
        for group in opt.param_groups:group["lr"]=lr
        epochs=0;last_kl=0;clip_fraction=0
        for epoch in range(4):
            losses=[];kls=[];clips=[]
            for g,r,old,adv in zip(gs,sampled,oldlogs,advantages):
                x=model(g["graph"]);logs,entropy=conditionals(x,r["decisions"],mode);delta=logs-old;ratio=delta.exp()
                pg=-torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv).sum()
                teacher,_=conditionals(x,g["teacher_decisions"],mode);bc=-teacher.mean()
                losses.append(pg+bc_anchor*bc-.001*(1-update/400)*entropy.sum());kls.append(((ratio-1)-delta).mean().detach());clips.append(((ratio<.8)|(ratio>1.2)).float().mean().detach())
            last_kl=float(torch.stack(kls).mean());clip_fraction=float(torch.stack(clips).mean())
            if last_kl>.03:break
            loss=torch.stack(losses).mean()
            if not torch.isfinite(loss):raise ValueError("nonfinite loss")
            opt.zero_grad(set_to_none=True);loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1));opt.step();epochs+=1
            if not math.isfinite(grad):raise ValueError("nonfinite gradient")
        updates_done=update;entry={"seed":seed,"mode":mode,"update":update,"epochs":epochs,"approx_kl":last_kl,"clip_fraction":clip_fraction,"lr":lr,"wall_seconds":time.perf_counter()-started}
        if update%100==0:
            value=validation(p,model,mode,init["val_ids"]);entry["val_C_over_HEFT"]=value
            if value<best:best=value;best_state=deepcopy(model.state_dict());best_step=update
            print(f"PPO {mode} seed{seed} step{update} valratio{value:.4f} best{best:.4f} {time.perf_counter()-started:.1f}s",flush=True)
        curve.append(entry)
    if updates_done%100:
        value=validation(p,model,mode,init["val_ids"])
        if value<best:best=value;best_state=deepcopy(model.state_dict());best_step=updates_done
    model.load_state_dict(best_state);detail={"seed":seed,"stage":"PPO-"+mode,"mode":mode,"selected_step":best_step,"val_ratio":best,"updates_done":updates_done,"training_seconds":time.perf_counter()-started,"bc_anchor":bc_anchor}
    torch.save({"state_dict":best_state,**detail},out/f"{mode}_seed{seed}.pt");return model,detail,curve

def main():
    out=ROOT/"experiments/budget_ppo/results/run001"
    if (out/"metadata.json").exists():raise RuntimeError("fresh output required")
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    paths=[Path(__file__),Path(__file__).with_name("server.py"),Path(__file__).with_name("protocol.md"),ROOT/"src/budget_feedback_actor.py",ROOT/"experiments/gnn_metric_pilot/results/run001/dataset.jsonl.gz"]+[ROOT/f"experiments/gnn_metric_pilot/results/run001/{stage}_seed{seed}.pt" for stage in ["bc","rl"] for seed in [11,22,33]]
    meta={"classification":"EXPLORATORY budget-state controlled PPO pilot","started":datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
          "commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"hashes":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
          "seeds":[11,22,33],"arms":["initial","dynamic"],"updates_max":400,"epochs_max":4,"batch_size":16,"arm_wall_limit_seconds":120,"torch":torch.__version__,"device":"cpu","threads":1}
    (out/"metadata.json").write_text(json.dumps(meta,indent=2));p=subprocess.Popen(["/opt/miniconda3/envs/dag-research/bin/python",str(Path(__file__).with_name("server.py"))],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1);started=time.perf_counter()
    try:
        init=rpc(p,{"command":"init","out":str(out)});models=[];curves=[]
        for seed in meta["seeds"]:
            for mode in meta["arms"]:
                model,detail,curve=train_one(p,init,seed,mode,out);models.append((model,detail));curves+=curve
                (out/"training_curves.json").write_text(json.dumps(curves,indent=2))
        # Retest frozen earlier baselines on the SAME new test, no reselection.
        for seed in meta["seeds"]:
            for stage in ["bc","rl"]:models.append((load_prior(seed,stage),{"seed":seed,"stage":"G1-"+stage,"mode":"dynamic","selected_step":"frozen_G1","val_ratio":None}))
        with (out/"evaluation.jsonl").open("w") as f:
            for model,detail in models:
                for fold in ["test","trace"]:
                    for r in evaluate(p,model,detail["mode"],init[fold+"_ids"]):f.write(json.dumps({**detail,"fold":fold,**r},allow_nan=False)+"\n")
        meta.update(models=[d for m,d in models],wall_seconds=time.perf_counter()-started,finished=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                    parameters=sum(v.numel() for v in models[0][0].parameters()),rl_sampled_trajectories=sum(d["updates_done"]*16 for m,d in models if "updates_done" in d))
        (out/"metadata.json").write_text(json.dumps(meta,indent=2));print(json.dumps({k:meta[k] for k in ["wall_seconds","parameters","rl_sampled_trajectories","models"]},indent=2),flush=True)
    finally:
        if p.poll() is None:
            try:rpc(p,{"command":"close"})
            except Exception:pass
            p.wait(timeout=10)

if __name__=="__main__":main()
