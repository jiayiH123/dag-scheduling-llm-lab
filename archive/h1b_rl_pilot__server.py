#!/usr/bin/env python3
"""Trusted local simulation process, stdlib + existing NumPy/OR-Tools."""
from copy import deepcopy
import gzip
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
import budget_core as core
import legacy_comm_core as legacy
spec=importlib.util.spec_from_file_location("exchange",ROOT/"experiments/h1_budget_exchange/run.py")
exchange=importlib.util.module_from_spec(spec);spec.loader.exec_module(exchange)


def add_budget(base, name):
    info=core.prepare(base)
    bmin=sum(info["min_cost"])
    fastest=[min(info["eligible"][v],key=lambda p:(info["duration"][v][p],info["costs"][v][p],p)) for v in range(info["n"])]
    bfast=sum(info["costs"][v][p] for v,p in enumerate(fastest))
    eta={"nonbinding":1,"tight25":.25,"tight60":.6}[name]
    base.update(budget=bmin+int(eta*(bfast-bmin)),budget_mode=name,bmin=bmin,bfast=bfast,budget_degenerate=bmin==bfast)
    return base


def graph_features(inst):
    started=time.perf_counter()
    info=core.prepare(inst);n,k=info["n"],info["k"]
    mean_work=sum(inst["workload"])/n
    mean_compute=sum(min(t) for t in info["duration"])/n
    mean_cost=sum(info["min_cost"])/n
    max_speed=max(inst["speeds"]);max_cap=max(inst["capacities"]);max_rate=max(inst["cost_rates"])
    eta=(inst["budget"]-inst["bmin"])/max(1,inst["bfast"]-inst["bmin"])
    x=[];eligible=[]
    for v in range(n):
        for p in range(k):
            x.append([inst["workload"][v]/mean_work,inst["demands"][v]/max_cap,
                      len(info["parents"][v])/n,len(info["children"][v])/n,
                      info["duration"][v][p]/mean_compute,info["costs"][v][p]/max(1,mean_cost),
                      inst["speeds"][p]/max_speed,inst["capacities"][p]/max_cap,
                      inst["cost_rates"][p]/max_rate,inst["demands"][v]/inst["capacities"][p],
                      info["ranks"][v]/max(info["ranks"]),eta])
            eligible.append(p in info["eligible"][v])
    src,dst,weight=[],[],[]
    for u,v in inst["edges"]:
        for p in info["eligible"][u]:
            for q in info["eligible"][v]:
                src.append(u*k+p);dst.append(v*k+q)
                weight.append(1/(1+info["comm"][u,v][p][q]/mean_compute))
    graph={"x":x,"eligible":eligible,"src":src,"dst":dst,"edge_weight":weight,"n":n,"k":k}
    return graph,time.perf_counter()-started


def rollout(inst, logits, mode, sample, seed):
    started=time.perf_counter();info=core.prepare(inst);state=legacy.blank_state(info["k"])
    rng=np.random.default_rng(seed);decisions=[]
    while len(state["tasks"])<info["n"]:
        ready=[v for v in range(info["n"]) if v not in state["tasks"] and all(u in state["tasks"] for u in info["parents"][v])]
        choices={}
        for v in ready:
            candidates=[]
            for p in info["eligible"][v]:
                if core.reserve_allows(info,state,v,p):
                    c=legacy.placement(inst,info,state,v,p)
                    candidates.append((c["task"]["end"],c["task"]["start"],p,c))
                    if mode=="joint":choices[v*info["k"]+p]=(v,p,c)
            if mode=="order" and candidates:
                _,_,p,c=min(candidates,key=lambda t:t[:3]);choices[v]=(v,p,c)
        available=sorted(choices)
        vals=np.asarray([logits[a] for a in available]);prob=np.exp(vals-vals.max());prob/=prob.sum()
        selected=int(rng.choice(len(available),p=prob)) if sample else int(np.argmax(vals))
        action=available[selected];v,p,c=choices[action]
        decisions.append({"available":available,"selected":selected,"action":action})
        state["tasks"][v]=c["task"];state["calendar"]=c["calendar"]
        state["communications"].extend(c["communications"]);state["decision_order"].append(v)
    sol=legacy.finish_schedule(inst,state,"RL-"+mode)
    sol["cost"]=sum(info["costs"][t["node"]][t["device"]] for t in sol["tasks"])
    core.validate(inst,sol)
    sol["runtime_seconds"]=time.perf_counter()-started
    return {"decisions":decisions,"makespan":sol["makespan"],"cost":sol["cost"],"schedule":sol,
            "decoder_seconds":sol["runtime_seconds"]}


def build(out, test_seed_offset=3000):
    out.mkdir(parents=True,exist_ok=True)
    records={};rng=random.Random(51627)
    for index in range(2000):
        family=rng.choice(["chain","balanced_forkjoin","asymmetric_merge","random_packing"])
        base=exchange.diagnostic.make_base(family,rng.randrange(5),rng.choice([0,.1,1,5]),rng.choice(["serial","pool"]),seed_offset=100000+10*index)
        inst=add_budget(base,rng.choice(["nonbinding","tight25","tight60"]))
        records[f"train{index}"]={"instance":inst,"reference":core.schedule(inst,"HEFT")["makespan"]}
    vals=list(exchange.instances(seed_offset=20000));indices=random.Random(98271).sample(range(360),120)
    for j,index in enumerate(indices):records[f"val{j}"]={"instance":vals[index],"reference":core.schedule(vals[index],"HEFT")["makespan"]}
    tests=list(exchange.instances(seed_offset=test_seed_offset))
    for index,inst in enumerate(tests):
        solutions=exchange.baselines(inst)
        exact=core.exact_solve(inst,solutions[-1],1)
        records[f"test{index}"]={"instance":inst,"reference":solutions[1]["makespan"],"baselines":solutions,"exact":exact}
        if (index+1)%90==0:print(f"RL pilot: prepared {index+1}/360 independent test baselines",file=sys.stderr,flush=True)
    for key,r in records.items():r["graph"],r["feature_seconds"]=graph_features(r["instance"])
    with gzip.open(out/"dataset.jsonl.gz","wt") as f:
        for key,r in records.items():f.write(json.dumps({"key":key,**r})+"\n")
    return records


def main():
    records={}
    for line in sys.stdin:
        request=json.loads(line);command=request["command"]
        if command=="init":
            records=build(Path(request["out"]), request.get("test_seed_offset",3000))
            result={"train_ids":[k for k in records if k.startswith("train")],
                    "val_ids":[k for k in records if k.startswith("val")],
                    "test_ids":[k for k in records if k.startswith("test")],
                    "global_scale":float(np.mean([r["reference"] for k,r in records.items() if k.startswith("train")]))}
        elif command=="get":
            result=[{"id":k,"graph":records[k]["graph"],"reference":records[k]["reference"],
                     "feature_seconds":records[k]["feature_seconds"]} for k in request["ids"]]
        elif command=="rollout":
            result=[]
            for item in request["items"]:
                value=rollout(records[item["id"]]["instance"],item["logits"],request["mode"],request["sample"],item["seed"])
                value["id"]=item["id"]
                if not request.get("include_schedule",False):value.pop("schedule")
                result.append(value)
        elif command=="close":
            print(json.dumps({"closed":True}),flush=True);break
        else:raise ValueError(command)
        print(json.dumps(result,allow_nan=False),flush=True)


if __name__=="__main__":main()
