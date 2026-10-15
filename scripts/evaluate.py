#!/usr/bin/env python3
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from daglab.io import DEFAULT_DATA,read_records
from daglab.reference import core,mslbl,portfolio,teacher,graph_features,CachedBudgetDecoder,random_policy_rollout
from daglab.metrics import paired

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default=str(DEFAULT_DATA));ap.add_argument('--split',choices=['val','test'],default='test');ap.add_argument('--limit',type=int);ap.add_argument('--checkpoint',default=str(ROOT/'checkpoints/local_seed11.pt'));ap.add_argument('--torch-python',default=os.getenv('TORCH_PYTHON',sys.executable));ap.add_argument('--teacher',action='store_true');ap.add_argument('--mode',choices=['initial','local'],default='local');ap.add_argument('--output',default='runs/reference_eval.json');args=ap.parse_args()
    out=Path(args.output)
    if out.exists():raise FileExistsError('refuses overwrite: '+str(out))
    records=read_records(args.data,args.split,args.limit);rows=[]
    p=subprocess.Popen([args.torch_python,'-m','daglab.actor_server','--checkpoint',args.checkpoint],cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
    try:
        for r in records:
            inst=r['instance'];graph,feature_seconds=graph_features(inst);start=time.perf_counter();decoder=CachedBudgetDecoder(inst);prepare_seconds=time.perf_counter()-start
            # Warm-up once, excluded from reported per-output calls/times.
            if not rows:
                p.stdin.write(json.dumps({'graph':graph})+'\n');p.stdin.flush();json.loads(p.stdout.readline())
            start=time.perf_counter();p.stdin.write(json.dumps({'graph':graph})+'\n');p.stdin.flush();a=json.loads(p.stdout.readline());actor_rpc=time.perf_counter()-start
            if a['network_calls']!=1:raise AssertionError('forward count')
            start=time.perf_counter();result=decoder.rollout(a['logits'],args.mode,materialize=True);decode_total=time.perf_counter()-start
            core.validate(inst,result['schedule'])
            row={'key':r['key'],'latent_id':inst['latent_id'],'n':inst['n'],'family':inst['family'],'ccr':inst['target_ccr'],'budget_mode':inst['budget_mode'],'student':result['makespan'],'network_calls':1,'feature_seconds':feature_seconds,'actor_rpc_seconds':actor_rpc,'forward_seconds':a['forward_seconds'],'decoder_total_seconds':decode_total,'decoder_prepare_seconds':prepare_seconds,'end_to_end_seconds':feature_seconds+prepare_seconds+actor_rpc+decode_total,'parameters':a['parameters']}
            timings={}
            for name in ['Greedy','HEFT','PEFT']:
                start=time.perf_counter();s=core.schedule(inst,name);timings[name]=time.perf_counter()-start;row[name]=s['makespan']
            start=time.perf_counter();s=mslbl(inst);timings['MSLBL']=time.perf_counter()-start;row['MSLBL']=s['makespan']
            start=time.perf_counter();ss,_=portfolio(inst);row['Portfolio']=min(s['makespan'] for s in ss);timings['Portfolio']=time.perf_counter()-start
            start=time.perf_counter();rs=[random_policy_rollout(inst,[0.]*(graph['n']*graph['k']),'joint',True,r.get('random_seed_base',131)+inst['seed']*17+j)['schedule'] for j in range(5)];row['RandomMean5']=sum(s['makespan'] for s in rs)/5;timings['RandomMean5']=time.perf_counter()-start
            if args.teacher:
                start=time.perf_counter();s,_,_=teacher(inst,decoder);row['Teacher64']=s['makespan'];timings['Teacher64']=time.perf_counter()-start
            row['baseline_wall_seconds']=timings;rows.append(row)
        names=['Greedy','HEFT','PEFT','MSLBL','Portfolio','RandomMean5']+(['Teacher64'] if args.teacher else [])
        result={'scope':'Retrospective known reference split; not a fresh final test','checkpoint':str(Path(args.checkpoint).name),'split':args.split,'decoder_mode':args.mode,'warmup_calls_excluded':1,'inference_policy':'one forward, one physical decode; CPU reference; preparation and RPC overhead included in end_to_end','comparisons':[paired(rows,n) for n in names],'rows':rows}
        out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2));print(json.dumps({'output':str(out),'comparisons':result['comparisons']},indent=2))
    finally:
        if p.poll() is None:p.stdin.write(json.dumps({'command':'close'})+'\n');p.stdin.flush();p.communicate(timeout=15)
if __name__=='__main__':main()
