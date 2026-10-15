#!/usr/bin/env python3
import argparse,gzip,itertools,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from daglab.io import read_records
from daglab.reference import FAMILIES,base,add_budget,graph_features,core

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed-offset',type=int,required=True);ap.add_argument('--replicates',type=int,default=2);ap.add_argument('--output',required=True);args=ap.parse_args();out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    if args.replicates<1:raise ValueError('replicates must be positive')
    known={r['instance']['latent_id'] for r in read_records()};count=0;latent=set();out.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(out,'wt') as f:
        for fi,(family,n,resource,rep) in enumerate(itertools.product(FAMILIES,[20,40,80],['serial','pool'],range(args.replicates))):
            seed=args.seed_offset+fi
            for ccr,budget in itertools.product([0,1,5],['nonbinding','tight25','tight60']):
                inst=add_budget(base(family,n,resource,seed,ccr),budget)
                if inst['latent_id'] in known:raise ValueError('overlap with bundled graphs')
                latent.add(inst['latent_id']);g,_=graph_features(inst);r={'key':'test'+str(count),'instance':inst,'graph':g,'reference':core.schedule(inst,'HEFT')['makespan']}
                f.write(json.dumps(r,allow_nan=False)+'\n');count+=1
    print(json.dumps({'configs':count,'latent_graphs':len(latent),'seed_offset':args.seed_offset,'labels_generated':False,'note':'Lock generator/offset before evaluating selected models; bundled-overlap check does not cover every historical discovery fold.'}))
if __name__=='__main__':main()
