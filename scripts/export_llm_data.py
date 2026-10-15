#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from daglab.io import DEFAULT_DATA,read_records,model_input
from daglab.reference import mapping_of

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default=str(DEFAULT_DATA));ap.add_argument('--split',choices=['train','val','test'],default='train');ap.add_argument('--limit',type=int);ap.add_argument('--output',required=True);args=ap.parse_args();out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True);count=0
    with out.open('w') as f:
        for r in read_records(args.data,args.split,args.limit):
            item={'id':r['key'],'latent_id':r['instance']['latent_id'],'input':model_input(r['instance'])}
            if args.split=='train':
                item['targets']={'task_to_pool':mapping_of(r['teacher']),'expert_decisions':r['teacher_decisions'],'provenance':'heuristic-portfolio plus64remap attempts,not OPT','order':'fixed HEFT construction; no improved-order target'}
            f.write(json.dumps(item,ensure_ascii=False,allow_nan=False)+'\n');count+=1
    print(json.dumps({'split':args.split,'examples':count,'labels_exported':args.split=='train','output':str(out)}))
if __name__=='__main__':main()
