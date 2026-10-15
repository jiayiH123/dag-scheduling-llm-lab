#!/usr/bin/env python3
"""Validate externally produced parallel a/w scores with the unchanged physical decoder."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from daglab.io import DEFAULT_DATA,read_records
from daglab.reference import CachedBudgetDecoder,core

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default=str(DEFAULT_DATA));ap.add_argument('--scores',required=True);ap.add_argument('--output',required=True);args=ap.parse_args();out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    cases={r['key']:r for r in read_records(args.data)};seen=set();rows=[]
    for line in Path(args.scores).read_text().splitlines():
        x=json.loads(line);key=x['id']
        if key in seen:raise ValueError('duplicate prediction')
        seen.add(key);inst=cases[key]['instance'];r=CachedBudgetDecoder(inst).rollout(x['scores'],'local',materialize=True);core.validate(inst,r['schedule'])
        rows.append({'id':key,'makespan':r['makespan'],'cost':r['cost'],'schedule':r['schedule'],'forward_count_status':'external-unverified; instrument your model, do not self-certify from this file'})
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps({'rows':rows,'coverage':'partial unless caller supplies every intended split ID'},indent=2));print(json.dumps({'validated_predictions':len(rows),'output':str(out)}))
if __name__=='__main__':main()
