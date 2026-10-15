#!/usr/bin/env python3
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from daglab.io import read_records
from daglab.reference import core,CachedBudgetDecoder,graph_features,teacher

def main():
    manifest=json.loads((ROOT/'BUNDLE_MANIFEST.json').read_text())
    for x in manifest['files']:
        p=ROOT/x['path']
        if hashlib.sha256(p.read_bytes()).hexdigest()!=x['sha256']:raise AssertionError('checksum mismatch: '+x['path'])
    records=read_records();groups={s:[r for r in records if r['key'].startswith(s)] for s in ['train','val','test']}
    assert {k:len(v) for k,v in groups.items()}=={'train':800,'val':216,'test':432}
    latent={s:{r['instance']['latent_id'] for r in rows} for s,rows in groups.items()}
    assert all(not latent[a]&latent[b] for a,b in [('train','val'),('train','test'),('val','test')])
    labels=sum(len(r['teacher_decisions']) for r in groups['train']);assert labels==36540
    assert all('teacher_decisions' not in r for s in ['val','test'] for r in groups[s])
    chosen=[groups['test'][0],groups['test'][3],groups['test'][12],groups['train'][0]]
    for r in chosen:
        inst=r['instance'];g,_=graph_features(inst);assert g==r['graph']
        d=CachedBudgetDecoder(inst);z=d.rollout([[0.,0.]]*(g['n']*g['k']),'local',materialize=True)
        h=core.schedule(inst,'HEFT');assert z['schedule']['tasks']==h['tasks'];core.validate(inst,z['schedule'])
    t=groups['train'][0];s,decisions,_=teacher(t['instance'],CachedBudgetDecoder(t['instance']))
    assert s['tasks']==t['teacher']['tasks'] and decisions==t['teacher_decisions']
    print(json.dumps({'checksummed_files':len(manifest['files']),'configs':{k:len(v) for k,v in groups.items()},'latent_graphs':{k:len(v) for k,v in latent.items()},'teacher_actions':labels,'zero_head_HEFT_identity_cases':len(chosen),'teacher_recipe_exact_replay_cases':1,'status':'PASS'}))
if __name__=='__main__':main()
