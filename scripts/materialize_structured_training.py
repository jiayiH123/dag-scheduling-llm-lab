#!/usr/bin/env python3
"""Offline bridge to the current small-model trainer; master data stays sparse."""
import argparse
import gzip
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from daglab.structured_dataset import iter_rows, to_instance


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data', required=True); ap.add_argument('--output', required=True)
    ap.add_argument('--train-limit', type=int, default=1000); ap.add_argument('--val-limit', type=int, default=100)
    ap.add_argument('--config-output'); args = ap.parse_args()
    from daglab.reference import graph_features
    out = Path(args.output)
    if out.exists() or (args.config_output and Path(args.config_output).exists()): raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True); counts = {}
    with gzip.open(out, 'wt') as f:
        for role, limit in [('train', args.train_limit), ('val', args.val_limit)]:
            count = 0
            for row in iter_rows(args.data, role, limit):
                inst = to_instance(row); graph, _ = graph_features(inst)
                record = dict(key=row['key'], instance=inst, graph=graph,
                              reference=row['quality']['native_makespans'][0] if role == 'train' else row['quality']['reference_HEFT'])
                if role == 'train':
                    s = row['supervision']; record['teacher_decisions'] = s['decisions']
                    # Existing simulator reads teacher tasks solely for exact replay equality.
                    from daglab.module_lab.decoder import NormalizedEFTDecoder
                    record['teacher'] = NormalizedEFTDecoder(inst).rollout([[0., 0.]]*(inst['n']*len(inst['speeds'])),
                        'local', forced_mapping=s['task_to_pool'], materialize=True)['schedule']
                f.write(json.dumps(record, allow_nan=False)+'\n'); count += 1
            if not count: raise ValueError('both TRAIN and VAL must be present')
            counts[role] = count
    if args.config_output:
        cfg = json.loads((ROOT/'configs/module_lab_public_scaled.json').read_text())
        cfg.update(classification='DRAFT structured-data training recipe; freeze training protocol before executing',
                   protocol_commit='UNLOCKED_TRAINING_RECIPE', train_limit=counts['train'], val_limit=counts['val'])
        p = Path(args.config_output); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(cfg, indent=2)+'\n')
    print(json.dumps(dict(counts=counts, TEST_loaded_for_training=False)))


if __name__ == '__main__': main()
