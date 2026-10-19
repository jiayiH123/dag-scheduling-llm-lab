#!/usr/bin/env python3
"""Materialize selected TRAIN/VAL sparse records and TRAIN teachers without network."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from daglab.offline_data import dense_instance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', default=str(ROOT / 'data/offline_20261019'))
    ap.add_argument('--dataset', choices=['wfinstances', 'tpch_composite', 'tpch_primitive'], required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--config-output')
    ap.add_argument('--min-tasks', type=int, default=1)
    ap.add_argument('--max-tasks', type=int, default=512)
    ap.add_argument('--train-bases', type=int, default=0, help='0 = all eligible TRAIN bases')
    ap.add_argument('--val-bases', type=int, default=0, help='0 = all eligible VAL bases')
    ap.add_argument('--pools', default='3')
    ap.add_argument('--profiles', default='10_40_MBps')
    ap.add_argument('--budget-etas', default='0.25,0.6')
    args = ap.parse_args()
    out, bundle = Path(args.output), Path(args.bundle)
    auditpath = Path(str(out) + '.manifest.json')
    if out.exists() or auditpath.exists() or (args.config_output and Path(args.config_output).exists()):
        raise FileExistsError('choose new output paths')
    from daglab.reference import CachedBudgetDecoder, core, graph_features, teacher
    index = json.loads((bundle / 'INDEX.json').read_text())
    manifest = json.loads((bundle / 'MANIFEST.json').read_text())
    hashes = {row['path']: row['sha256'] for row in manifest['files']}
    selected = []
    for role, limit in [('train', args.train_bases), ('val', args.val_bases)]:
        candidates = sorted([b for b in index['bases'] if b['dataset'] == args.dataset and b['split'] == role and
                             args.min_tasks <= b['n'] <= args.max_tasks], key=lambda b: b['base_id'])
        selected.extend(candidates[:limit] if limit else candidates)
    if not any(b['split'] == 'train' for b in selected) or not any(b['split'] == 'val' for b in selected):
        raise ValueError('need nonempty TRAIN and VAL at the requested task range')
    pools = {int(x) for x in args.pools.split(',')}
    profiles = set(args.profiles.split(','))
    etas = {float(x) for x in args.budget_etas.split(',')}
    counts, teacher_actions, keys = Counter(), 0, []
    started = time.perf_counter()
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, 'wt') as handle:
        for number, base in enumerate(selected, 1):
            path = bundle / base['adapted_file']
            if hashlib.sha256(path.read_bytes()).hexdigest() != hashes[base['adapted_file']]:
                raise AssertionError('adapted source checksum')
            with gzip.open(path, 'rt') as rows:
                for line in rows:
                    row = json.loads(line)
                    s = row['instance_sparse']
                    if len(s['speeds']) not in pools or s['link_profile'] not in profiles or s['budget_eta'] not in etas:
                        continue
                    inst = dense_instance(row, args.max_tasks)
                    graph, _ = graph_features(inst)
                    record = {'key': row['key'], 'instance': inst, 'graph': graph,
                              'reference': core.schedule(inst, 'HEFT')['makespan']}
                    if base['split'] == 'train':
                        solution, decisions, seconds = teacher(inst, CachedBudgetDecoder(inst))
                        record.update(teacher=solution, teacher_decisions=decisions, teacher_seconds=seconds)
                        teacher_actions += len(decisions)
                    handle.write(json.dumps(record, allow_nan=False) + '\n')
                    counts[base['split']] += 1; keys.append(row['key'])
            if number % 10 == 0 or number == len(selected):
                print(json.dumps({'materialized_base': number, 'selected_bases': len(selected),
                                  'configs': dict(counts)}), flush=True)
    if not counts['train'] or not counts['val']:
        raise ValueError('profile filter selected no TRAIN/VAL configurations')
    audit = {'dataset': args.dataset, 'counts': dict(counts), 'selected_base_ids': [b['base_id'] for b in selected],
             'base_counts': dict(Counter(b['split'] for b in selected)), 'keys': keys,
             'task_range_requested': [args.min_tasks, args.max_tasks],
             'source_bundle_manifest_sha256': hashlib.sha256((bundle / 'MANIFEST.json').read_bytes()).hexdigest(),
             'dataset_sha256': hashlib.sha256(out.read_bytes()).hexdigest(),
             'teacher_actions_TRAIN_only': teacher_actions, 'TEST_loaded_for_use': False,
             'seconds': time.perf_counter() - started, 'network_requests': 0,
             'scope': 'offline data/teacher preparation, not trained policy results'}
    auditpath.write_text(json.dumps(audit, indent=2) + '\n')
    if args.config_output:
        config = json.loads((ROOT / 'configs/module_lab_public_scaled.json').read_text())
        config.update(classification='DRAFT training recipe for expanded offline data; lock a training protocol before running',
                      protocol_commit='UNLOCKED_TRAINING_RECIPE', train_limit=counts['train'], val_limit=counts['val'],
                      bc_updates=400, batch=8)
        configpath = Path(args.config_output); configpath.parent.mkdir(parents=True, exist_ok=True)
        configpath.write_text(json.dumps(config, indent=2) + '\n')
    print(json.dumps({k: v for k, v in audit.items() if k not in ['keys', 'selected_base_ids']}, indent=2))


if __name__ == '__main__':
    main()
