#!/usr/bin/env python3
"""Standard-library integrity, split, numeric-prefix and TRAIN-label validation."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from daglab.structured_dataset import sha256, validate_record
from daglab.offline_data import topology_fingerprint


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data', required=True); ap.add_argument('--output')
    args = ap.parse_args(); root = Path(args.data); start = time.perf_counter()
    manifest = json.loads((root/'MANIFEST.json').read_text())
    for f in manifest['files']+manifest['shards']:
        p = root/f['path']
        if p.stat().st_size != f['bytes'] or sha256(p) != f['sha256']:
            raise AssertionError(('file hash/size', str(p)))
    with gzip.open(root/'BASES.jsonl.gz', 'rt') as f: bases = [json.loads(line) for line in f]
    ids = {b['base_id']: b for b in bases}; seen_fp = set()
    for b in bases:
        if b['topology_fingerprint'] in seen_fp: raise AssertionError('duplicate topology base')
        seen_fp.add(b['topology_fingerprint'])
    counts = Counter(); base_roles = {}; keys = set(); action_count = 0; maxpos = 0
    actual_topologies = set(); physical_pairs = {}; per_base_counts = Counter()
    sizes = Counter(); budgets = Counter(); cr = Counter(); pools = Counter(); ccr_error = 0.
    for shard in manifest['shards']:
        rows = 0; labels = 0
        with gzip.open(root/shard['path'], 'rt') as f:
            for line in f:
                row = json.loads(line); rows += 1
                if row['key'] in keys: raise AssertionError('duplicate instance key')
                keys.add(row['key']); b = ids[row['base_id']]
                if b['split'] != row['split'] or b['topology_fingerprint'] != row['meta']['topology_fingerprint']:
                    raise AssertionError('base provenance')
                if row['base_id'] not in base_roles:
                    fp = topology_fingerprint(row['input']['n'], [e[:2] for e in row['input']['network']['edges']])
                    if fp != b['topology_fingerprint'] or fp in actual_topologies:
                        raise AssertionError('serialized topology hash/overlap')
                    actual_topologies.add(fp)
                # Exactly the same physical instance, apart from the paired budget.
                import hashlib
                physical = {k: v for k, v in row['input'].items() if k != 'budget'}
                ph = hashlib.sha256(json.dumps(physical, sort_keys=True).encode()).hexdigest()
                if row['base_id'] in physical_pairs and physical_pairs[row['base_id']] != ph:
                    raise AssertionError('paired budget changed other physics')
                physical_pairs[row['base_id']] = ph; per_base_counts[row['base_id']] += 1
                base_roles[row['base_id']] = row['split']; maxpos = max(maxpos, validate_record(row))
                counts[row['split']] += 1
                labels += len(row.get('supervision', {}).get('decisions', []))
                x = row['input']; role = row['split']
                sizes[f"{role}:{x['n']}"] += 1; pools[f"{role}:{x['k']}"] += 1
                budgets[f"{role}:{x['budget']['mode']}"] += 1; cr[f"{role}:{row['meta']['target_ccr']}"] += 1
                ccr_error = max(ccr_error, abs(row['meta']['actual_ccr']-row['meta']['target_ccr']))
        if rows != shard['rows'] or labels != shard['teacher_actions']:
            raise AssertionError('shard counts')
        action_count += labels
    if dict(counts) != manifest['record_counts'] or action_count != manifest['TRAIN_teacher_actions']:
        raise AssertionError('total counts')
    if any(v != 2 for v in per_base_counts.values()): raise AssertionError('two configurations per base')
    if manifest['status'] == 'complete':
        expected = {role: 2*count for role, count in manifest['all_role_base_counts'].items()}
        if dict(counts) != expected: raise AssertionError('incomplete main data')
    result = dict(status='PASS', scope='data/labels/integrity only, no learned-model quality',
                  record_counts=dict(counts), base_counts=dict(Counter(base_roles.values())),
                  TRAIN_teacher_actions=action_count, topology_overlap=False, max_numeric_positions=maxpos,
                  max_absolute_CCR_rounding_error=ccr_error, task_counts=dict(sizes), pool_counts=dict(pools),
                  budget_counts=dict(budgets), ccr_counts=dict(cr), neural_optimizer_steps=0,
                  model_TEST_evaluated=False, source_manifest_sha256=sha256(root/'MANIFEST.json'),
                  seconds=time.perf_counter()-start)
    if args.output:
        p = Path(args.output); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if not k.endswith('_counts') or k in ['record_counts', 'base_counts']}, indent=2))


if __name__ == '__main__': main()
