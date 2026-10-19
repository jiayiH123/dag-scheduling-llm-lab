#!/usr/bin/env python3
"""Audit raw versus unique physical configs, without scheduling held-out data."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from daglab.structured_dataset import iter_rows, sha256
from daglab.structured_unique import configuration_key, iter_unique_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    raw, unique, removed, raw_actions, unique_actions = [Counter() for _ in range(5)]
    pairs, pair_kinds, seen, duplicate_teacher_differences = Counter(), Counter(), {}, 0
    for row in iter_rows(args.data):
        role = row['split']; raw[role] += 1
        actions = len(row.get('supervision', {}).get('task_to_pool', []))
        raw_actions[role] += actions
        # Budget mode/eta/min/fast/max are derived annotations, not extra limits.
        x = dict(row['input']); x['budget'] = {'limit': x['budget']['limit']}
        physical = hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        key = configuration_key(row)
        target = tuple(row.get('supervision', {}).get('task_to_pool', []))
        if key in seen:
            old_role, old_physical, old_target = seen[key]
            assert role == old_role and physical == old_physical, 'dedup key is not physically equivalent'
            removed[role] += 1
            pair_kinds[str(row['input']['budget']['eta'])] += 1
            if target != old_target:
                duplicate_teacher_differences += 1
        else:
            seen[key] = (role, physical, target)
            unique[role] += 1; unique_actions[role] += actions
        pairs[row['base_id']] += 1
    assert set(pairs.values()) == {2}
    # Independently exercise the delivered stream and verify first-row selection.
    actual_counts, actual_actions = Counter(), Counter()
    for row in iter_unique_rows(args.data):
        role = row['split']; actual_counts[role] += 1
        target = tuple(row.get('supervision', {}).get('task_to_pool', []))
        assert target == seen[configuration_key(row)][2]
        actual_actions[role] += len(target)
    assert actual_counts == unique and actual_actions == unique_actions
    report = dict(status='PASS', data_manifest_sha256=sha256(Path(args.data)/'MANIFEST.json'),
                  scope='data identity and deterministic first-row view; no learned-policy or TEST scheduling result',
                  selection='first serialized row; never best-of-two labels or quality',
                  independent_base_DAGs=len(pairs), raw_records=dict(raw), unique_physical_configurations=dict(unique),
                  removed_repeated_configurations=dict(removed), raw_action_labels=dict(raw_actions),
                  unique_view_action_labels=dict(unique_actions), repeated_second_eta=dict(pair_kinds),
                  repeated_TRAIN_configs_with_different_teacher_mappings=duplicate_teacher_differences,
                  raw_data_modified=False, heldout_labels_or_schedules_generated=False)
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
