#!/usr/bin/env python3
"""Independent portability audit; leaves the original strict checker intact.

Only selected_log_probability tolerates1e-12 numeric drift. Every action, mask,
offset, budget feature, teacher schedule and frozen-file SHA remains exact.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from daglab.io import read_records
from daglab.reference import CachedBudgetDecoder, core, graph_features, teacher


def compare(a, b, path='', differences=None):
    differences = [] if differences is None else differences
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            raise AssertionError(('field mismatch', path))
        for key in a:
            compare(a[key], b[key], path + '/' + key, differences)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            raise AssertionError(('length mismatch', path))
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, path + '/' + str(i), differences)
    elif path.endswith('/selected_log_probability'):
        if not math.isfinite(a) or not math.isfinite(b) or not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12):
            raise AssertionError(('probability exceeds declared tolerance', path, a, b))
        if a != b:
            differences.append({'path': path, 'absolute_difference': abs(a - b)})
    elif a != b:
        raise AssertionError(('exact field mismatch', path, a, b))
    return differences


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    files = json.loads((ROOT / 'BUNDLE_MANIFEST.json').read_text())['files']
    for entry in files:
        if hashlib.sha256((ROOT / entry['path']).read_bytes()).hexdigest() != entry['sha256']:
            raise AssertionError('frozen checksum mismatch: ' + entry['path'])
    records = read_records()
    groups = {s: [r for r in records if r['key'].startswith(s)] for s in ['train', 'val', 'test']}
    if {s: len(rows) for s, rows in groups.items()} != {'train': 800, 'val': 216, 'test': 432}:
        raise AssertionError('frozen split counts')
    latent = {s: {r['instance']['latent_id'] for r in rows} for s, rows in groups.items()}
    if any(latent[a] & latent[b] for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]):
        raise AssertionError('latent graph leakage')
    if sum(len(r['teacher_decisions']) for r in groups['train']) != 36540:
        raise AssertionError('teacher action count')
    if any('teacher_decisions' in r for s in ['val', 'test'] for r in groups[s]):
        raise AssertionError('labels on evaluation split')
    # Known reference replay, not a fresh TEST performance measurement.
    chosen = [groups['test'][0], groups['test'][3], groups['test'][12], groups['train'][0]]
    for r in chosen:
        if graph_features(r['instance'])[0] != r['graph']:
            raise AssertionError('feature identity')
        zero = CachedBudgetDecoder(r['instance']).rollout([[0., 0.]] * (r['graph']['n'] * r['graph']['k']),
                                                         'local', materialize=True)
        if zero['schedule']['tasks'] != core.schedule(r['instance'], 'HEFT')['tasks']:
            raise AssertionError('zero-head HEFT task identity')
    r = groups['train'][0]
    solution, decisions, _ = teacher(r['instance'], CachedBudgetDecoder(r['instance']))
    if solution['tasks'] != r['teacher']['tasks']:
        raise AssertionError('teacher schedule changed')
    differences = compare(decisions, r['teacher_decisions'])
    strict = subprocess.run([sys.executable, str(ROOT / 'scripts/check_bundle.py')], cwd=ROOT,
                            text=True, capture_output=True)
    result = {'frozen_SHA256_files': len(files), 'checksums': 'PASS',
              'semantic_audit': 'PASS_with_declared_logprob_tolerance',
              'original_strict_checker': 'PASS' if strict.returncode == 0 else 'FAIL',
              'original_strict_returncode': strict.returncode,
              'original_strict_stdout': strict.stdout, 'original_strict_stderr': strict.stderr,
              'tolerance': {'only_field': 'selected_log_probability', 'absolute': 1e-12, 'relative': 1e-12},
              'floating_differences': differences, 'known_reference_replay_cases': len(chosen),
              'new_method_TEST_evaluated': False}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
