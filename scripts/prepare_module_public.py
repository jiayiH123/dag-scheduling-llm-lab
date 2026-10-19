#!/usr/bin/env python3
"""Pinned public TRAIN/VAL pilot; reserved TEST files are never fetched."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from daglab.workflow_trace import load_trace
from daglab.reference import CachedBudgetDecoder, core, graph_features, teacher


def topology(raw):
    tasks = raw['workflow']['specification']['tasks']
    index = {t['id']: i for i, t in enumerate(tasks)}
    edges = [(i, index[v]) for i, t in enumerate(tasks) for v in t['children']]
    parents, children = [[] for _ in tasks], [[] for _ in tasks]
    for u, v in edges:
        parents[v].append(u); children[u].append(v)
    h = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    colors = [h([len(parents[i]), len(children[i])]) for i in range(len(tasks))]
    for _ in range(16):
        colors = [h([colors[i], sorted(colors[u] for u in parents[i]), sorted(colors[v] for v in children[i])]) for i in range(len(tasks))]
    return h([len(tasks), sorted(colors), sorted((colors[u], colors[v]) for u, v in edges)])


def platform(inst, pools, eta):
    bandwidth = inst['bandwidth']
    inst['speeds'], inst['cost_rates'], inst['capacities'] = [1, 2, 4] * (pools // 3), [1, 4, 16] * (pools // 3), [2] * pools
    inst['bandwidth'] = [[0 if p == q else bandwidth[p % 3][q % 3] or bandwidth[0][1] for q in range(pools)] for p in range(pools)]
    inst['budget'] = None
    info = core.prepare(inst)
    bmin = sum(info['min_cost'])
    bfast = sum(info['costs'][v][min(info['eligible'][v], key=lambda p: (info['duration'][v][p], p))] for v in range(info['n']))
    inst.update(bmin=bmin, bfast=bfast, budget=bmin + int(eta * (bfast - bmin)), budget_degenerate=bmin == bfast)
    return inst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sources', default=str(ROOT / 'data/module_public_sources_20261019.json'))
    ap.add_argument('--raw-dir', required=True); ap.add_argument('--download', action='store_true'); ap.add_argument('--output', required=True)
    args = ap.parse_args(); out, rawdir = Path(args.output), Path(args.raw_dir)
    auditpath = Path(str(out) + '.manifest.json')
    if out.exists() or auditpath.exists(): raise FileExistsError('choose new output paths')
    manifest = json.loads(Path(args.sources).read_text()); rawdir.mkdir(parents=True, exist_ok=True)
    sources, fingerprints = [], {'train': set(), 'val': set()}
    for entry in manifest['entries']:
        if entry['split'] not in ['train', 'val']: continue
        path = rawdir / entry['file']
        if not path.exists():
            if not args.download: raise FileNotFoundError(path)
            with urllib.request.urlopen(entry['source_url'], timeout=45) as response: path.write_bytes(response.read())
        blob = path.read_bytes(); oid = hashlib.sha1(b'blob ' + str(len(blob)).encode() + b'\0' + blob).hexdigest()
        if oid != entry['git_blob_sha1'] or len(blob) != entry['bytes']: raise AssertionError(('source Git object mismatch', entry['file']))
        raw = json.loads(blob); fingerprint = topology(raw); fingerprints[entry['split']].add(fingerprint)
        sources.append({**entry, 'sha256': hashlib.sha256(blob).hexdigest(), 'topology_fingerprint': fingerprint, 'tasks': len(raw['workflow']['specification']['tasks'])})
    if fingerprints['train'] & fingerprints['val']: raise AssertionError('topology fingerprint crosses TRAIN/VAL; revise protocol before training')
    started = time.perf_counter(); counts, actions = Counter(), 0; out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, 'wt') as handle:
        for entry in sources:
            for pools in manifest['pool_counts']:
                for profile in manifest['network_profiles']:
                    for eta in manifest['budget_etas']:
                        inst = platform(load_trace(rawdir / entry['file'], entry['file'], profile, eta, family=entry['family']), pools, eta)
                        inst.update(split_role=entry['split'] + '_public_development', previously_inspected=False, previously_inspected_at_selection=False,
                                    now_development=True, topology_fingerprint=entry['topology_fingerprint'], source_commit=manifest['pinned_commit'])
                        graph, _ = graph_features(inst)
                        key = f"{entry['split']}_public_{entry['file']}_{pools}_{profile}_{eta}"
                        record = {'key': key, 'instance': inst, 'graph': graph, 'reference': core.schedule(inst, 'HEFT')['makespan']}
                        if entry['split'] == 'train':
                            solution, decisions, seconds = teacher(inst, CachedBudgetDecoder(inst)); actions += len(decisions)
                            record.update(teacher=solution, teacher_decisions=decisions, teacher_seconds=seconds)
                        handle.write(json.dumps(record, allow_nan=False) + '\n'); counts[entry['split']] += 1
            print(json.dumps({'prepared': entry['file'], 'split': entry['split'], 'n': entry['tasks'], 'cumulative_configs': dict(counts)}), flush=True)
    audit = {'sources_manifest_sha256': hashlib.sha256(Path(args.sources).read_bytes()).hexdigest(), 'pinned_commit': manifest['pinned_commit'],
             'license_url': manifest['license_url'], 'source_entries': sources, 'configs': dict(counts),
             'base_traces': dict(Counter(r['split'] for r in sources)), 'unique_topology_fingerprints': {k: len(v) for k, v in fingerprints.items()},
             'TRAIN_VAL_topology_overlap': False, 'teacher_actions_TRAIN_only': actions, 'pool_counts': manifest['pool_counts'],
             'task_range': [min(r['tasks'] for r in sources), max(r['tasks'] for r in sources)], 'TEST_downloaded': False, 'TEST_evaluated': False,
             'dataset_sha256': hashlib.sha256(out.read_bytes()).hexdigest(), 'prepare_seconds': time.perf_counter() - started,
             'measurement_scope': 'measured topology/runtime/filebytes; simulated pools,unit-demand capacities,prices,network,budget',
             'classification': 'small public development pilot; no cross-application or final-test claim'}
    auditpath.write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps({k: v for k, v in audit.items() if k != 'source_entries'}, indent=2))


if __name__ == '__main__': main()
