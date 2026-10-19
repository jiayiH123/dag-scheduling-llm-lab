#!/usr/bin/env python3
"""Resumable ordinary-Git shards for the locked structured simulation protocol."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import io
import json
import multiprocessing
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from daglab.structured_dataset import sha256, numeric_view
from daglab.structured_synthetic import base_index, build_base_job


def emit(value):
    print(json.dumps(value, separators=(',', ':')), flush=True)


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n'); tmp.replace(path)


def write_rows(path, rows):
    tmp = Path(str(path)+'.tmp'); path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0, compresslevel=6) as z:
            with io.TextIOWrapper(z, encoding='utf-8', newline='\n') as f:
                for row in rows:
                    f.write(json.dumps(row, separators=(',', ':'), allow_nan=False)+'\n')
    tmp.replace(path)


def summarize(rows):
    labels = 0; ratios = []; paired = []; mslbl = []; wins = 0; equal = 0
    positions = []; ns = []; ks = []; groups = {}
    for r in rows:
        ns.append(r['input']['n']); ks.append(r['input']['k']); positions.append(numeric_view(r['input'])['position_count'])
        if 'supervision' not in r: continue
        s, q = r['supervision'], r['quality']; labels += len(s['decisions'])
        heft, ms = q['native_makespans'][0], q['native_makespans'][4]
        ratios.append(s['makespan']/heft); paired.append(100*(heft-s['makespan'])/heft)
        mslbl.append(100*(ms-s['makespan'])/ms)
        wins += s['makespan'] < heft; equal += s['makespan'] == heft
        key = '|'.join(map(str, [r['meta']['family'], r['meta']['target_ccr'], r['input']['budget']['mode']]))
        g = groups.setdefault(key, dict(count=0, ratio_sum=0., gain_sum=0.))
        g['count'] += 1; g['ratio_sum'] += ratios[-1]; g['gain_sum'] += paired[-1]
    return dict(rows=len(rows), teacher_actions=labels, max_numeric_positions=max(positions),
                min_tasks=min(ns), max_tasks=max(ns), pool_counts=dict(Counter(ks)),
                quality=dict(count=len(ratios), teacher_heft_ratio_sum=sum(ratios),
                    teacher_heft_gain_percent_sum=sum(paired), teacher_mslbl_gain_percent_sum=sum(mslbl),
                    teacher_better_HEFT=wins, teacher_equal_HEFT=equal, groups=groups))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(ROOT/'configs/structured_scheduling_v1.json'))
    ap.add_argument('--output', required=True)
    ap.add_argument('--stage', choices=['engineering', 'pilot', 'full'], default='pilot')
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args(); started = time.perf_counter()
    cfg = json.loads(Path(args.config).read_text())
    if args.stage == 'engineering': cfg['counts'] = {role: 20 for role in cfg['counts']}
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    config_bytes = (json.dumps(cfg, sort_keys=True, separators=(',', ':'))+'\n').encode()
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    files = ['daglab/structured_dataset.py', 'daglab/structured_synthetic.py', 'scripts/build_structured_dataset.py',
             'daglab/reference.py', 'daglab/module_lab/decoder.py', 'vendor/budget_core.py',
             'vendor/search_controls.py', 'vendor/prepared_sgs.py', 'vendor/legacy_comm_core.py', 'vendor/budget_controls.py']
    source_hashes = {f: sha256(ROOT/f) for f in files}
    path = out/'MANIFEST.json'
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest['config_sha256'] != config_hash or manifest['source_hashes'] != source_hashes:
            raise ValueError('config/source changed; choose new output and preserve old attempt')
        for item in manifest['shards']:
            if sha256(out/item['path']) != item['sha256']:
                raise AssertionError('existing shard corrupted')
        with gzip.open(out/'BASES.jsonl.gz', 'rt') as f: bases = [json.loads(line) for line in f]
    else:
        if any(out.iterdir()): raise FileExistsError('nonempty output without a manifest')
        bases = base_index(cfg, emit)
        write_rows(out/'BASES.jsonl.gz', bases)
        (out/'CONFIG.json').write_bytes(config_bytes)
        manifest = dict(schema='structured-scheduling-delivery-v1', classification='engineering' if args.stage == 'engineering' else 'locked synthetic data; no trained-model results',
                        config_sha256=config_hash, source_hashes=source_hashes,
                        build_code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                        protocol='protocols/structured_scheduling_v1_20261019.md',
                        files=[dict(path=f, bytes=(out/f).stat().st_size, sha256=sha256(out/f)) for f in ['CONFIG.json', 'BASES.jsonl.gz']],
                        all_role_base_counts=dict(Counter(b['split'] for b in bases)),
                        topology_duplicates_rejected=sum(b['duplicate_attempts'] for b in bases),
                        shards=[], build_sessions=[], status='index_ready',
                        neural_optimizer_steps=0, model_TEST_evaluated=False, external_downloads=0)
        atomic_json(path, manifest)
    completed = {a['path'] for a in manifest['shards']}
    per_shard = cfg['rows_per_shard']//2
    selected = [b for b in bases if args.stage != 'pilot' or
                (b['split'] == 'train' and b['index'] < cfg['pilot_train_bases'])]
    context = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        for role in cfg['counts']:
            role_bases = [b for b in selected if b['split'] == role]
            for start in range(0, len(role_bases), per_shard):
                group = role_bases[start:start+per_shard]
                rel = f"{role}/part-{start//per_shard:05d}.jsonl.gz"
                if rel in completed: continue
                rows = []; worker_seconds = 0.
                for result, seconds in pool.map(build_base_job, [(b, cfg) for b in group], chunksize=1):
                    rows.extend(result); worker_seconds += seconds
                write_rows(out/rel, rows)
                stat = summarize(rows)
                item = dict(path=rel, split=role, bases=len(group), bytes=(out/rel).stat().st_size,
                            sha256=sha256(out/rel), worker_seconds=worker_seconds, **stat)
                if item['bytes'] >= 50*1024**2:
                    raise AssertionError('shard exceeds ordinary Git size budget')
                manifest['shards'].append(item); completed.add(rel)
                manifest['status'] = 'building'; atomic_json(path, manifest)
                emit(dict(shard=rel, records=stat['rows'], teacher_actions=stat['teacher_actions'],
                          completed_records=sum(s['rows'] for s in manifest['shards']),
                          elapsed_seconds=round(time.perf_counter()-started, 2)))
    manifest['build_sessions'].append(dict(stage=args.stage, wall_seconds=time.perf_counter()-started, workers=args.workers))
    counts = Counter(); labels = 0
    for s in manifest['shards']: counts[s['split']] += s['rows']; labels += s['teacher_actions']
    manifest.update(record_counts=dict(counts), TRAIN_teacher_actions=labels,
                    status='pilot_ready' if args.stage == 'pilot' else 'complete')
    atomic_json(path, manifest)
    emit(dict(status=manifest['status'], counts=dict(counts), TRAIN_teacher_actions=labels,
              bytes=sum(s['bytes'] for s in manifest['shards']), max_numeric_positions=max(s['max_numeric_positions'] for s in manifest['shards'])))


if __name__ == '__main__': main()
