#!/usr/bin/env python3
"""Build the complete pinned offline source+physical-data bundle, without models."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from daglab.offline_data import digest, physical_variants, validate_sparse, workflow_base
from daglab.offline_tpch import COMMIT as TPCH_COMMIT, query_bases, query_roles, make_compositions


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def source_entries(root, entries):
    result = []
    for e in entries:
        blob = (root / e['path']).read_bytes()
        oid = hashlib.sha1(b'blob ' + str(len(blob)).encode() + b'\0' + blob).hexdigest()
        if oid != e['sha'] or len(blob) != e['size']:
            raise ValueError(('source object integrity', e['path']))
        result.append({**e, 'sha256': hashlib.sha256(blob).hexdigest()})
    return result


def archive_source(root, entries, out, temp, prefix):
    temp.parent.mkdir(parents=True, exist_ok=True)
    with temp.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0, compresslevel=6) as z:
        with tarfile.open(fileobj=z, mode='w|') as archive:
            for e in entries:
                info = tarfile.TarInfo(prefix + '/' + e['path'])
                info.size, info.mode, info.mtime = e['size'], 0o644, 0
                with (root / e['path']).open('rb') as handle:
                    archive.addfile(info, handle)
    pieces = []
    with temp.open('rb') as handle:
        number = 0
        while True:
            chunk = handle.read(32 * 1024 * 1024)
            if not chunk:
                break
            dest = out / f'source.tar.gz.part{number:03}'
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(chunk); pieces.append(str(dest.relative_to(out.parents[1])))
            number += 1
    return {'parts': pieces, 'archive_sha256': hashlib.sha256(temp.read_bytes()).hexdigest(),
            'archive_bytes': temp.stat().st_size, 'source_files': entries, 'prefix': prefix}


def wf_roles(bases):
    old = json.loads((ROOT / 'data/module_public_sources_20261019.json').read_text())
    smoke = json.loads((ROOT / 'data/public_smoke_sources.json').read_text())
    explicit = {e['path']: ('test' if e['split'].startswith('test') else e['split']) for e in old['entries']}
    seen = {e['path'] for e in old['entries'] if e['split'] in ['train', 'val']}
    smoke_paths = set()
    for e in smoke['entries']:
        url = e['source_url']; path = url.split('/' + old['pinned_commit'] + '/', 1)[1]
        seen.add(path)
        smoke_paths.add(path)
    forced = {}
    for b in bases:
        if b['source_path'] in explicit:
            role = explicit[b['source_path']]; fp = b['topology_fingerprint']
            if fp in forced and forced[fp] != role:
                # Historical use takes priority over a metadata-only TEST reservation.
                pair = {forced[fp], role}
                if pair == {'train', 'val'}:
                    raise ValueError('historical TRAIN/VAL topology conflict')
                role = next(x for x in pair if x != 'test')
            forced[fp] = role
    # Historical smoke has no formal TRAIN/VAL role; keep it in development,
    # preferring the explicit prior pilot role of any matching topology.
    for b in bases:
        if b['source_path'] in smoke_paths:
            forced.setdefault(b['topology_fingerprint'], 'val')
    for b in bases:
        fp = b['topology_fingerprint']; fraction = int(digest(['wf-split-20261019', fp])[:8], 16) / 2**32
        b['split'] = forced.get(fp, 'train' if fraction < .7 else 'val' if fraction < .85 else 'test')
        if b['source_kind'] == 'synthetic_example':
            b['split'] = 'example'
        b['previously_inspected'] = b['source_path'] in seen
    grouped = defaultdict(set)
    for b in bases:
        if b['split'] != 'example':
            grouped[b['topology_fingerprint']].add(b['split'])
    if any(len(roles) > 1 for roles in grouped.values()):
        raise AssertionError('cross-split topology fingerprint')
    return {'rule': 'directed WL16 groups, hash70/15/15; prior seen TRAIN/VAL preserved; toy examples separate',
            'forced_prior_group_roles': forced, 'prior_seen_source_paths': sorted(seen),
            'TEST_downloaded_by_explicit_user_request': True, 'TEST_model_evaluated': False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wfinstances', required=True)
    ap.add_argument('--wf-tree', required=True)
    ap.add_argument('--tpch', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--temp', required=True)
    args = ap.parse_args()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError('choose a fresh complete bundle directory')
    out.mkdir(parents=True)
    wfroot, tpchroot = Path(args.wfinstances), Path(args.tpch)
    tree = json.loads(Path(args.wf_tree).read_text())
    if tree['truncated'] or tree['sha'] != 'd7bfead08a4dbf985ebbf427b3685a6512c259e2':
        raise ValueError('unexpected/incomplete WF source tree')
    wf_files = source_entries(wfroot, [e for e in tree['tree'] if e['type'] == 'blob'])
    wf = []
    for e in wf_files:
        if e['path'].endswith('.json'):
            wf.append(workflow_base(json.loads((wfroot / e['path']).read_bytes()), e['path'], e['sha256']))
            if len(wf) % 30 == 0:
                print(json.dumps({'wfinstances_converted': len(wf)}), flush=True)
    if len(wf) != 318:
        raise ValueError('missing WfInstances source')
    split_info = wf_roles(wf)
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=tpchroot, text=True).strip()
    if actual != TPCH_COMMIT:
        raise ValueError('unexpected author TPC-H commit')
    lines = subprocess.check_output(['git', 'ls-tree', '-r', '-l', 'HEAD', 'dag_data/tpch',
                                     'README.md', 'dag_data/dag_generator.py', 'dag_data/dag_dict.py',
                                     'dag_data/utils.py'], cwd=tpchroot, text=True).splitlines()
    entries = []
    for line in lines:
        meta, name = line.split('\t', 1); mode, kind, oid, size = meta.split()
        entries.append({'path': name, 'mode': mode, 'type': kind, 'sha': oid, 'size': int(size)})
    tpch_files = source_entries(tpchroot, entries)
    roles = query_roles()
    primitives = query_bases(tpchroot, {e['path']: e['sha256'] for e in tpch_files}, roles)
    composites = make_compositions(primitives, roles)
    bases = wf + primitives + composites
    index, records = [], 0
    for b in bases:
        dest = out / 'adapted' / b['dataset'] / (b['base_id'] + '.jsonl.gz')
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0, compresslevel=6) as z:
            for row in physical_variants(b):
                validate_sparse(row)
                z.write((json.dumps(row, separators=(',', ':'), allow_nan=False) + '\n').encode())
                records += 1
        if dest.stat().st_size >= 32 * 1024 * 1024:
            raise ValueError(('adapted record file needs sharding', str(dest)))
        index.append({k: b[k] for k in ['base_id', 'dataset', 'n', 'family', 'split', 'topology_fingerprint',
                                        'source_path', 'source_sha256', 'source_kind', 'previously_inspected']} |
                     {'edges': len(b['edges']), 'adapted_file': str(dest.relative_to(out)), 'variants': 8})
        if len(index) % 100 == 0:
            print(json.dumps({'bases_written': len(index), 'physical_records': records}), flush=True)
    with (out / 'base_catalog.jsonl.gz').open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0) as z:
        for b in bases:
            z.write((json.dumps(b, separators=(',', ':'), allow_nan=False) + '\n').encode())
    archives = {}
    for name, root, entries in [('wfinstances', wfroot, wf_files), ('tpch', tpchroot, tpch_files)]:
        print(json.dumps({'packaging_sources': name, 'files': len(entries)}), flush=True)
        archives[name] = archive_source(root, entries, out / 'raw' / name, Path(args.temp) / (name + '.tar.gz'), name)
    shutil.copy2(wfroot / 'LICENSE', out / 'WfInstances-LICENSE.txt')
    shutil.copy2(wfroot / 'README.md', out / 'WfInstances-UPSTREAM-README.md')
    dump(out / 'INDEX.json', {'schema': 'offline-index-v1', 'bases': index, 'workflow_split': split_info,
                              'tpch_query_roles': roles,
                              'tpch_primitive_scope': 'all154sourcequeries; primitive shapes may overlap roles; use compositional holdout for model comparisons',
                              'tpch_composite_scope': 'generated800TRAIN/100VAL/100TEST, query-template disjoint, WL16 disjoint across composite roles',
                              'physical_records': records, 'teacher_labels_generated': False,
                              'model_TEST_evaluated': False})
    dump(out / 'SOURCES.json', {'wfinstances': {'repository': 'https://github.com/wfcommons/WfInstances',
                  'commit': tree['sha'], 'license': 'LGPL-3.0', 'archive': archives['wfinstances']},
                  'tpch': {'repository': 'https://github.com/Thinklab-SJTU/PPO-BiHyb', 'commit': TPCH_COMMIT,
                  'official_benchmark': 'https://www.tpc.org/tpch/', 'archive': archives['tpch'],
                  'license_note': 'No repository LICENSE file at this author commit; preserve author attribution; no new license granted for upstream material.',
                  'scope': 'author query stage data and recipes, not TPC DBGEN tables or certified TPC-H results'}})
    summary = {'classification': 'complete offline data/physical adaptation; no new learned method result',
               'protocol_commits': ['2a557f7', 'e27ea88'],
               'code_commit_at_build': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
               'source_WfInstances_JSONs': len(wf), 'source_WfInstances_blobs': len(wf_files),
               'source_TPC_H_query_bases': len(primitives), 'generated_TPC_H_composites': len(composites),
               'physical_records': records, 'all_sparse_records_validated': records,
               'all_records_serial_min_cost_feasible_witness': True,
               'by_dataset_split': {}, 'n_range_by_dataset': {}, 'topology_groups_by_dataset_split': {},
               'teacher_labels_generated': False, 'model_TEST_evaluated': False, 'external_download_needed_after_pull': False}
    for dataset in sorted({b['dataset'] for b in bases}):
        selected = [b for b in bases if b['dataset'] == dataset]
        summary['by_dataset_split'][dataset] = dict(Counter(b['split'] for b in selected))
        summary['n_range_by_dataset'][dataset] = [min(b['n'] for b in selected), max(b['n'] for b in selected)]
        summary['topology_groups_by_dataset_split'][dataset] = {s: len({b['topology_fingerprint'] for b in selected if b['split'] == s})
                                                               for s in {b['split'] for b in selected}}
    dump(out / 'BUILD_SUMMARY.json', summary)
    files = []
    for path in sorted(out.rglob('*')):
        if path.is_file():
            blob = path.read_bytes()
            files.append({'path': str(path.relative_to(out)), 'bytes': len(blob), 'sha256': hashlib.sha256(blob).hexdigest()})
    dump(out / 'MANIFEST.json', {'schema': 'offline-bundle-v1', 'files': files,
                                'total_tracked_bytes_without_manifest': sum(f['bytes'] for f in files),
                                'max_file_bytes': max(f['bytes'] for f in files)})
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
