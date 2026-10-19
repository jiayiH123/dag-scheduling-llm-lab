#!/usr/bin/env python3
"""Offline SHA verification and safe source extraction, Python standard library only."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def verify(bundle):
    manifest = json.loads((bundle / 'MANIFEST.json').read_text())
    for row in manifest['files']:
        p = bundle / row['path']
        if not p.is_file() or p.stat().st_size != row['bytes'] or sha(p) != row['sha256']:
            raise AssertionError(('offline bundle checksum', row['path']))
    return manifest


def extract(bundle, out):
    if out.exists():
        raise FileExistsError('source output already exists; use a new directory')
    out.mkdir(parents=True)
    sources = json.loads((bundle / 'SOURCES.json').read_text())
    results = {}
    for name, source in sources.items():
        metadata = source['archive']
        combined = out / (name + '.source.tar.gz')
        with combined.open('wb') as dest:
            for part in metadata['parts']:
                with (bundle / part).open('rb') as handle:
                    shutil.copyfileobj(handle, dest)
        if sha(combined) != metadata['archive_sha256']:
            raise AssertionError('source part reconstruction')
        allowed = {metadata['prefix'] + '/' + row['path']: row for row in metadata['source_files']}
        extracted = set()
        with tarfile.open(combined, 'r|gz') as archive:
            for member in archive:
                parts = PurePosixPath(member.name).parts
                if member.name not in allowed or not member.isfile() or '..' in parts or PurePosixPath(member.name).is_absolute():
                    raise ValueError(('unexpected/unsafe source member', member.name))
                if member.name in extracted:
                    raise ValueError('duplicate archive member')
                dest = out / member.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as handle, dest.open('wb') as output:
                    shutil.copyfileobj(handle, output)
                expected = allowed[member.name]
                if dest.stat().st_size != expected['size'] or sha(dest) != expected['sha256']:
                    raise AssertionError(('raw source hash', member.name))
                extracted.add(member.name)
        if extracted != set(allowed):
            raise AssertionError('missing source members')
        results[name] = {'extracted_source_files': len(extracted), 'all_SHA256_match': True}
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', default=str(ROOT / 'data/offline_20261019'))
    ap.add_argument('--extract-sources')
    ap.add_argument('--output', help='optional new audit JSON path')
    args = ap.parse_args()
    bundle = Path(args.bundle)
    manifest = verify(bundle)
    result = {'status': 'PASS', 'bundle_files': len(manifest['files']),
              'verified_bytes': sum(row['bytes'] for row in manifest['files']),
              'external_network_requests': 0,
              'scope': 'offline engineering integrity, not model evaluation'}
    if args.extract_sources:
        result['sources'] = extract(bundle, Path(args.extract_sources))
    if args.output:
        output = Path(args.output)
        if output.exists():
            raise FileExistsError(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
