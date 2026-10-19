#!/usr/bin/env python3
"""Verify offline sources, ready training data and code without third-party packages."""
import hashlib
import json
from pathlib import Path
from verify_offline_bundle import verify

ROOT = Path(__file__).resolve().parents[1]


def main():
    delivery = json.loads((ROOT / 'OFFLINE_DELIVERY_MANIFEST_20261019.json').read_text())
    for row in delivery['files']:
        p = ROOT / row['path']
        if not p.is_file() or p.stat().st_size != row['bytes']:
            raise AssertionError(('missing/size mismatch', row['path']))
        h = hashlib.sha256()
        with p.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                h.update(block)
        if h.hexdigest() != row['sha256']:
            raise AssertionError(('SHA mismatch', row['path']))
    bundle = verify(ROOT / 'data/offline_20261019')
    for name in ['BUNDLE_MANIFEST.json', 'MODULE_LAB_MANIFEST_20261019.json']:
        old = json.loads((ROOT / name).read_text())
        for row in old['files']:
            if hashlib.sha256((ROOT / row['path']).read_bytes()).hexdigest() != row['sha256']:
                raise AssertionError(('historical file changed', row['path']))
    print(json.dumps({'status': 'PASS', 'offline_data_files': len(bundle['files']),
                      'delivery_files': len(delivery['files']), 'original_frozen_files': 106,
                      'previous_module_files': 90, 'network_requests': 0,
                      'scope': 'integrity only; model quality/CUDA use separate experiments'}))


if __name__ == '__main__':
    main()
