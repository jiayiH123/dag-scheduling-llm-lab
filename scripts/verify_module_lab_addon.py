#!/usr/bin/env python3
"""Verify the additive module handoff without Torch/OR-Tools dependencies."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    addon = json.loads((ROOT / 'MODULE_LAB_MANIFEST_20261019.json').read_text())
    original = json.loads((ROOT / 'BUNDLE_MANIFEST.json').read_text())
    for label, entries in [('frozen_reference', original['files']), ('module_addon', addon['files'])]:
        for entry in entries:
            path = ROOT / entry['path']
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
                raise AssertionError(f'{label} checksum mismatch: {entry["path"]}')
    print(json.dumps({'frozen_reference_SHA256_files': len(original['files']),
                      'addon_SHA256_files': len(addon['files']), 'status': 'PASS',
                      'scope': 'file integrity only; training/physics use separate checks'}))


if __name__ == '__main__':
    main()
