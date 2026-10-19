#!/usr/bin/env python3
"""Verify the new structured package plus all prior offline/frozen deliveries."""
import json
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from daglab.structured_dataset import sha256


def main():
    manifest = json.loads((ROOT/'STRUCTURED_DATA_DELIVERY_20261019.json').read_text())
    for row in manifest['files']:
        p = ROOT/row['path']
        if not p.is_file() or p.stat().st_size != row['bytes'] or sha256(p) != row['sha256']:
            raise AssertionError(('new delivery hash', row['path']))
    subprocess.run([sys.executable, str(ROOT/'scripts/verify_structured_dataset.py'),
                    '--data', str(ROOT/'data/structured_scheduling_v1')], check=True, cwd=ROOT)
    subprocess.run([sys.executable, str(ROOT/'scripts/verify_offline_delivery.py')], check=True, cwd=ROOT)
    print(json.dumps(dict(structured_delivery='PASS', additional_files=len(manifest['files']),
                         network_requests=0, scope='data/target/code integrity, not trained Qwen quality')))


if __name__ == '__main__': main()
