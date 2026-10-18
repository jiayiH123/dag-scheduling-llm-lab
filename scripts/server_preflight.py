#!/usr/bin/env python3
"""Record hardware/runtime inventory without installing packages or changing CUDA."""
import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    ap.add_argument('--require-cuda', action='store_true')
    args = ap.parse_args()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(f'choose a new output: {out}')
    versions = {}
    for package in ('torch', 'numpy', 'ortools', 'wfcommons'):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    info = {
        'purpose': 'inventory only; not a training or GPU performance benchmark',
        'python': sys.version,
        'executable': sys.executable,
        'platform': platform.platform(),
        'machine': platform.machine(),
        'cpu_logical_cores': os.cpu_count(),
        'disk_free_bytes': shutil.disk_usage(Path.cwd()).free,
        'package_versions': versions,
    }
    try:
        info['physical_memory_bytes'] = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    except (ValueError, OSError, AttributeError):
        info['physical_memory_bytes'] = None
    if shutil.which('nvidia-smi'):
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=20, check=False)
            info['nvidia_smi'] = {'returncode': result.returncode,
                                  'gpu_rows': result.stdout.strip().splitlines(),
                                  'stderr': result.stderr.strip()}
        except (OSError, subprocess.TimeoutExpired) as exc:
            info['nvidia_smi'] = {'error': str(exc)}
    else:
        info['nvidia_smi'] = {'available': False}
    info['cuda_available'] = False
    try:
        import torch
        info['torch_cuda_build'] = torch.version.cuda
        info['cuda_available'] = torch.cuda.is_available()
        info['torch_devices'] = [
            {'index': i, 'name': torch.cuda.get_device_name(i),
             'total_memory_bytes': torch.cuda.get_device_properties(i).total_memory}
            for i in range(torch.cuda.device_count())
        ] if info['cuda_available'] else []
        info['torch_num_threads'] = torch.get_num_threads()
    except Exception as exc:
        info['torch_import_error'] = str(exc)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as f:
        json.dump(info, f, indent=2)
        f.write('\n')
    print(json.dumps({'output': str(out), 'cuda_available': info['cuda_available'],
                      'packages': versions}, ensure_ascii=False))
    if args.require_cuda and not info['cuda_available']:
        raise SystemExit('CUDA is unavailable in this interpreter; inventory saved. Check the driver and PyTorch wheel before training.')


if __name__ == '__main__':
    main()
