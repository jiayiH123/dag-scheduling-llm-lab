#!/usr/bin/env python3
"""One-time source acquisition; offline users never need to run this script."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile


def matches(path, entry):
    if not path.is_file() or path.stat().st_size != entry['size']:
        return False
    blob = path.read_bytes()
    return hashlib.sha1(b'blob ' + str(len(blob)).encode() + b'\0' + blob).hexdigest() == entry['sha']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tree', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--partial-archive')
    ap.add_argument('--workers', type=int, default=6)
    args = ap.parse_args()
    tree = json.loads(Path(args.tree).read_text())
    if tree.get('truncated'):
        raise ValueError('incomplete source tree')
    entries = {e['path']: e for e in tree['tree'] if e['type'] == 'blob'}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    salvaged = 0
    if args.partial_archive:
        try:
            with tarfile.open(args.partial_archive, 'r|gz') as archive:
                for member in archive:
                    if not member.isfile():
                        continue
                    name = str(PurePosixPath(member.name).relative_to(PurePosixPath(member.name).parts[0]))
                    if name not in entries:
                        raise ValueError(('unexpected archive path', name))
                    dest = out / name
                    if matches(dest, entries[name]):
                        continue
                    blob = archive.extractfile(member).read()
                    e = entries[name]
                    if len(blob) != e['size'] or hashlib.sha1(b'blob ' + str(len(blob)).encode() + b'\0' + blob).hexdigest() != e['sha']:
                        raise ValueError(('archive Git object mismatch', name))
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(blob)
                    salvaged += 1
        except (EOFError, tarfile.ReadError, OSError) as error:
            print(json.dumps({'partial_archive_salvaged': salvaged, 'interrupted_archive_error': str(error)}), flush=True)
    missing = [e for e in entries.values() if not matches(out / e['path'], e)]
    print(json.dumps({'verified_existing': len(entries) - len(missing), 'remaining': len(missing)}), flush=True)

    def fetch(e):
        dest = out / e['path']
        dest.parent.mkdir(parents=True, exist_ok=True)
        temp = dest.with_name(dest.name + '.partial')
        url = f"https://raw.githubusercontent.com/wfcommons/WfInstances/{tree['sha']}/{e['path']}"
        done = subprocess.run(['curl', '-sS', '-L', '--fail', '--retry', '3', '--retry-all-errors',
                               '--connect-timeout', '20', '--max-time', '180', url, '-o', str(temp)],
                              capture_output=True, text=True)
        if done.returncode or not matches(temp, e):
            return {'path': e['path'], 'returncode': done.returncode, 'error': done.stderr[-800:] or 'Git object mismatch'}
        temp.replace(dest)
        return None

    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, e): e for e in missing}
        for n, future in enumerate(as_completed(futures), 1):
            error = future.result()
            if error:
                errors.append(error)
            if n % 10 == 0 or n == len(missing):
                print(json.dumps({'finished_downloads': n, 'total_downloads': len(missing), 'errors': len(errors)}), flush=True)
    if errors:
        (out / '_download_errors.json').write_text(json.dumps(errors, indent=2) + '\n')
        raise RuntimeError(errors)
    assert all(matches(out / e['path'], e) for e in entries.values())
    print(json.dumps({'status': 'PASS', 'source_files': len(entries),
                      'JSON_files': sum(p.endswith('.json') for p in entries),
                      'source_bytes': sum(e['size'] for e in entries.values()), 'commit': tree['sha']}))


if __name__ == '__main__':
    main()
