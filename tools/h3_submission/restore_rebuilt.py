"""Restore a freshly built artifact; reject historical cached framework archives."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from runtime_links import connect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--artifact-sha256', required=True)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--work', type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != 'linux' or sys.version_info[:2] != (3, 11):
        parser.error('The specified Linux Python 3.11 base is required')
    if hashlib.sha256(args.artifact.read_bytes()).hexdigest() != args.artifact_sha256:
        parser.error('New runtime artifact identity mismatch')
    work = args.work.resolve()
    if work.exists():
        parser.error('New work directory must not exist')
    with tarfile.open(args.artifact) as archive:
        identity = json.load(archive.extractfile('runtime/reviewer-artifact.json'))
        if not identity.get('from_clean_host_build') or identity['source_bundle_sha256'] != args.source_sha256:
            parser.error('Not an artifact of this source build')
        work.mkdir(parents=True)
        seen = set()
        for member in archive:
            rel = Path(member.name).relative_to('runtime')
            if not member.isfile() or not (work / rel).resolve().is_relative_to(work):
                raise ValueError('Unsafe runtime member')
            data = archive.extractfile(member).read()
            if str(rel) != 'reviewer-artifact.json':
                if rel.as_posix() in seen:
                    raise ValueError('Duplicate runtime member')
                seen.add(rel.as_posix())
                if hashlib.sha256(data).hexdigest() != identity['files'].get(rel.as_posix()):
                    raise ValueError('Runtime member identity mismatch: ' + str(rel))
            target = work / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        if seen != set(identity['files']):
            raise ValueError('Incomplete runtime artifact')
    original = identity['original_work']
    for rel in ('state.json', 'setup.sh', 'build-status.json', 'ccl-build-status.json'):
        path = work / rel
        path.write_text(path.read_text().replace(original, str(work)))
    subprocess.run([sys.executable, '-m', 'venv', '--system-site-packages', '--without-pip',
                    str(work / 'venv')], check=True)
    state = json.loads((work / 'state.json').read_text())
    connect(work, state)
    print('REBUILT_ARTIFACT_RESTORED', work, flush=True)


if __name__ == '__main__':
    main()
