"""Build in a new temporary directory and export only the newly built runtime.

Designed for shared CPU: no model is loaded and no DCU kernel is launched.
The source bundle must have been produced from committed public checkouts.
"""
import argparse
import hashlib
import io
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import time


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-bundle', type=Path, required=True)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=1650)
    args = parser.parse_args()
    if sys.platform != 'linux' or sys.version_info[:2] != (3, 11):
        parser.error('Linux Python 3.11 with the specified DTK/Torch base is required')
    if sha(args.source_bundle) != args.source_sha256:
        parser.error('Source bundle SHA mismatch')
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        parser.error('Output already exists')
    started = time.monotonic()
    root = Path(tempfile.mkdtemp(prefix='h3-reviewer-build-'))
    source = root / 'input'
    source.mkdir()
    report = dict(complete=False, passed=False, root=str(root), stages=[],
                  source_sha256=args.source_sha256, gpu_executed=False)
    receipt = output.with_suffix(output.suffix + '.json')
    try:
        with tarfile.open(args.source_bundle) as archive:
            manifest = json.load(archive.extractfile('reviewer-source-manifest.json'))
            seen = set()
            for member in archive:
                if not member.isfile() or not (source / member.name).resolve().is_relative_to(source):
                    raise ValueError('Unsafe source member')
                data = archive.extractfile(member).read()
                if member.name != 'reviewer-source-manifest.json':
                    if member.name in seen:
                        raise ValueError('Duplicate source member')
                    seen.add(member.name)
                    if hashlib.sha256(data).hexdigest() != manifest['files'].get(member.name):
                        raise ValueError('Source member identity mismatch: ' + member.name)
                target = source / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                target.chmod(member.mode)
            if seen != set(manifest['files']):
                raise ValueError('Incomplete source bundle')
        report['source_commits'] = manifest['commits']
        # Optional offline wheels are separately hash-pinned external assets.
        # Move the archive out of the project after verification so saving the
        # resulting runtime does not also archive a duplicate dependency bundle.
        wheels = Path('reviewer-offline-wheels.tar.gz')
        if wheels.is_file():
            wheel_sha = Path('reviewer-offline-wheels.sha256').read_text().split()[0]
            if sha(wheels) != wheel_sha:
                raise ValueError('Offline dependency archive identity mismatch')
            wheelhouse = root / 'wheelhouse'
            wheelhouse.mkdir()
            with tarfile.open(wheels) as archive:
                wheel_manifest = json.load(archive.extractfile('wheel-manifest.json'))
                seen = set()
                for member in archive:
                    if member.name == 'wheel-manifest.json':
                        continue
                    if not member.isfile() or Path(member.name).name != member.name:
                        raise ValueError('Unsafe wheel archive member')
                    data = archive.extractfile(member).read()
                    if hashlib.sha256(data).hexdigest() != wheel_manifest['files'].get(member.name):
                        raise ValueError('Offline wheel identity mismatch')
                    if member.name in seen:
                        raise ValueError('Duplicate wheel member')
                    seen.add(member.name)
                    (wheelhouse / member.name).write_bytes(data)
                if seen != set(wheel_manifest['files']):
                    raise ValueError('Incomplete offline dependencies')
            import shutil
            shutil.move(str(wheels), str(root / wheels.name))
            os.environ['H3_REVIEW_WHEELHOUSE'] = str(wheelhouse)
            report['offline_dependencies_sha256'] = wheel_sha
        lm, core = source / 'repos/InfiniLM', source / 'repos/InfiniCore'
        scripts = lm / 'tools/h3_submission'
        work = root / 'build'

        def run(label, arguments, setup=False):
            command = [str(x) for x in arguments]
            if setup:
                command = ['bash', '-c', 'source "$1"; shift; exec "$@"', 'h3-reviewer',
                           str(work / 'setup.sh'), *command]
            log = output.parent / ('reviewer-' + label + '.log')
            before = time.monotonic()
            remaining = args.seconds - (before - started)
            if remaining <= 0:
                raise TimeoutError('CPU build budget exhausted')
            with log.open('w') as stream:
                process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                try:
                    process.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise
            report['stages'].append(dict(label=label, command=command,
                                        exit_code=process.returncode, seconds=time.monotonic() - before))
            receipt.write_text(json.dumps(report, indent=2))
            print('REVIEWER_STAGE', label, process.returncode, flush=True)
            if process.returncode:
                raise RuntimeError(label + ' failed; see ' + str(log) + '\n' + log.read_text()[-5000:])

        run('prepare', [sys.executable, scripts / 'prepare_build.py', '--core', core, '--work', work])
        run('dependencies', [sys.executable, scripts / 'install_dependencies.py', '--work', work])
        run('framework', [sys.executable, scripts / 'build_framework.py'], setup=True)
        run('ccl', [sys.executable, scripts / 'build_ccl.py'], setup=True)
        state = json.loads((work / 'state.json').read_text())
        selected = {}

        def add(path):
            if path.is_file() and path.suffix != '.pyc' and '__pycache__' not in path.parts:
                selected[path.relative_to(work).as_posix()] = path

        for rel in ('state.json', 'setup.sh', 'build-status.json', 'ccl-build-status.json',
                    'dependency-base-identities.json', 'dependency-install.json',
                    'requirements-private-resolved.txt'):
            add(work / rel)
        for folder in ('runtime/python', 'venv/lib/python3.11/site-packages', 'logs'):
            for path in (work / folder).rglob('*'):
                add(path)
        for path in Path(state['lib']).glob('*.so'):
            add(path)
        add(Path(state['extension']))
        add(Path(state['lm']) / 'examples/minimax_h3.py')
        identity = dict(from_clean_host_build=True, gpu_execution_validated=False,
                        original_work=str(work), source_commits=manifest['commits'],
                        source_bundle_sha256=args.source_sha256, files={})
        # Only explicit regular files are emitted, never absolute venv/lib symlinks.
        with tarfile.open(output, 'w:gz', compresslevel=3) as archive:
            for rel, path in sorted(selected.items()):
                data = path.read_bytes()
                identity['files'][rel] = hashlib.sha256(data).hexdigest()
                item = tarfile.TarInfo('runtime/' + rel)
                item.size = len(data)
                item.mode = 0o644
                archive.addfile(item, io.BytesIO(data))
            data = json.dumps(identity, indent=2).encode()
            item = tarfile.TarInfo('runtime/reviewer-artifact.json')
            item.size = len(data)
            archive.addfile(item, io.BytesIO(data))
        report.update(passed=True, artifact_bytes=output.stat().st_size,
                      artifact_sha256=sha(output))
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        report.update(complete=True, elapsed_seconds=time.monotonic() - started)
        receipt.write_text(json.dumps(report, indent=2))
        print('REVIEWER_BUILD_RESULT', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
