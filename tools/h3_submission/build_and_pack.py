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
import shutil

from build_monitor import run_logged, write_json, tail
from build_checkpoint import pack_checkpoint, restore_checkpoint, publish_file


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
    parser.add_argument('--resume-root', type=Path)
    parser.add_argument('--checkpoint-from', type=Path)
    parser.add_argument('--checkpoint-sha256')
    parser.add_argument('--checkpoint-output', type=Path)
    parser.add_argument('--checkpoint-max-mib', type=int, default=96)
    parser.add_argument('--checkpoint-reserve', type=int, default=120)
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
    if args.resume_root and args.checkpoint_from:
        parser.error('Choose in-place resume or checkpoint restore')
    resuming = bool(args.resume_root or args.checkpoint_from)
    root = args.resume_root.resolve() if args.resume_root else Path(tempfile.mkdtemp(prefix='h3-reviewer-build-'))
    if args.checkpoint_from:
        if not args.checkpoint_sha256:
            parser.error('A checkpoint SHA is required')
        root = root / 'restored'
        restore_checkpoint(args.checkpoint_from, args.checkpoint_sha256, args.source_sha256, root)
        # Move only this verified input copy out of the project, to leave room
        # for its successor checkpoint. The stored platform project is unchanged.
        shutil.move(str(args.checkpoint_from), str(root.parent / 'input-checkpoint.tar.gz'))
        subprocess.run([sys.executable, '-m', 'venv', '--system-site-packages', '--without-pip',
                        str(root / 'build/venv')], check=True)
    if resuming:
        previous = json.loads((root / 'session-report.json').read_text())
        if previous['source_sha256'] != args.source_sha256:
            parser.error('Resume source identity mismatch')
        if not (root / 'build/build-origin.json').is_file():
            parser.error('No verified build origin; legacy partial directories cannot be blindly resumed')
    source = root / 'input'
    source.mkdir(exist_ok=resuming)
    report = dict(complete=False, passed=False, root=str(root), stages=[],
                  source_sha256=args.source_sha256, gpu_executed=False)
    if resuming:
        report.update(stages=previous['stages'], resumed=True)
    receipt = output.with_suffix(output.suffix + '.json')
    # A single owner is required for incremental objects and checkpoints.
    import fcntl
    lock = (root / 'reviewer-build.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
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
                if resuming:
                    if not target.is_file() or sha(target) != hashlib.sha256(data).hexdigest():
                        raise ValueError('Restored source differs from the public bundle: ' + member.name)
                else:
                    target.write_bytes(data)
                    target.chmod(member.mode)
            if seen != set(manifest['files']):
                raise ValueError('Incomplete source bundle')
        report['source_commits'] = manifest['commits']
        # Optional offline wheels are separately hash-pinned external assets.
        # Move the archive out of the project after verification so saving the
        # resulting runtime does not also archive a duplicate dependency bundle.
        wheels = Path('reviewer-offline-wheels.tar.gz')
        if args.checkpoint_from and not wheels.is_file():
            raise RuntimeError('Restored checkpoints require the previously verified offline wheel archive; no online fallback')
        if wheels.is_file() and (not resuming or args.checkpoint_from):
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
            reserve = args.checkpoint_reserve if args.checkpoint_output else 0
            remaining = args.seconds - (time.monotonic() - started) - reserve
            if remaining <= 0:
                raise TimeoutError('CPU build budget exhausted')
            def progress(record):
                report['current_stage'] = dict(label=label, **record)
                write_json(receipt, report)
                chunks = []
                for path in sorted((work / 'logs').glob('*.log'), key=lambda p:p.stat().st_mtime)[-3:]:
                    chunks.append(path.name + '\n' + tail(path, 16384))
                (output.parent / 'reviewer-live.log').write_text('\n'.join(chunks), encoding='utf-8')
                for name in ('build-progress.json', 'build-status.json', 'ccl-build-status.json'):
                    if (work / name).is_file():
                        shutil.copyfile(work / name, output.parent / ('reviewer-' + name))
            record = run_logged(command, log, timeout=remaining, new_session=True, on_progress=progress)
            report['stages'].append(dict(label=label, **record))
            write_json(receipt, report)
            print('REVIEWER_STAGE', label, record['exit_code'], record['status'], flush=True)
            if record['exit_code']:
                raise RuntimeError(label + ' ' + record['status'] + '; see ' + str(log) + '\n' + record['tail'])

        if not resuming:
            run('prepare', [sys.executable, scripts / 'prepare_build.py', '--core', core, '--work', work])
            run('dependencies', [sys.executable, scripts / 'install_dependencies.py', '--work', work])
        else:
            # The framework origin and every restored file are verified separately.
            # Re-check the same base package constraints before reusing the venv.
            run('resume-base-check', [sys.executable, scripts / 'install_dependencies.py', '--work', work, '--check-only'])
            if args.checkpoint_from:
                run('resume-private-dependencies', [sys.executable, scripts / 'install_dependencies.py', '--work', work])
        framework_passed = resuming and (work / 'build-status.json').is_file() and json.loads((work / 'build-status.json').read_text()).get('passed')
        framework_mode = ['--verify-completed'] if framework_passed else (['--resume'] if resuming else [])
        run('framework', [sys.executable, scripts / 'build_framework.py', *framework_mode], setup=True)
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
        runtime_temporary = root / 'new-runtime.partial'
        with tarfile.open(runtime_temporary, 'w:gz', compresslevel=3) as archive:
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
        publish_file(runtime_temporary, output)
        report.update(passed=True, artifact_bytes=output.stat().st_size,
                      artifact_sha256=sha(output))
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        report.update(complete=True, elapsed_seconds=time.monotonic() - started)
        if args.checkpoint_output and not report['passed']:
            try:
                write_json(root / 'session-report.json', report)
                (root / 'session-logs').mkdir(exist_ok=True)
                for path in output.parent.glob('reviewer-*.log'):
                    shutil.copyfile(path, root / 'session-logs' / path.name)
                report['checkpoint'] = pack_checkpoint(root, args.checkpoint_output,
                    args.source_sha256, args.checkpoint_max_mib * 1024 * 1024)
                print('REVIEWER_CHECKPOINT_SAVED', json.dumps(report['checkpoint']), flush=True)
            except Exception as error:
                report['checkpoint'] = dict(portable_checkpoint_created=False, error=repr(error))
                print('REVIEWER_CHECKPOINT_FAILED', repr(error), flush=True)
        write_json(receipt, report)
        write_json(root / 'session-report.json', report)
        lock.close()
        print('REVIEWER_BUILD_RESULT', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
