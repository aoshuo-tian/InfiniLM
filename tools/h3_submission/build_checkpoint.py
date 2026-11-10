"""Hash-verified build checkpoints, distinct from successful runtime artifacts."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile
import tempfile
import gzip
import errno
import shutil

from build_monitor import write_json


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


class LimitedWriter:
    def __init__(self, stream, maximum):
        self.stream, self.maximum = stream, maximum

    def write(self, data):
        if self.stream.tell() + len(data) > self.maximum:
            raise RuntimeError('Checkpoint exceeds configured byte limit; no portable checkpoint published')
        return self.stream.write(data)

    def tell(self):
        return self.stream.tell()

    def flush(self):
        self.stream.flush()


def safe_name(name):
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or path.as_posix() != name or '\\' in name or any(x in ('..', '.') or ':' in x for x in path.parts):
        raise ValueError('Unsafe checkpoint member: ' + name)
    return path


def included(path, root):
    rel = path.relative_to(root)
    return (rel.parts[0] in ('input', 'build', 'session-logs') or rel.name == 'session-report.json') \
        and not any(x in ('__pycache__', '.git') for x in rel.parts) \
        and rel.parts[:2] != ('build', 'cache') \
        and rel.parts[:2] not in (('build', 'venv'), ('build', 'external-wheels')) \
        and path.suffix != '.pyc' and path.name != 'reviewer-build.lock'


def publish_file(source, destination):
    """Publish across the node scratch/project filesystem boundary atomically."""
    source, destination = Path(source), Path(destination)
    try:
        source.replace(destination)
        return
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
    descriptor, name = tempfile.mkstemp(prefix='.h3-publish-', suffix='.partial', dir=destination.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        shutil.copyfile(source, temporary)
        if sha(temporary) != sha(source):
            raise RuntimeError('Cross-filesystem publication SHA mismatch')
        temporary.replace(destination)
        source.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()


def pack_checkpoint(root, output, source_sha256, maximum=96 * 1024 * 1024):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.is_relative_to(root):
        raise ValueError('Checkpoint output must be outside the build tree')
    origin = root / 'build/build-origin.json'
    if not origin.is_file():
        raise RuntimeError('No verified clean-build origin; refusing a resumable checkpoint')
    manifest = dict(kind='h3-incomplete-build-checkpoint-v1', original_root=str(root),
                    source_sha256=source_sha256, python_base=str(Path(sys.executable).resolve()),
                    successful_runtime=False, offline_dependency_restore_required=True, files={})
    output.parent.mkdir(parents=True, exist_ok=True)
    # Incomplete output remains outside the project. Only a fully closed, bounded
    # archive replaces an older checkpoint, which remains usable on pack failure.
    descriptor, temporary_name = tempfile.mkstemp(prefix='checkpoint-', suffix='.partial', dir=root.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open('wb') as stream:
            with gzip.GzipFile(fileobj=LimitedWriter(stream, maximum), mode='wb', compresslevel=3) as compressed, tarfile.open(fileobj=compressed, mode='w|') as archive:
                for path in sorted(root.rglob('*')):
                    if not included(path, root) or (not path.is_file() and not path.is_symlink()):
                        continue
                    name = path.relative_to(root).as_posix()
                    info = archive.gettarinfo(str(path), arcname=name)
                    if path.is_symlink():
                        target = os.readlink(path)
                        resolved = (path.parent / target).resolve()
                        if not resolved.is_relative_to(root) and resolved != Path(sys.executable).resolve():
                            raise ValueError('External checkpoint symlink: ' + name)
                        manifest['files'][name] = dict(link=target)
                        archive.addfile(info)
                    else:
                        manifest['files'][name] = dict(sha256=sha(path), bytes=path.stat().st_size)
                        with path.open('rb') as source:
                            archive.addfile(info, source)
                data = json.dumps(manifest, indent=2).encode()
                info = tarfile.TarInfo('checkpoint-manifest.json')
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
        digest, size = sha(temporary), temporary.stat().st_size
        publish_file(temporary, output)
        output.with_suffix(output.suffix + '.sha256').write_text(digest + '  ' + output.name + '\n')
        return dict(path=str(output), sha256=digest, bytes=size, portable_checkpoint_created=True,
                    successful_runtime=False)
    finally:
        if temporary.exists():
            temporary.unlink()


def restore_checkpoint(archive_path, expected_sha, source_sha256, destination):
    archive_path, destination = Path(archive_path), Path(destination).resolve()
    if sha(archive_path) != expected_sha:
        raise ValueError('Checkpoint archive SHA mismatch')
    if destination.exists():
        raise ValueError('Checkpoint destination must not exist')
    links, seen, total = [], set(), 0
    with tarfile.open(archive_path) as archive:
        if sum(m.name == 'checkpoint-manifest.json' for m in archive.getmembers()) != 1:
            raise ValueError('Missing/duplicate checkpoint manifest')
        manifest = json.load(archive.extractfile('checkpoint-manifest.json'))
        if manifest.get('kind') != 'h3-incomplete-build-checkpoint-v1' or \
                manifest['source_sha256'] != source_sha256 or manifest['successful_runtime']:
            raise ValueError('Wrong checkpoint provenance')
        if Path(manifest['python_base']).resolve() != Path(sys.executable).resolve():
            raise ValueError('Different base Python; cross-image resume is not validated')
        destination.mkdir(parents=True)
        for member in archive:
            if member.name == 'checkpoint-manifest.json':
                continue
            rel = safe_name(member.name)
            if member.name in seen or member.name not in manifest['files']:
                raise ValueError('Duplicate/unlisted checkpoint member')
            seen.add(member.name)
            expected = manifest['files'][member.name]
            target = destination.joinpath(*rel.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            if member.issym() and expected == dict(link=member.linkname):
                links.append((target, member.linkname))
            elif member.isfile() and 'sha256' in expected:
                total += member.size
                if member.size != expected['bytes'] or total > 2 * 1024 ** 3:
                    raise ValueError('Checkpoint expansion limit exceeded')
                data = archive.extractfile(member).read()
                if hashlib.sha256(data).hexdigest() != expected['sha256']:
                    raise ValueError('Checkpoint member SHA mismatch: ' + member.name)
                target.write_bytes(data)
                target.chmod(member.mode & 0o777)
                os.utime(target, (member.mtime, member.mtime))
            else:
                raise ValueError('Unsupported checkpoint member')
        if seen != set(manifest['files']):
            raise ValueError('Incomplete checkpoint')
    original = manifest['original_root']
    for target, link in links:
        link = link.replace(original, str(destination))
        resolved = (target.parent / link).resolve()
        if not resolved.is_relative_to(destination) and resolved != Path(sys.executable).resolve():
            raise ValueError('Unsafe restored symlink')
        target.symlink_to(link, target_is_directory=resolved.is_dir())
    # Relocate build bookkeeping, not model/operator source or object bytes.
    relocated = []
    for path in destination.rglob('*'):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(destination)
        metadata = path.name == 'setup.sh' or path.suffix in ('.json', '.d', '.xmake') \
            or '.xmake' in rel.parts or ('venv' in rel.parts and 'bin' in rel.parts)
        if metadata:
            data = path.read_bytes()
            if b'\0' not in data and original.encode() in data:
                before = path.stat()
                path.write_bytes(data.replace(original.encode(), str(destination).encode()))
                os.utime(path, (before.st_atime, before.st_mtime))
                relocated.append(rel.as_posix())
    receipt = dict(checkpoint_sha256=expected_sha, source_sha256=source_sha256,
                   restored_root=str(destination), files_verified=len(seen),
                   metadata_relocated=relocated, gpu_execution_validated=False,
                   offline_dependency_restore_required=True,
                   incremental_target_validation_pending=True)
    write_json(destination / 'checkpoint-restored.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(restore_checkpoint(args.archive, args.sha256, args.source_sha256,
                                        args.destination), indent=2))


if __name__ == '__main__':
    main()
