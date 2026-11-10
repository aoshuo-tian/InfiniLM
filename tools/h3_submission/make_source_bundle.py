"""Make a verified public-source transfer bundle when GitHub is unreachable on-node."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    lm = Path(__file__).resolve().parents[2]
    roots = dict(InfiniCore=args.core.resolve(), InfiniLM=lm)
    payload, commits = {}, {}
    for name, root in roots.items():
        if subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain',
                                    '--untracked-files=no']).strip():
            parser.error('Commit reviewed changes before making a source bundle: ' + name)
        commits[name] = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
        files = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z']).decode().rstrip('\0').split('\0')
        for rel in files:
            path = root / rel
            if path.is_file():
                payload['repos/' + name + '/' + rel] = path
        for sub in (('third_party/spdlog',) if name == 'InfiniCore' else ('third_party/spdlog', 'third_party/json')):
            child = root / sub
            commits[name + '/' + sub] = subprocess.check_output(['git', '-C', str(child), 'rev-parse', 'HEAD'], text=True).strip()
            for folder in ('include', 'single_include'):
                for path in (child / folder).rglob('*'):
                    if path.is_file():
                        payload['repos/' + name + '/' + path.relative_to(root).as_posix()] = path
            for path in child.glob('LICENSE*'):
                if path.is_file():
                    payload['repos/' + name + '/' + path.relative_to(root).as_posix()] = path
    hashes = {}
    if args.output.exists():
        parser.error('Refusing to overwrite an existing source bundle')
    with tarfile.open(args.output, 'w:gz', compresslevel=6) as archive:
        for name, path in sorted(payload.items()):
            data = path.read_bytes()
            if path.suffix in ('.so', '.o', '.a', '.pyc', '.whl', '.pt', '.mp4'):
                raise ValueError('Compiled/model payload in source bundle: ' + name)
            hashes[name] = hashlib.sha256(data).hexdigest()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o755 if name.endswith('/tooling/xmake/bin/xmake') else 0o644
            archive.addfile(info, io.BytesIO(data))
        manifest = json.dumps(dict(commits=commits, files=hashes,
            precompiled_framework_libraries=False,
            reused_build_tool='xmake 3.0.5; identity and license bundled'), indent=2).encode()
        info = tarfile.TarInfo('reviewer-source-manifest.json')
        info.size = len(manifest)
        archive.addfile(info, io.BytesIO(manifest))
    print(json.dumps(dict(path=str(args.output), bytes=args.output.stat().st_size,
        sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(), commits=commits), indent=2))


if __name__ == '__main__':
    main()
