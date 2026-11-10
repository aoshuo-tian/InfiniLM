"""Prepare dependencies only in the new build venv; pin system Torch and NumPy.

--plan does not install. --dry-run asks pip to resolve without installing.
torchvision's matching CPU wheel is handled separately with --no-deps.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

VISION_SHA = '51fe8d72ddcc88a50ff27c50dd4987d82fba8b0dfd591898ee1f60e2902d18d1'


def pip_command(python):
    # Some managed nodes disallow /etc/debian_version. Omit optional OS metadata
    # from the HTTP User-Agent; keep pip's resolver, TLS and installation intact.
    bootstrap = ('import mimetypes;mimetypes.knownfiles=[];mimetypes.init([]);'
                 'import pip,runpy;import pip._internal.network.session as session;'
                 'session.user_agent=lambda:"pip/"+pip.__version__;'
                 'runpy.run_module("pip",run_name="__main__")')
    return [str(python), '-c', bootstrap]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--torchvision-wheel', type=Path)
    parser.add_argument('--wheelhouse', type=Path, default=os.environ.get('H3_REVIEW_WHEELHOUSE'))
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    work = args.work.resolve()
    requirements = Path(__file__).with_name('requirements-reviewer.txt')
    if args.plan:
        print(json.dumps(dict(work=str(work), requirements=requirements.read_text().splitlines(),
            venv_only=True, system_torch_numpy_pinned=True, gpu_execution=False,
            dry_run=args.dry_run), indent=2))
        return
    if sys.platform != 'linux':
        parser.error('Run inside the specified Linux image')
    py = work / 'venv/bin/python'
    if not py.is_file():
        parser.error('Prepare the new build directory first')
    # The child verifies its actual venv prefix before any package operation.
    probe = '''import importlib.metadata as m,json,sys
print(json.dumps(dict(prefix=sys.prefix,torch=m.version("torch"),numpy=m.version("numpy"),
 torch_path=str(m.distribution("torch").locate_file("")),
 numpy_path=str(m.distribution("numpy").locate_file("")))))'''
    before = json.loads(subprocess.check_output([str(py), '-c', probe], text=True))
    if Path(before['prefix']).resolve() != work / 'venv':
        parser.error('Refusing to install outside the new private venv')
    if before['torch'] != '2.9.0+das.opt1.dtk2604':
        parser.error('The recorded vendor Torch is required; it will not be installed/replaced')
    constraints = work / 'reviewer-base-constraints.txt'
    constraints.write_text('torch==' + before['torch'] + '\nnumpy==' + before['numpy'] + '\n')
    env = os.environ.copy()
    env.update(PIP_REQUIRE_VIRTUALENV='true', PIP_DISABLE_PIP_VERSION_CHECK='1',
               PIP_NO_CACHE_DIR='1', PIP_CACHE_DIR=str(work / 'cache/pip'))
    report = work / ('dependency-resolution.json' if args.dry_run else 'dependency-install.json')
    source = (['--no-index', '--find-links', str(args.wheelhouse)] if args.wheelhouse
              else ['--index-url', 'https://pypi.org/simple'])
    command = [*pip_command(py), 'install', '--only-binary=:all:', *source,
               '--constraint', str(constraints),
               '--report', str(report), '-r', str(requirements)]
    if args.dry_run:
        command.append('--dry-run')
    with (work / 'logs/reviewer-dependencies.log').open('w') as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    if not args.dry_run:
        wheel = args.torchvision_wheel
        if wheel is None and args.wheelhouse:
            wheel = args.wheelhouse / 'torchvision-0.24.0+cpu-cp311-cp311-manylinux_2_28_x86_64.whl'
        if wheel is None:
            download = work / 'external-wheels'
            download.mkdir(exist_ok=True)
            subprocess.run([*pip_command(py), 'download', '--no-deps', '--only-binary=:all:',
                '--index-url', 'https://download.pytorch.org/whl/cpu', '--dest', str(download),
                'torchvision==0.24.0+cpu'], env=env, check=True)
            wheels = list(download.glob('torchvision-0.24.0+cpu-cp311-cp311-manylinux_2_28_x86_64.whl'))
            if len(wheels) != 1:
                raise RuntimeError('Expected the recorded CPython 3.11 torchvision CPU wheel')
            wheel = wheels[0]
        if hashlib.sha256(wheel.read_bytes()).hexdigest() != VISION_SHA:
            raise RuntimeError('Recorded torchvision CPU wheel SHA mismatch')
        subprocess.run([*pip_command(py), 'install', '--no-deps', str(wheel)], env=env, check=True)
        with (work / 'requirements-private-resolved.txt').open('w') as stream:
            subprocess.run([*pip_command(py), 'freeze', '--local'], stdout=stream, check=True)
    after = json.loads(subprocess.check_output([str(py), '-c', probe], text=True))
    if after != before:
        raise RuntimeError('System Torch/NumPy identities changed')
    (work / 'dependency-base-identities.json').write_text(json.dumps(after, indent=2))
    print('DEPENDENCY_RESOLUTION_PASS' if args.dry_run else 'DEPENDENCY_INSTALL_PASS', flush=True)


if __name__ == '__main__':
    main()
