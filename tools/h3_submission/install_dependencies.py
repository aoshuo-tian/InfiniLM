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
from dependency_probe import probe as dependency_probe

VISION_SHA = '51fe8d72ddcc88a50ff27c50dd4987d82fba8b0dfd591898ee1f60e2902d18d1'


def safe_pip_environment(work):
    env = os.environ.copy()
    for key in ('PIP_TARGET','PIP_PREFIX','PIP_ROOT','PIP_USER','PIP_REQUIREMENT',
                'PIP_CONSTRAINT','PIP_BUILD_CONSTRAINT','PIP_INDEX_URL',
                'PIP_EXTRA_INDEX_URL','PIP_FIND_LINKS','PIP_NO_INDEX'):
        env.pop(key, None)
    env.update(PIP_CONFIG_FILE=os.devnull, PIP_REQUIRE_VIRTUALENV='true',
               PIP_DISABLE_PIP_VERSION_CHECK='1', PIP_NO_CACHE_DIR='1',
               PIP_CACHE_DIR=str(work / 'cache/pip'))
    return env


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
    parser.add_argument('--check-only', action='store_true', help='Verify the existing venv base; do not install')
    parser.add_argument('--allow-online', action='store_true', help='Explicit opt-in for a network-enabled preparation machine')
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
    if args.check_only:
        saved = json.loads((work / 'dependency-base-identities.json').read_text())
        if before != saved or not (work / 'dependency-install.json').is_file():
            parser.error('Restored private environment/base identities differ')
        recheck = dependency_probe(py, [requirements], vision=True)
        (work / 'dependency-recheck.json').write_text(json.dumps(recheck, ensure_ascii=False, indent=2))
        if not recheck['passed']:
            parser.error('已有依赖不能通过版本/接口复核；见 dependency-recheck.json，未安装或下载')
        print('DEPENDENCY_BASE_REUSE_VERIFIED', flush=True)
        return
    # The later comparison stage also needs these constraints when inference
    # dependencies were already usable and pip was never called here.
    constraints = work / 'reviewer-base-constraints.txt'
    constraints.write_text('torch==' + before['torch'] + '\nnumpy==' + before['numpy'] + '\n')
    existing = dependency_probe(py, [requirements], vision=True)
    (work / 'dependency-existing-probe.json').write_text(json.dumps(existing, ensure_ascii=False, indent=2))
    if existing['passed']:
        record = dict(passed=True, mode='reused_existing', probe=existing, protected_base=before)
        destination = 'dependency-resolution.json' if args.dry_run else 'dependency-install.json'
        (work / destination).write_text(json.dumps(record, ensure_ascii=False, indent=2))
        if not args.dry_run:
            (work / 'dependency-base-identities.json').write_text(json.dumps(before, indent=2))
            (work / 'requirements-private-resolved.txt').write_text(''.join(name+'=='+version+'\n' for name,version in sorted(existing['versions'].items())))
        print('DEPENDENCY_EXISTING_REUSE_PASS', flush=True)
        return
    if existing['import_errors']:
        parser.error('已有包导入或H3接口检查失败，请先查看 dependency-existing-probe.json；未盲目重装')
    only_vision = existing['missing_or_version_mismatch'] == ['torchvision==0.24.0+cpu']
    if not args.wheelhouse and not args.allow_online and not (only_vision and args.torchvision_wheel):
        parser.error('缺少或版本不符：'+', '.join(existing['missing_or_version_mismatch'])+
                     '；请提供 --wheelhouse。Fabric 默认不联网；仅联网准备机可显式 --allow-online')
    env = safe_pip_environment(work)
    report = work / ('dependency-resolution.json' if args.dry_run else 'dependency-install.json')
    source = (['--no-index', '--find-links', str(args.wheelhouse)] if args.wheelhouse
              else ['--index-url', 'https://pypi.org/simple'] if args.allow_online else ['--no-index'])
    command = [*pip_command(py), 'install', '--only-binary=:all:', *source,
               '--constraint', str(constraints),
               '--report', str(report), '-r', str(requirements)]
    if args.dry_run:
        command.append('--dry-run')
    with (work / 'logs/reviewer-dependencies.log').open('w') as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    if not args.dry_run:
        vision_ready = 'torchvision==0.24.0+cpu' not in existing['missing_or_version_mismatch']
        wheel = args.torchvision_wheel
        if not vision_ready and wheel is None and args.wheelhouse:
            wheel = args.wheelhouse / 'torchvision-0.24.0+cpu-cp311-cp311-manylinux_2_28_x86_64.whl'
        if not vision_ready and wheel is None:
            download = work / 'external-wheels'
            download.mkdir(exist_ok=True)
            subprocess.run([*pip_command(py), 'download', '--no-deps', '--only-binary=:all:',
                '--index-url', 'https://download.pytorch.org/whl/cpu', '--dest', str(download),
                'torchvision==0.24.0+cpu'], env=env, check=True)
            wheels = list(download.glob('torchvision-0.24.0+cpu-cp311-cp311-manylinux_2_28_x86_64.whl'))
            if len(wheels) != 1:
                raise RuntimeError('Expected the recorded CPython 3.11 torchvision CPU wheel')
            wheel = wheels[0]
        if not vision_ready:
            if hashlib.sha256(wheel.read_bytes()).hexdigest() != VISION_SHA:
                raise RuntimeError('Recorded torchvision CPU wheel SHA mismatch')
            subprocess.run([*pip_command(py), 'install', '--no-deps', str(wheel)], env=env, check=True)
        with (work / 'requirements-private-resolved.txt').open('w') as stream:
            subprocess.run([*pip_command(py), 'freeze', '--local'], stdout=stream, check=True)
    after = json.loads(subprocess.check_output([str(py), '-c', probe], text=True))
    if after != before:
        raise RuntimeError('System Torch/NumPy identities changed')
    if not args.dry_run:
        final_probe = dependency_probe(py, [requirements], vision=True)
        (work / 'dependency-final-probe.json').write_text(json.dumps(final_probe, ensure_ascii=False, indent=2))
        if not final_probe['passed']:
            raise RuntimeError('依赖准备后接口检查未通过，停止；见 dependency-final-probe.json')
    (work / 'dependency-base-identities.json').write_text(json.dumps(after, indent=2))
    print('DEPENDENCY_RESOLUTION_PASS' if args.dry_run else 'DEPENDENCY_INSTALL_PASS', flush=True)


if __name__ == '__main__':
    main()
