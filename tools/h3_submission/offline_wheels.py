"""Collect a resolved dependency closure from a verified preparation environment.

Run on a networked Linux Python 3.11 machine matching the target base image,
after prepare_offline.sh has checked the Python APIs. Torch and NumPy remain
target-image prerequisites, not downloads. This does not load a model/kernel.
"""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import platform
from pathlib import Path
import subprocess
import sys
import zipfile

from dependency_probe import Requirement
from install_dependencies import pip_command, safe_pip_environment, VISION_SHA

PROTECTED = {'torch', 'numpy'}


def normalized(name):
    return name.lower().replace('_', '-').replace('.', '-')


def closure(requirements, distribution=metadata.distribution):
    """Resolve installed requirements, including extras/markers, without pip."""
    queue = list(requirements)
    selected, extras_by_name, expanded = {}, {}, {}
    while queue:
        requirement = Requirement(queue.pop(0))
        if requirement.marker and not requirement.marker.evaluate({'extra': ''}):
            continue
        name = normalized(requirement.name)
        dist = distribution(requirement.name)
        version = dist.version
        if not requirement.specifier.contains(version, prereleases=True):
            raise ValueError('已安装依赖不满足约束：' + str(requirement) + '，实际 ' + version)
        if name in selected and selected[name] != version:
            raise ValueError('同名包版本不一致：' + name)
        selected[name] = version
        wanted = extras_by_name.setdefault(name, set())
        wanted.update(requirement.extras)
        if name in PROTECTED or (name in expanded and expanded[name] == wanted):
            continue
        expanded[name] = wanted.copy()
        for text in dist.requires or []:
            child = Requirement(text)
            if child.marker and not any(child.marker.evaluate({'extra': extra})
                                        for extra in wanted | {''}):
                continue
            # The marker was evaluated in the parent's extra context. Do not
            # re-evaluate it using the child's empty extra on the next visit.
            child.marker = None
            queue.append(str(child))
    return dict(sorted(selected.items()))


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def wheel_identity(path):
    from email.parser import BytesParser
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.endswith('.dist-info/METADATA')]
        if len(names) != 1:
            raise ValueError('wheel 元数据数量错误：' + str(path))
        record = BytesParser().parsebytes(archive.read(names[0]))
    return normalized(record['Name']), record['Version']


def collect(output, resolved, download=False):
    if output.exists():
        raise ValueError('离线包输出目录必须不存在，避免混入旧文件：' + str(output))
    if not download:
        return dict(resolved=resolved, download=False,
                    protected_base={n: resolved[n] for n in PROTECTED},
                    note='只读计划；未创建输出目录或下载')
    output.mkdir(parents=True)
    wheels = output / 'wheels'
    wheels.mkdir()
    env = safe_pip_environment(output)
    # This explicit preparation command runs inside the private prep venv.
    if sys.prefix == sys.base_prefix:
        raise RuntimeError('只允许在专用准备 venv 下载')
    for name, version in resolved.items():
        if name in PROTECTED:
            continue
        index = ('https://download.pytorch.org/whl/cpu' if name == 'torchvision'
                 else 'https://pypi.org/simple')
        subprocess.run([*pip_command(sys.executable), 'download', '--no-deps',
            '--only-binary=:all:', '--index-url', index, '--dest', str(wheels),
            name + '==' + version], check=True, env=env)
    identities, inventory = {}, {}
    for wheel in sorted(wheels.glob('*.whl')):
        name, version = wheel_identity(wheel)
        if name in identities or name in PROTECTED or resolved.get(name) != version:
            raise ValueError('wheel 身份或数量不符合解析结果：' + wheel.name)
        digest = sha(wheel)
        if name == 'torchvision' and digest != VISION_SHA:
            raise ValueError('torchvision CPU wheel SHA 不匹配')
        identities[name] = version
        inventory[wheel.name] = dict(name=name, version=version,
                                     bytes=wheel.stat().st_size, sha256=digest)
    if identities != {n: v for n, v in resolved.items() if n not in PROTECTED}:
        raise ValueError('离线依赖闭包不完整；不写成功清单')
    (output / 'requirements-resolved.txt').write_text(''.join(
        n + '==' + v + '\n' for n, v in resolved.items() if n not in PROTECTED))
    record = dict(passed=True, python=sys.version, platform=sys.platform,
                  protected_base={n: resolved[n] for n in PROTECTED},
                  resolved=resolved, wheels=inventory,
                  vendor_flash_included=False, model_included=False,
                  gpu_execution=False)
    (output / 'manifest.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--download', action='store_true', help='显式允许联网下载；默认只读计划')
    args = parser.parse_args()
    if (sys.platform != 'linux' or sys.version_info[:2] != (3, 11)
            or platform.machine() != 'x86_64'):
        parser.error('需要匹配目标镜像的 Linux x86_64 Python3.11 准备环境')
    requirements = ['torch==2.9.0+das.opt1.dtk2604', 'numpy', 'torchvision==0.24.0+cpu']
    for name in ('requirements-reviewer.txt', 'requirements-comparison.txt'):
        requirements.extend(line.strip() for line in Path(__file__).with_name(name).read_text().splitlines()
                            if line.strip() and not line.lstrip().startswith('#'))
    print(json.dumps(collect(args.output.resolve(), closure(requirements), args.download),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
