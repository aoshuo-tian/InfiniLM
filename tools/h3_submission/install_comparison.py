"""Install CPU comparison dependencies in the existing private build venv only."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from install_dependencies import pip_command, safe_pip_environment
from dependency_probe import probe as dependency_probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--wheelhouse', type=Path)
    parser.add_argument('--allow-online', action='store_true')
    args = parser.parse_args()
    work = args.work.resolve()
    python = work / 'venv/bin/python'
    if sys.platform != 'linux' or not python.is_file():
        parser.error('An existing Linux private build venv is required')
    probe = ('import importlib.metadata as m,json,sys;'
             'print(json.dumps(dict(prefix=sys.prefix,torch=m.version("torch"),'
             'numpy=m.version("numpy"),torch_path=str(m.distribution("torch").locate_file("")),'
             'numpy_path=str(m.distribution("numpy").locate_file("")))))')
    before = json.loads(subprocess.check_output([str(python), '-c', probe], text=True))
    if (Path(before['prefix']).resolve() != work / 'venv'
            or before != json.loads((work / 'dependency-base-identities.json').read_text())):
        parser.error('Private venv or protected base identities changed')
    requirements = Path(__file__).with_name('requirements-comparison.txt')
    existing = dependency_probe(python, [requirements])
    (work / 'comparison-existing-probe.json').write_text(json.dumps(existing, ensure_ascii=False, indent=2))
    if existing['passed']:
        (work / 'comparison-dependency-install.json').write_text(json.dumps(dict(passed=True,
            mode='reused_existing', probe=existing), ensure_ascii=False, indent=2))
        print('COMPARISON_EXISTING_REUSE_PASS', flush=True)
        return
    if existing['import_errors']:
        parser.error('已有评分依赖导入失败；查看 comparison-existing-probe.json，未盲目重装')
    if not args.wheelhouse and not args.allow_online:
        parser.error('评分依赖缺少或版本不符：'+', '.join(existing['missing_or_version_mismatch'])+
                     '；请提供 --wheelhouse，仅联网准备机可显式 --allow-online')
    source = (['--no-index', '--find-links', str(args.wheelhouse)] if args.wheelhouse
              else ['--index-url', 'https://pypi.org/simple'])
    env = safe_pip_environment(work)
    subprocess.run([*pip_command(python), 'install', '--only-binary=:all:', *source,
        '--constraint', str(work / 'reviewer-base-constraints.txt'),
        '--report', str(work / 'comparison-dependency-install.json'),
        '-r', str(Path(__file__).with_name('requirements-comparison.txt'))], check=True, env=env)
    after = json.loads(subprocess.check_output([str(python), '-c', probe], text=True))
    if after != before:
        raise RuntimeError('Protected vendor Torch/NumPy identities changed')
    final_probe = dependency_probe(python, [requirements])
    (work / 'comparison-final-probe.json').write_text(json.dumps(final_probe, ensure_ascii=False, indent=2))
    if not final_probe['passed']:
        raise RuntimeError('评分依赖准备后检查未通过，停止；见 comparison-final-probe.json')
    subprocess.run([str(python), '-c', 'import av,scipy,skimage;print("COMPARISON_IMPORT_PASS")'], check=True)


if __name__ == '__main__':
    main()
