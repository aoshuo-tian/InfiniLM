"""Install CPU comparison dependencies in the existing private build venv only."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from install_dependencies import pip_command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--wheelhouse', type=Path)
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
    source = (['--no-index', '--find-links', str(args.wheelhouse)] if args.wheelhouse
              else ['--index-url', 'https://pypi.org/simple'])
    env = dict(os.environ, PIP_REQUIRE_VIRTUALENV='true', PIP_DISABLE_PIP_VERSION_CHECK='1')
    subprocess.run([*pip_command(python), 'install', '--only-binary=:all:', *source,
        '--constraint', str(work / 'reviewer-base-constraints.txt'),
        '--report', str(work / 'comparison-dependency-install.json'),
        '-r', str(Path(__file__).with_name('requirements-comparison.txt'))], check=True, env=env)
    after = json.loads(subprocess.check_output([str(python), '-c', probe], text=True))
    if after != before:
        raise RuntimeError('Protected vendor Torch/NumPy identities changed')
    subprocess.run([str(python), '-c', 'import av,scipy,skimage;print("COMPARISON_IMPORT_PASS")'], check=True)


if __name__ == '__main__':
    main()
