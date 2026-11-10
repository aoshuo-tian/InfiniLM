"""Stage the two source repositories for the recorded gfx936 host build.

This script does not install packages or allocate devices. --plan is read-only.
"""
from pathlib import Path
import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import sysconfig


def ignored(directory, names):
    result = []
    for name in names:
        if (name in ('.git', '__pycache__', '.cache') or name.startswith('.xmake')
                or Path(name).suffix in ('.o', '.a', '.so', '.pyc')):
            result.append(name)
    if Path(directory).name == 'build':
        result.extend(name for name in names if not name.startswith('ninetoothed'))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True, help='New, nonexistent directory')
    parser.add_argument('--xmake', default='bundled', help='bundled verified SDK, or explicit xmake 3.0.5')
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args()
    lm = Path(__file__).resolve().parents[2]
    core, work = args.core.resolve(), args.work.resolve()
    if not (core / 'src/infinicore').is_dir():
        parser.error('--core is not an InfiniCore checkout')
    if work.is_relative_to(core) or work.is_relative_to(lm):
        parser.error('--work must be outside both source checkouts')
    plan = dict(core=str(core), lm=str(lm), new_work=str(work), xmake=args.xmake,
                architecture='gfx936', install_dependencies=False, allocate_gpu=False,
                reuse_embedded_device_code=True)
    if args.plan:
        print(json.dumps(plan, indent=2))
        return
    if sys.platform != 'linux' or sys.version_info[:2] != (3, 11):
        parser.error('Build inside the Linux Hygon Python 3.11 image')
    if work.exists():
        parser.error('--work must not exist; existing directories are never overwritten')
    for header in (core / 'third_party/spdlog/include/spdlog/spdlog.h',
                   lm / 'third_party/spdlog/include/spdlog/spdlog.h',
                   lm / 'third_party/json/single_include/nlohmann/json.hpp'):
        if not header.is_file():
            parser.error('Initialize the documented submodules first: ' + str(header))
    if not Path('/opt/dtk/cuda/cuda/bin/nvcc').is_file():
        parser.error('Recorded DTK 26.04 CUDA compatibility toolchain is required')
    sdk = lm / 'tools/h3_submission/support/tooling/xmake'
    if args.xmake == 'bundled':
        import hashlib
        identity = json.loads((sdk.parent / 'xmake-sdk-manifest.json').read_text())
        for rel, digest in identity['files'].items():
            if hashlib.sha256((sdk / rel).read_bytes()).hexdigest() != digest:
                parser.error('Bundled xmake identity changed: ' + rel)
        xmake = str(work / 'tooling/xmake/bin/xmake')
    else:
        xmake = shutil.which(args.xmake)
        if not xmake:
            parser.error('xmake 3.0.5 is required')
    work.mkdir(parents=True)
    for name, source in (('InfiniCore', core), ('InfiniLM', lm)):
        shutil.copytree(source, work / 'sources' / name, ignore=ignored)
    for name in ('dependencies', 'tooling'):
        shutil.copytree(lm / 'tools/h3_submission/support' / name, work / name)
    if args.xmake == 'bundled':
        (work / 'tooling/xmake/bin/xmake').chmod(0o755)
    for directory in ('logs', 'install', 'runtime/python', 'cache/nine', 'cache/triton'):
        (work / directory).mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, '-m', 'venv', '--system-site-packages',
                    '--without-pip', str(work / 'venv')], check=True)
    py = work / 'venv/bin/python'
    staged_core, staged_lm = work / 'sources/InfiniCore', work / 'sources/InfiniLM'
    lib = staged_core / 'build/linux/x86_64/release'
    lib.mkdir(parents=True)
    for name, source in (('infinicore', staged_core / 'python/infinicore'),
                         ('infinilm', staged_lm / 'python/infinilm')):
        shutil.copytree(source, work / 'runtime/python' / name, ignore=ignored)
    for name, source in (('include', staged_core / 'include'), ('lib', lib)):
        (work / 'install' / name).symlink_to(source, target_is_directory=True)
    state = dict(work=str(work), core=str(staged_core), lm=str(staged_lm), lib=str(lib),
                 python=str(py), xmake=str(Path(xmake).resolve()))
    (work / 'state.json').write_text(json.dumps(state, indent=2))
    (work / 'protected-before.json').write_text('{}')
    (work / 'restoration-status.json').write_text(json.dumps(dict(passed=True,
        source='public source repositories', gpu_execution=False,
        reused_embedded_device_code=True)))
    include = sysconfig.get_path('include')
    if not (Path(include) / 'Python.h').is_file():
        raise RuntimeError('Python 3.11 development headers are required')
    gomp = Path(subprocess.check_output(['g++', '-print-file-name=libgomp.so.1'], text=True).strip()).resolve()
    if not gomp.is_file():
        raise RuntimeError('GCC OpenMP runtime is required')
    env = dict(H3_HEAD56_CLEAN_WORK=str(work), H3_REVIEW_WORK=str(work),
        H3_BLOCK_STAGE_WORK=str(work), H3_REVIEW_VENDOR_HEADERS='1', HYGON_ARCH='gfx936',
        INFINI_ROOT=str(work / 'install'), DTK_ROOT='/opt/dtk', HIP_PATH='/opt/dtk/hip',
        ROCM_PATH='/opt/dtk', HIP_PLATFORM='amd', XMAKE_ROOT='y', CCACHE_DISABLE='1',
        PYTHONUTF8='1', PYTHON=str(py), H3_REVIEW_PYTHON_INCLUDE=include,
        H3_REVIEW_PYTHON_LIBDIR=sysconfig.get_config_var('LIBDIR'),
        H3_REVIEW_PYTHON_LINK='python' + sysconfig.get_python_version(),
        H3_REVIEW_OPENMP_RUNTIME=str(gomp), LD_PRELOAD=str(gomp),
        NINETOOTHED_CACHE_DIR=str(work / 'cache/nine'),
        TRITON_CACHE_DIR=str(work / 'cache/triton'),
        PYTHONPATH=str(work / 'runtime/python'))
    if args.xmake == 'bundled':
        env.update(XMAKE_PROGRAM_DIR=str(work / 'tooling/xmake/share/xmake'),
                   XMAKE_GLOBALDIR=str(work / 'tooling/xmake-global'))
    env['LD_LIBRARY_PATH'] = ':'.join([str(lib), '/opt/dtk/dcc/lib', '/opt/dtk/dcc/gcvm/lib',
        '/opt/dtk/lib', '/opt/dtk/hip/lib', '/opt/dtk/cuda/cuda/lib64',
        '/opt/dtk/.hyhal/rocm_smi/lib', '/opt/dtk/lib64', '/opt/dtk/llvm/lib',
        '/opt/dtk/cuda/cuda/targets/x86_64-linux/lib',
        '/usr/local/lib/python3.11/site-packages/torch/lib'])
    setup = '\n'.join('export ' + k + '=' + shlex.quote(v) for k, v in env.items())
    setup += '\nexport PATH=' + shlex.quote(str(work / 'venv/bin') +
        ':/opt/dtk/cuda/cuda/bin:/opt/dtk/bin:/opt/dtk/llvm/bin') + ':"$PATH"\n'
    (work / 'setup.sh').write_text(setup)
    print(json.dumps(plan, indent=2))


if __name__ == '__main__':
    main()
