"""Rebuild CCL using the recovered successful configure flags and target.

Historical provenance does not mean a new clean build has already passed.
This script records its own commands and resulting library identities.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import hashlib
import time
from build_monitor import run_logged, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args()
    work = Path(os.environ['H3_HEAD56_CLEAN_WORK'])
    state = json.loads((work / 'state.json').read_text())
    previous = json.loads((work / 'build-status.json').read_text())
    configure = previous['status']['core-configure.log']['command']
    if configure.count('--ccl=n') != 1:
        raise RuntimeError('Expected the recorded CCL-off framework configure command')
    flags = [('--ccl=y' if word == '--ccl=n' else word) for word in configure[5:]]
    provenance = json.loads((Path(__file__).resolve().parents[2] /
                            'h3_submission/ccl-build-provenance.json').read_text())
    if flags != provenance['flags']:
        raise RuntimeError('Configure flags differ from the recovered successful CCL build')
    commands = [[('--ccl=y' if word == '--ccl=n' else word) for word in configure],
                [state['xmake'], 'build', '-j', '4', '-v', 'infiniccl']]
    if args.plan:
        print(json.dumps(dict(commands=commands, cwd=state['core'], HYGON_ARCH='gfx936',
                             historical_command_recovered=True), indent=2))
        return
    if sys.platform != 'linux' or os.environ.get('HYGON_ARCH') != 'gfx936':
        raise RuntimeError('Source setup.sh first; Linux/gfx936 is required')
    record = dict(complete=False, passed=False, commands=[], historical_command_recovered=True)
    target = work / 'ccl-build-status.json'
    try:
        core = Path(state['core'])
        for name in ('infinirt', 'infinirt-cpu', 'infinirt-hygon', 'infini-utils',
                     'infiniccl', 'infiniccl-hygon'):
            for suffix in ('*.cc', '*.cpp', '*.cu'):
                for source in (core / 'src').rglob(suffix):
                    (core / 'build/.objs' / name / 'linux/x86_64/release' /
                     source.relative_to(core).parent).mkdir(parents=True, exist_ok=True)
        for index, command in enumerate(commands):
            def progress(value):
                record['current'] = value
                write_json(target, record)
            result = run_logged(command, work / 'logs' / ('ccl-source-build-' + str(index) + '.log'),
                                cwd=core, on_progress=progress)
            record['commands'].append(result)
            if result['exit_code']:
                raise RuntimeError('CCL build stage failed; see the corresponding log')
        lib = Path(state['lib']) / 'libinfiniccl.so'
        symbols = subprocess.check_output(['nm', '-D', str(lib)], text=True)
        if 'ncclAllReduce' not in symbols or 'ncclAllGather' not in symbols:
            raise RuntimeError('Built CCL library does not reference required collectives')
        record['libraries'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (lib, Path(state['lib']) / 'libinfinirt.so')}
        record['passed'] = True
    finally:
        record['complete'] = True
        write_json(target, record)
    print('CCL source build completed; device execution is still a separate validation step.')


if __name__ == '__main__':
    main()
