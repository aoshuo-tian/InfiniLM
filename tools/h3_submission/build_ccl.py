"""Source-derived CCL-on build step, not a recovered historical command.

The tested runtime used a separately built CCL-on overlay. This packaging-time
replacement recipe has not been executed on DCU. It fails on build/import errors;
it does not run models, qualification tests or video generation.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


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
    commands = [[('--ccl=y' if word == '--ccl=n' else word) for word in configure],
                [state['xmake'], 'build', '-j', '4', '-v', 'infiniccl'],
                [state['xmake'], 'build', '-j', '4', '-v', 'infinirt']]
    if args.plan:
        print(json.dumps(dict(commands=commands, cwd=state['core'], HYGON_ARCH='gfx936',
                             historical_command_recovered=False), indent=2))
        return
    if sys.platform != 'linux' or os.environ.get('HYGON_ARCH') != 'gfx936':
        raise RuntimeError('Source setup.sh first; Linux/gfx936 is required')
    for index, command in enumerate(commands):
        with (work / 'logs' / ('ccl-source-build-' + str(index) + '.log')).open('w') as log:
            subprocess.run(command, cwd=state['core'], stdout=log,
                           stderr=subprocess.STDOUT, check=True)
    print('CCL source build completed; run CCL qualification before claiming TP validation.')


if __name__ == '__main__':
    main()
