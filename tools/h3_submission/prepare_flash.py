"""Verify the official vendor wheel and extract only its Flash shared library."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile

WHEEL_SHA256 = '8546e46bcbfcd346757c67a0ff180ace2a49716d9724d52ee870d04b4c3b41d1'
WHEEL_BYTES = 658898543


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def prepare(wheel, output):
    wheel, output = Path(wheel).resolve(), Path(output).resolve()
    if len(WHEEL_SHA256) != 64 or wheel.stat().st_size != WHEEL_BYTES or sha(wheel) != WHEEL_SHA256:
        raise ValueError('Official Flash wheel size/SHA256 mismatch')
    if output.exists():
        record = json.loads((output / 'prepare.json').read_text())
        so = output / record['filename']
        if record['wheel_sha256'] != WHEEL_SHA256 or sha(so) != record['flash_so_sha256']:
            raise ValueError('Existing Flash cache differs; it will not be overwritten')
        return record
    with zipfile.ZipFile(wheel) as archive:
        candidates = [name for name in archive.namelist()
                      if PurePosixPath(name).name.startswith('flash_attn_2_cuda') and name.endswith('.so')]
        if len(candidates) != 1:
            raise ValueError('Expected exactly one vendor FlashAttention v2 shared library')
        name = candidates[0]
        output.mkdir(parents=True)
        so = output / PurePosixPath(name).name
        with archive.open(name) as source, so.open('xb') as target:
            shutil.copyfileobj(source, target)
    record = dict(passed=True, source='SHA256-verified vendor Flash wheel; shared library only',
                  wheel_sha256=WHEEL_SHA256, wheel_bytes=WHEEL_BYTES, filename=so.name,
                  flash_so=str(so), flash_so_sha256=sha(so), flash_so_bytes=so.stat().st_size)
    (output / 'prepare.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.wheel, args.output), indent=2))


if __name__ == '__main__':
    main()
