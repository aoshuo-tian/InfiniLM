"""Prepare canonical vendor Flash by reusing an existing .so or verified wheel.

File identity is not a device execution test. The frozen DCU Flash gate remains
mandatory. This module does not import Flash, install packages or run kernels.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sysconfig
import tempfile
import zipfile

WHEEL_SHA256='8546e46bcbfcd346757c67a0ff180ace2a49716d9724d52ee870d04b4c3b41d1'
WHEEL_BYTES=658898543
SO_SHA256='a56d9b54127d86777b55b41e06b55d246897f7ea202048af8413d9d65f1f4395'
SO_BYTES=2088777840
SO_FILENAME='flash_attn_2_cuda.cpython-311-x86_64-linux-gnu.so'


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def verify_so(path):
    path=Path(path).resolve()
    if not path.is_file() or path.stat().st_size!=SO_BYTES or sha(path)!=SO_SHA256:
        raise ValueError('现有Flash动态库与已验证厂商版本不一致：'+str(path))
    with path.open('rb') as stream:
        header=stream.read(20)
    if (header[:6]!=b'\x7fELF\x02\x01' or len(header)!=20
            or int.from_bytes(header[18:20],'little')!=62):
        raise ValueError('需要Linux ELF64 x86_64厂商Flash动态库')
    return path


def candidates():
    result=[os.environ[name] for name in ('FA_SO_PATH','FLASH_ATTN_PREBUILT') if os.environ.get(name)]
    try:
        spec=importlib.util.find_spec('flash_attn_2_cuda')
    except (ImportError,ValueError):
        spec=None
    if spec and spec.origin:
        result.append(spec.origin)
    for name in ('platlib','purelib'):
        folder=sysconfig.get_path(name)
        if folder:
            result.append(str(Path(folder)/SO_FILENAME))
    return list(dict.fromkeys(result))


def prepare(wheel, output, flash_so=None, discover=False):
    output=Path(output).resolve()
    explicit=verify_so(flash_so) if flash_so else None
    if output.exists():
        receipt=output/'prepare.json'
        if not receipt.is_file():
            raise ValueError('输出目录已有其他内容，不覆盖；请选择新的输出目录')
        record=json.loads(receipt.read_text())
        verify_so(record['flash_so'])
        if not record.get('passed') or record.get('flash_so_sha256')!=SO_SHA256:
            raise ValueError('已有Flash清单无效，不覆盖')
        return record
    found=explicit
    rejected=[]
    if not found and discover:
        for candidate in candidates():
            if not Path(candidate).is_file():
                continue
            try:
                found=verify_so(candidate)
                break
            except ValueError as error:
                rejected.append(str(error))
    if found:
        output.mkdir(parents=True)
        record=dict(passed=True,source='现有动态库，逐字节身份匹配已验证厂商版本',
            canonical_wheel_sha256=WHEEL_SHA256,filename=found.name,flash_so=str(found),
            flash_so_sha256=SO_SHA256,flash_so_bytes=SO_BYTES,reused_existing=True,
            wheel_required=False,abi_execution_validated=False,gpu_kernel_executed=False)
    else:
        if wheel is None:
            raise ValueError('未找到可复用的已验证Flash动态库。请用 --flash-so 指定现有库，'
                '或用 --wheel 提供完整厂商wheel；没有安装或下载任何内容。'+repr(rejected))
        wheel=Path(wheel).resolve()
        if not wheel.is_file() or wheel.stat().st_size!=WHEEL_BYTES or sha(wheel)!=WHEEL_SHA256:
            raise ValueError('Official Flash wheel size/SHA256 mismatch')
        output.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(wheel) as archive:
            names=[name for name in archive.namelist()
                   if PurePosixPath(name).name.startswith('flash_attn_2_cuda') and name.endswith('.so')]
            if len(names)!=1:
                raise ValueError('Expected exactly one vendor FlashAttention v2 shared library')
            with tempfile.TemporaryDirectory(prefix='flash-prepare-',dir=output.parent) as tmp:
                staged=Path(tmp)/PurePosixPath(names[0]).name
                with archive.open(names[0]) as source, staged.open('xb') as target:
                    shutil.copyfileobj(source,target)
                verify_so(staged)
                if output.exists():
                    raise ValueError('输出目录在准备期间已出现，停止，不覆盖')
                Path(tmp).rename(output)
        found=output/PurePosixPath(names[0]).name
        record=dict(passed=True,source='SHA256-verified vendor Flash wheel; shared library only',
            wheel_sha256=WHEEL_SHA256,wheel_bytes=WHEEL_BYTES,filename=found.name,
            flash_so=str(found),flash_so_sha256=SO_SHA256,flash_so_bytes=SO_BYTES,
            reused_existing=False,wheel_required=True,abi_execution_validated=False,gpu_kernel_executed=False)
    (output/'prepare.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel',type=Path)
    parser.add_argument('--flash-so',type=Path,help='已有厂商动态库；严格核对已验证版本SHA')
    parser.add_argument('--discover',action='store_true',help='仅检查当前解释器包目录与显式环境变量')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(prepare(args.wheel,args.output,args.flash_so,args.discover),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
