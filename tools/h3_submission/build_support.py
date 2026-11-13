"""Validate public AOT inputs and their quoted local headers, without compiling."""
import hashlib
from pathlib import Path
import re


def validate(dependencies, core=None):
    dependencies = Path(dependencies).resolve()
    manifest = {}
    for kind, count in (('qknorm', 12), ('qkrope', 8)):
        generated = dependencies / kind / 'generated'
        sources = list(generated.glob('*.cpp'))
        if len(sources) != count:
            raise ValueError(f'{kind} 应有 {count} 个公开 C++ 源文件：{generated}')
        for required in (generated / (kind + '.c'), generated / (kind + '.h'),
                         dependencies / 'abi/ninetoothed.h'):
            if not required.is_file():
                raise ValueError('缺少公开构建输入：' + str(required))
        for source in [*sources, *generated.glob('*.h'), generated / (kind + '.c')]:
            manifest[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
            for include in re.findall(r'^\s*#\s*include\s*"([^"]+)"', source.read_text(), re.M):
                if not (source.parent / include).is_file():
                    raise ValueError('AOT 本地头文件无法解析：' + str(source) + ' -> ' + include)
    if core is not None:
        generated_roots = list((Path(core) / 'build').glob('ninetoothed*'))
        if not generated_roots:
            raise ValueError('Core 缺少已提交的 build/ninetoothed* 生成源码')
        for root in generated_roots:
            for source in root.rglob('*'):
                if not source.is_file() or source.suffix not in ('.c', '.h', '.cpp'):
                    continue
                # Other system/project headers are resolved by the Core targets.
                # This verifies the preserved cross-tree ABI dependency itself.
                for include in re.findall(r'^\s*#\s*include\s*"([^"]*dependencies/abi/[^"]+)"',
                                          source.read_text(), re.M):
                    if not (source.parent / include).is_file():
                        raise ValueError('Core AOT ABI 路径无法解析：' + str(source) + ' -> ' + include)
    return manifest
