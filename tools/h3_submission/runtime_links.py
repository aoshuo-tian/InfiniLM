"""Connect new build outputs without replacing the InfiniLM Python package.

The native-only probe is suitable for the shared CPU build node. It does not
establish full-package import, DCU communication, or video correctness.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys


LIBRARIES = ('libinfinicore_cpp_api.so', 'libinfiniop.so', 'libinfinirt.so',
             'libinfiniccl.so', '_infinicore.', '_infinilm.')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def link_plan(work, state):
    work = Path(work).resolve()
    lib, extension = Path(state['lib']).resolve(), Path(state['extension']).resolve()
    if not all(p.is_relative_to(work) for p in (lib, extension)):
        raise ValueError('Build outputs must stay within the new work directory')
    if not lib.is_dir() or not extension.is_file():
        raise ValueError('New build library directory or LM extension missing')
    packages = work / 'runtime/python'
    lm_package = packages / 'infinilm/lib'
    if lm_package.is_symlink() or not (lm_package / '__init__.py').is_file():
        raise ValueError('Preserve the real infinilm/lib package and its __init__.py')
    return [(packages / 'infinicore/lib', lib), (lm_package / extension.name, extension)]


def ensure_relative_link(link, target):
    if link.is_symlink():
        if link.resolve() != target.resolve():
            raise ValueError('Refusing to replace an unrelated link: ' + str(link))
        if not Path(os.readlink(link)).is_absolute():
            return
        # Only the already-verified link is replaced, never its target.
        link.unlink()
    elif link.exists():
        raise ValueError('Refusing to replace an existing file/directory: ' + str(link))
    link.symlink_to(os.path.relpath(target, link.parent), target_is_directory=target.is_dir())


def connect(work, state):
    plan = link_plan(work, state)
    initializer = Path(work) / 'runtime/python/infinilm/lib/__init__.py'
    before = initializer.read_bytes()
    # Reject every collision before changing any link.
    for link, target in plan:
        if ((link.is_symlink() and link.resolve() != target.resolve())
                or (link.exists() and not link.is_symlink())):
            raise ValueError('Runtime link collision: ' + str(link))
    for link, target in plan:
        ensure_relative_link(link, target)
    if initializer.read_bytes() != before:
        raise RuntimeError('InfiniLM package initializer changed')
    return {str(link): os.readlink(link) for link, _ in plan}


def loaded_objects():
    """Read loader metadata without depending on access to /proc/self/maps."""
    class Info(ctypes.Structure):
        _fields_ = [('address', ctypes.c_void_p), ('name', ctypes.c_char_p)]
    paths = []
    callback_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(Info),
                                    ctypes.c_size_t, ctypes.c_void_p)

    @callback_type
    def collect(info, size, data):
        if info.contents.name:
            paths.append(info.contents.name.decode())
        return 0

    ctypes.CDLL(None).dl_iterate_phdr(collect, None)
    return sorted({str(Path(path).resolve()) for path in paths
                   if any(name in Path(path).name for name in LIBRARIES)})


def verify_native(work):
    work = Path(work).resolve()
    state = json.loads((work / 'state.json').read_text())
    lib = Path(state['lib']).resolve()
    extension = Path(state['extension']).resolve()
    sys.path[:0] = [str(lib), str(extension.parent)]
    import _infinicore
    import _infinilm
    paths = loaded_objects()
    required = [name for name in LIBRARIES if name != 'libinfiniccl.so']
    if (not all(any(name in Path(p).name for p in paths) for name in required)
            or any(not Path(p).is_relative_to(work) for p in paths)):
        raise RuntimeError('Missing or foreign native build output: ' + repr(paths))
    for module in (_infinicore, _infinilm):
        if not Path(module.__file__).resolve().is_relative_to(work):
            raise RuntimeError('Native extension loaded outside the new work directory')
    return dict(passed=True, scope='native-extensions-only', gpu_execution=False,
                full_package_import_validated=False,
                loaded_libraries={p: sha(p) for p in paths})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-native', type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('The native probe requires the target Linux build environment')
    print(json.dumps(verify_native(args.verify_native), indent=2), flush=True)


if __name__ == '__main__':
    main()
