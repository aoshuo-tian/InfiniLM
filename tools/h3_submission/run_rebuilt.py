"""Run the frozen TP8 computation against explicitly rebuilt libraries.

The platform restorer is never called. No arguments or --plan cannot load models.
"""
import argparse
import ast
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
import types
from runtime_links import loaded_objects

IMAGE_SHA = '8dacfbb2bf2ca5593c8b33222b74e7c265380c8441a41bc09718083fe1ca3fd8'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def worker_tree(tree):
    """Only adapt external-asset paths; preserve the frozen worker's math/calls."""
    counts = dict(home=0, descriptor=0, descriptor_gate=0, allowed=0, image=0)
    worker = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'worker')

    class Paths(ast.NodeTransformer):
        def visit_Assign(self, node):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                replacements = {'home': 'output', 'descriptor': 'FLASH_DESCRIPTOR',
                                'allowed': '(FLASH_SO.parent,)', 'image': 'INPUT_IMAGE'}
                if name in replacements:
                    counts[name] += 1
                    node.value = ast.parse(replacements[name], mode='eval').body
            return self.generic_visit(node)

        def visit_If(self, node):
            if ast.unparse(node.test) == "descriptor.parent != home / 'h3-official-flash-cache'":
                counts['descriptor_gate'] += 1
                return ast.copy_location(ast.parse('check_descriptor(descriptor)').body[0], node)
            return self.generic_visit(node)

    Paths().visit(worker)
    if any(value != 1 for value in counts.values()):
        raise ValueError('Frozen worker asset anchors changed: ' + repr(counts))
    # The old platform entry and asset-cache discovery are deliberately absent.
    tree.body = [x for x in tree.body
                 if not (isinstance(x, ast.FunctionDef) and x.name in ('main', 'resolve_current_flash'))
                 and not (isinstance(x, ast.If) and '__name__' in ast.unparse(x.test))]
    return ast.fix_missing_locations(tree), counts


def make_worker(backend='native'):
    repo = Path(__file__).resolve().parents[2]
    profile = repo / 'examples/h3_submission/tp8_20261109'
    sources = profile / 'profile-sources'
    sys.path.insert(0, str(profile))
    sys.path.insert(0, str(sources))
    import h3_profile_768x928 as base
    import h3_tp8_profile as tp8
    function, tree, original_checks = base.build(backend, sources / 'reviewer_worker.py')
    if backend == 'native':
        tree, tp8_checks = tp8.tp8_tree(tree)
    elif backend == 'reference':
        tp8_checks = dict(reference_legacy_device_map=True)
    else:
        raise ValueError('Unknown backend')
    tree, path_checks = worker_tree(tree)
    cache = types.ModuleType('h3_native_cache_budget')
    cache.__file__ = str(sources / 'h3_native_cache_budget.py')
    exec(compile(tp8.cache_tree(Path(cache.__file__).read_text(encoding='utf-8')),
                 '<reviewer-tp8-cache>', 'exec'), cache.__dict__)
    sys.modules[cache.__name__] = cache
    namespace = dict(function.__globals__)
    namespace.update(__name__='reviewer_frozen_worker', __file__=str(sources / 'reviewer_worker.py'),
                     load_native_entry=tp8.original_entry)
    exec(compile(tree, '<reviewer-frozen-worker>', 'exec'), namespace)
    if 'main' in namespace:
        # Copied outer globals can contain a main; it is not an execution route.
        namespace.pop('main')
    return namespace, dict(profile=original_checks, tp8=tp8_checks, paths=path_checks)


def check_loaded_paths(work, state):
    import torch
    lib = Path(state['lib']).resolve()
    for name in ('libinfinirt.so', 'libinfiniccl.so'):
        path = lib / name
        if not path.is_file():
            raise RuntimeError('Newly built library missing: ' + str(path))
        ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    import infinicore
    import infinilm
    packages = dict(infinicore=Path(infinicore.__file__).resolve(),
                    infinilm=Path(infinilm.__file__).resolve())
    for path in packages.values():
        if not path.is_relative_to(work):
            raise RuntimeError('Package came from outside the new build: ' + str(path))
    names = ('libinfinicore_cpp_api.so', 'libinfiniop.so', 'libinfinirt.so',
             'libinfiniccl.so', '_infinicore.', '_infinilm.')
    maps = loaded_objects()
    for name in names:
        matched = [Path(value).resolve() for value in maps if name in Path(value).name]
        if not matched or any(not path.is_relative_to(work) for path in matched):
            raise RuntimeError('Missing or foreign build library: ' + name + ': ' + repr(matched))
    return dict(python=sys.executable, torch=torch.__version__,
                packages={key: str(value) for key, value in packages.items()},
                mapped_libraries={value: sha(value) for value in maps})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--first-frame', type=Path)
    parser.add_argument('--flash-so', type=Path)
    parser.add_argument('--flash-sha256')
    parser.add_argument('--backend', choices=('native', 'reference'), default='native')
    parser.add_argument('--verify-only', action='store_true', help='Check full packages/new libraries on DCU, without loading weights')
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args()
    namespace, anchors = make_worker(args.backend)
    if args.plan or len(sys.argv) == 1:
        print(json.dumps(dict(mode='plan_only', anchors=anchors,
            parameters=namespace['PARAMETERS'], uses_old_runtime_archive=False,
            platform_restorer_available=False, gpu_execution=False), indent=2))
        return
    for name in ('work', 'model', 'output', 'first_frame', 'flash_so', 'flash_sha256'):
        if getattr(args, name) is None:
            parser.error('--' + name.replace('_', '-') + ' is required for execution')
    if sys.platform != 'linux':
        parser.error('Run inside the specified Linux Hygon image')
    work, output = args.work.resolve(), args.output.resolve()
    if output.exists():
        parser.error('Output directory must not exist; existing evidence is never overwritten')
    state = json.loads((work / 'state.json').read_text())
    if Path(state['work']).resolve() != work:
        parser.error('Build state belongs to a different work directory')
    build = json.loads((work / 'build-status.json').read_text())
    ccl = json.loads((work / 'ccl-build-status.json').read_text())
    if not (build.get('passed') and build.get('clean_host_framework')
            and ccl.get('passed') and ccl.get('complete')):
        parser.error('A completed clean host-framework build and CCL build are required')
    for path, digest in ccl['libraries'].items():
        if not Path(path).resolve().is_relative_to(work) or sha(path) != digest:
            parser.error('Rebuilt CCL library identity changed: ' + path)
    if sha(args.first_frame) != IMAGE_SHA:
        parser.error('Input image identity differs from the frozen comparison profile')
    if sha(args.flash_so) != args.flash_sha256:
        parser.error('Vendor Flash library SHA mismatch')
    if not args.model.is_dir():
        parser.error('Model directory unavailable')
    output.mkdir(parents=True)
    loaded = (check_loaded_paths(work, state) if args.backend == 'native'
              else dict(backend='PyTorch legacy reference', native_framework_not_loaded=True))
    if args.backend == 'native':
        expected = {**build['libraries'], **ccl['libraries']}
        for path, digest in loaded['mapped_libraries'].items():
            if expected.get(str(Path(path).resolve())) != digest:
                raise RuntimeError('Loaded library differs from this build manifest: ' + path)
    (output / 'loaded-rebuilt-libraries.json').write_text(json.dumps(loaded, indent=2))
    if args.verify_only:
        print(json.dumps(dict(passed=True, verification=loaded, model_loaded=False), indent=2))
        return
    flash = dict(passed=True, flash_so=str(args.flash_so.resolve()),
                 flash_so_sha256=args.flash_sha256,
                 flash_so_bytes=args.flash_so.stat().st_size,
                 source='Explicit external vendor library with caller-supplied SHA256')
    descriptor = output / 'external-flash.json'
    descriptor.write_text(json.dumps(flash, indent=2))

    def check_descriptor(path):
        if Path(path).resolve() != descriptor or json.loads(Path(path).read_text()) != flash:
            raise ValueError('Explicit Flash descriptor changed')

    namespace.update(FLASH_DESCRIPTOR=descriptor, FLASH_SO=args.flash_so.resolve(),
                     INPUT_IMAGE=args.first_frame.resolve(), check_descriptor=check_descriptor)
    (output / 'reviewer-entry.json').write_text(json.dumps(dict(anchors=anchors,
        old_runtime_restored=False, model=str(args.model.resolve()),
        first_frame_sha256=IMAGE_SHA, parameters=namespace['PARAMETERS']), indent=2))
    code = namespace['worker'](work, output, args.model.resolve())
    raise SystemExit(code)


if __name__ == '__main__':
    main()
