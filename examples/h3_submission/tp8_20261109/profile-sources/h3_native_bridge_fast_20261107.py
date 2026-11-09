import ast
import functools
import hashlib
from pathlib import Path
import runpy
import time

ROOT = Path(__file__).resolve().parent
ALIGNED_SHA = '271cb1dee6283bf4b51aa224fcfb93f8629984386984f6c00c8bc3907982a9b3'
PEER_SHA = '2f3850c1880759bcf74e1df7f0b8fafcfef433070597f4181d6b7b361a51d5a7'


def checked_namespace(name, expected):
    path = ROOT / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('Dependency SHA mismatch: ' + name)
    return runpy.run_path(str(path), run_name='bridge_fast_dependency')


def load_candidate():
    namespace = checked_namespace('h3_native_aligned_20261107.py', ALIGNED_SHA)
    peer = checked_namespace('h3_native_bridge_peer_20261107.py', PEER_SHA)
    transport = peer['source_device_copy']
    original_audit = namespace['audit_pipeline']
    original_transform = namespace['transformed_official']

    def audit(pipe, encoder, transformer, report):
        import torch

        original_audit(pipe, encoder, transformer, report)
        runner = transformer.runner
        original_forward = runner.engine.forward_raw
        original_copy = runner._to_torch
        state = {'transport': 'validated_source_device_torch_transport',
                 'forward_calls': 0, 'copy_calls': 0, 'producer_wait_seconds': 0.0,
                 'copy_seconds': 0.0, 'cpu_source_diagnostic_copies': False,
                 'full_boundary_statistics': False,
                 'quality_gate_passed': False,
                 'performance_scope': 'single full-video regression; compare decoded media before drawing conclusions'}
        report['bridge_fast'] = state

        @functools.wraps(original_forward)
        def forward(*args, **kwargs):
            indices = {tensor.device.index for tensor in kwargs.values()
                       if tensor is not None and hasattr(tensor, 'device')
                       and str(tensor.device.type) != 'cpu'}
            started = time.perf_counter()
            for index in sorted(indices):
                torch.cuda.synchronize(index)
            state['producer_wait_seconds'] += time.perf_counter() - started
            state['forward_calls'] += 1
            return original_forward(*args, **kwargs)

        def copy(tensor, device):
            started = time.perf_counter()
            result = transport(original_copy, tensor, device)
            state['copy_calls'] += 1
            state['copy_seconds'] += time.perf_counter() - started
            return result

        runner.engine.forward_raw = forward
        runner._to_torch = copy

    def transform(source):
        tree = original_transform(source)
        class Rename(ast.NodeTransformer):
            def visit_Constant(self, node):
                replacements = {
                    'native-aligned.mp4': 'native-bridge-fast.mp4',
                    'h3-native-aligned-': 'h3-native-bridge-fast-',
                    'H3_NATIVE_ALIGNED_RESULT': 'H3_NATIVE_BRIDGE_FAST_RESULT',
                }
                if isinstance(node.value, str) and node.value in replacements:
                    return ast.copy_location(ast.Constant(replacements[node.value]), node)
                return node
        tree = ast.fix_missing_locations(Rename().visit(tree))
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'SCOPE' for target in node.targets):
                node.value = ast.Constant('Native 4+4 TP validated source-device Torch transport without diagnostic CPU copies; complete same-spec regression')
        return ast.fix_missing_locations(tree)

    namespace['main'].__globals__.update(__file__=str(Path(__file__).resolve()),
                                         audit_pipeline=audit, transformed_official=transform)
    return namespace['main']


if __name__ == '__main__':
    load_candidate()()
