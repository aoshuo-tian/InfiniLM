import ast
import hashlib
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parent
SYNC_SHA = '3a3b91445566056be3736504f4cefd13ff8370f4a76a1d9adfc47f0efa068d60'


def source_device_copy(original_copy, tensor, destination):
    import infinicore
    import torch

    previous = torch.cuda.current_device()
    source_index = tensor.device.index
    try:
        torch.cuda.set_device(source_index)
        infinicore.set_device(tensor.device)
        torch.cuda.synchronize(source_index)
        if destination.type == 'cpu':
            return original_copy(tensor, destination)
        source_device = torch.device('cuda', source_index)
        local = original_copy(tensor, source_device)
        torch.cuda.synchronize(source_index)
        result = local.to(destination)
        torch.cuda.synchronize(destination.index)
        return result
    finally:
        infinicore.set_device(infinicore.device('cuda', previous))
        torch.cuda.set_device(previous)


def load_candidate():
    path = ROOT / 'h3_native_bridge_sync_20261107.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != SYNC_SHA:
        raise ValueError('Synchronized bridge dependency SHA mismatch')
    namespace = runpy.run_path(str(path), run_name='peer_dependency')
    original_install = namespace['install_bridge_sync']

    def install(transformer, report, output):
        runner = transformer.runner
        original_copy = runner._to_torch
        runner._to_torch = lambda tensor, device: source_device_copy(original_copy, tensor, device)
        original_install(transformer, report, output)
        report['bridge_sync']['transport'] = 'same-device InfiniCore-to-Torch copy, then Torch device transfer'
        report['bridge_sync']['first_unsynchronized_probe'] = {
            'not_run': 'Source-device transport always waits for source completion.'}
        report['bridge_sync']['candidate'] = 'producer_completion_barriers_and_source_device_torch_transport'
        (output / 'bridge-state.json').write_text(json.dumps(report['bridge_sync'], indent=2) + '\n')

    namespace['load_candidate'].__globals__['install_bridge_sync'] = install
    main = namespace['load_candidate']()
    scope = main.__globals__
    original_transform = scope['transformed_official']

    def transform(source):
        tree = original_transform(source)
        class Rename(ast.NodeTransformer):
            def visit_Constant(self, node):
                replacements = {
                    'native-bridge-sync.mp4': 'native-bridge-peer.mp4',
                    'h3-native-bridge-sync-': 'h3-native-bridge-peer-',
                    'H3_NATIVE_BRIDGE_SYNC_RESULT': 'H3_NATIVE_BRIDGE_PEER_RESULT',
                }
                if isinstance(node.value, str) and node.value in replacements:
                    return ast.copy_location(ast.Constant(replacements[node.value]), node)
                return node
        tree = ast.fix_missing_locations(Rename().visit(tree))
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'SCOPE' for target in node.targets):
                node.value = ast.Constant('Native 4+4 TP source-device Torch transport with CPU source verification; same-spec full-video quality regression, not a speed benchmark')
        return ast.fix_missing_locations(tree)

    scope.update(__file__=str(Path(__file__).resolve()), transformed_official=transform)
    return main


if __name__ == '__main__':
    load_candidate()()
