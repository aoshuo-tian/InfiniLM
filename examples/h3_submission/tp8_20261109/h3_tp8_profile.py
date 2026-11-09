"""Same frozen cache computation, original generator entry, shared eight-rank groups."""
import ast
import hashlib
import json
from pathlib import Path
import runpy
import sys
import types

ENTRY_SHA = 'ca54a18a73a7d760d644ca4eb2c3b20678f88dfdbcfa26a62dcd3de617b31af4'
DEVICES = list(range(8))


def cache_tree(source):
    tree = ast.parse(source)
    count = 0
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'cache_shapes':
            class Ranks(ast.NodeTransformer):
                def visit_Compare(self, value):
                    nonlocal count
                    if ast.unparse(value) == 'len(shapes) != 4':
                        count += 1
                        return ast.copy_location(ast.parse('len(shapes) != 8', mode='eval').body, value)
                    return self.generic_visit(value)

                def visit_Constant(self, value):
                    if value.value == 'Expected four ranks of 50 conditioner KV layers':
                        return ast.copy_location(ast.Constant('Expected eight ranks of 50 conditioner KV layers'), value)
                    return value
            Ranks().visit(node)
    if count != 1:
        raise ValueError('Frozen KV rank guard changed')
    return ast.fix_missing_locations(tree)


def tp8_tree(tree):
    counts = dict(parameters=0, placement=0, source_flag=0, source_note=0,
                  scope=0, video=0, directory=0)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [v.id for v in node.targets if isinstance(v, ast.Name)]
            if 'PARAMETERS' in names:
                params = ast.literal_eval(node.value)
                if (params['native_text_gpus'] != [0, 1, 2, 3]
                        or params['native_transformer_gpus'] != [4, 5, 6, 7]):
                    raise ValueError('Original split profile changed')
                params.update(native_text_gpus=DEVICES, native_transformer_gpus=DEVICES)
                node.value = ast.parse(repr(params), mode='eval').body
                counts['parameters'] += 1
            if 'SCOPE' in names:
                node.value = ast.Constant('Shared TP8 native cache; original generator entry; 768x928, 50 requested steps; original BF16 VAE and vendor Flash')
                counts['scope'] += 1

    class Adapt(ast.NodeTransformer):
        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id == 'native_placement':
                if ast.literal_eval(node.args[1]) not in ([0, 1, 2, 3], [4, 5, 6, 7]):
                    raise ValueError('Native placement audit changed')
                node.args[1] = ast.parse(repr(DEVICES), mode='eval').body
                counts['placement'] += 1
            return self.generic_visit(node)

        def visit_Dict(self, node):
            for i, key in enumerate(node.keys):
                if isinstance(key, ast.Constant) and key.value == 'no_model_source_patch':
                    if ast.literal_eval(node.values[i]) is not False:
                        raise ValueError('Source patch record changed')
                    node.values[i] = ast.Constant(True)
                    counts['source_flag'] += 1
                if isinstance(key, ast.Constant) and key.value == 'isolated_patch':
                    if ast.literal_eval(node.values[i]) != 'two native TP device expressions only':
                        raise ValueError('Source patch description changed')
                    node.values[i] = ast.Constant('Original generator source unchanged; source-device bridge and bounded KV wrappers retained')
                    counts['source_note'] += 1
            return self.generic_visit(node)

        def visit_Constant(self, node):
            replacements = {
                'native-cache-768x928-s50.mp4': ('video', 'native-tp8-768x928-s50.mp4'),
                'h3-native-cache-768x928-s50-': ('directory', 'h3-native-tp8-768x928-s50-'),
            }
            if isinstance(node.value, str) and node.value in replacements:
                label, value = replacements[node.value]
                counts[label] += 1
                return ast.copy_location(ast.Constant(value), node)
            return node

    tree = ast.fix_missing_locations(Adapt().visit(tree))
    expected = dict(parameters=1, placement=2, source_flag=1, source_note=1,
                    scope=1, video=1, directory=1)
    if counts != expected:
        raise ValueError('TP8 wrapper anchors changed: ' + repr(counts))
    return tree, counts


def original_entry(entry, output):
    data = entry.read_bytes()
    if hashlib.sha256(data).hexdigest() != ENTRY_SHA:
        raise ValueError('Original official generator identity mismatch')
    record = {'entry_sha256': ENTRY_SHA, 'source_modified': False,
              'text_tp_device_ids': DEVICES, 'transformer_tp_device_ids': DEVICES,
              'scope': 'Original entry shared devices; wrappers retain cache allocation and device transport only'}
    import torch
    record['starting_memory'] = [dict(device=i, free_bytes=torch.cuda.mem_get_info(i)[0],
                                     total_bytes=torch.cuda.get_device_properties(i).total_memory)
                                 for i in DEVICES]
    (output / 'tp8-entry-and-memory.json').write_text(json.dumps(record, indent=2))
    print('TP8_ORIGINAL_ENTRY', json.dumps(record), flush=True)
    return runpy.run_path(str(entry), run_name='h3_original_shared_tp8_entry')


def main(entry_file):
    import h3_profile_768x928 as base
    caller = Path(entry_file).resolve()
    if caller.parent not in (base.ROOT.resolve(), base.ROOT.parent.resolve()):
        raise ValueError('Unexpected TP8 entry location')
    # The frozen environment launcher resolves its source siblings from __file__.
    # Both the platform parent and worker therefore enter through this inner file.
    inner = base.ROOT / 'h3_tp8_worker_entry_20261109.py'
    inner_source = ("from pathlib import Path\nimport sys\n"
                    "sys.path.insert(0,str(Path(__file__).resolve().parent.parent))\n"
                    "from h3_tp8_profile import main\nmain(__file__)\n")
    if inner.exists() and inner.read_text() != inner_source:
        raise ValueError('Existing TP8 worker entry differs')
    inner.write_text(inner_source, encoding='utf-8')
    base.PROFILE.update(platform_seconds=1800, worker_seconds=1740)
    function, inherited_tree, inherited_counts = base.build('native', inner.resolve())
    # The cache module is imported inside the worker. Only its four-rank guard
    # changes; allocation/growth policy, dtype and model calls stay byte-equivalent.
    module = types.ModuleType('h3_native_cache_budget')
    module.__file__ = str(base.ROOT / 'h3_native_cache_budget.py')
    tree = cache_tree(Path(module.__file__).read_text())
    exec(compile(tree, '<tp8-cache-rank-guard>', 'exec'), module.__dict__)
    sys.modules[module.__name__] = module
    scope = function.__globals__
    inherited_transform = scope['transformed_official']

    def transform(source):
        return tp8_tree(inherited_transform(source))[0]

    derived_tree, counts = tp8_tree(inherited_tree)
    scope.update(transformed_official=transform, load_native_entry=original_entry)
    if len(sys.argv) == 1:
        print(json.dumps({'mode': 'plan_only', 'profile': base.PROFILE,
                          'native_text_gpus': DEVICES, 'native_transformer_gpus': DEVICES,
                          'expected_transformer_calls': 49, 'generator_sha256': ENTRY_SHA,
                          'worker_entry': str(inner.resolve()),
                          'new_model_math': False, 'inherited_checks': inherited_counts,
                          'tp8_checks': counts, 'kv_ranks': 8, 'new_gpu_jobs': 0}))
        return
    if sys.argv[1] not in ('--platform', '--worker'):
        raise ValueError('Expected no arguments, --platform or --worker')
    return function()
