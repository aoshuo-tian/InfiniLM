"""Isolated profile adaptation of SHA-pinned reference/cache launchers; VAE math untouched."""
import ast
import hashlib
import json
from pathlib import Path
import runpy
import sys
import types

ROOT = Path(__file__).resolve().parent
PROFILE = {'prompt': 'A dog falls asleep', 'seed': 42, 'steps': 50,
           'width': 768, 'height': 928, 'num_frames': 124, 'fps': 24,
           'gpus': list(range(8)), 'platform_seconds': 3600,
           'worker_seconds': 3300, 'auto_retry': False}
EXPECTED_FORWARDS = 49


def checked_sources():
    manifest = json.loads((ROOT / 'frozen-manifest.json').read_text())
    for item in manifest['files']:
        data = (ROOT / item['file']).read_bytes()
        if hashlib.sha256(data).hexdigest() != item['sha256']:
            raise ValueError('Frozen baseline source/image identity changed: ' + item['file'])


def profile_checks_tree():
    tree = ast.parse((ROOT / 'h3_competition_checks.py').read_text())
    counts = {'profile': 0, 'calls': 0, 'raw_shape': 0}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'PROFILE' for t in node.targets):
            # PROFILE uses PROMPT, so preserve its literal expression then change only the three numeric fields.
            for key, value in zip(node.value.keys, node.value.values):
                if isinstance(key, ast.Constant) and key.value in ('steps', 'width', 'height'):
                    value.value = PROFILE[key.value]
                    counts['profile'] += 1
        if isinstance(node, ast.FunctionDef) and node.name in ('run_single_entry', 'validate_generation_record'):
            class Checks(ast.NodeTransformer):
                def visit_Constant(self, value):
                    if type(value.value) is int and value.value == 9:
                        counts['calls'] += 1
                        return ast.copy_location(ast.Constant(EXPECTED_FORWARDS), value)
                    return value

                def visit_List(self, value):
                    if ast.dump(value, include_attributes=False) == ast.dump(ast.parse('[124,384,576,3]', mode='eval').body, include_attributes=False):
                        counts['raw_shape'] += 1
                        return ast.copy_location(ast.parse('[124,928,768,3]', mode='eval').body, value)
                    return self.generic_visit(value)
            Checks().visit(node)
    if counts != {'profile': 3, 'calls': 6, 'raw_shape': 1}:
        raise ValueError('Profile check anchors changed: ' + repr(counts))
    return ast.fix_missing_locations(tree), counts


def adapt_tree(tree, backend):
    """Change configuration, resource bounds, shape/call guards and optional preview output only."""
    counts = {'parameters': 0, 'command': 0, 'frame_shape': 0,
              'native_calls': 0, 'timeout': 0, 'png_preview_loop': 0}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == 'PARAMETERS':
                    p = ({'legacy_text_gpus': [0, 1, 2, 3], 'legacy_transformer_gpus': [4, 5, 6, 7]}
                         if backend == 'reference' else ast.literal_eval(node.value))
                    p.update(PROFILE, legacy=backend == 'reference')
                    node.value = ast.parse(repr(p), mode='eval').body
                    counts['parameters'] += 1
                if isinstance(target, ast.Name) and target.id == 'SCOPE':
                    node.value = ast.Constant('FL2VA 768x928, 50 requested steps, 124 frames; pinned original BF16 VAE/vendor Flash; ' + backend)
        if isinstance(node, ast.FunctionDef) and node.name == 'transformed':
            # The inherited environment restorer has one 1450->1500 timeout adaptation.
            class Timeout(ast.NodeTransformer):
                def visit_Constant(self, value):
                    if type(value.value) is int and value.value == (1500 if backend == 'reference' else 480):
                        counts['timeout'] += 1
                        return ast.copy_location(ast.Constant(PROFILE['worker_seconds']), value)
                    return value
            Timeout().visit(node)
        if not isinstance(node, ast.FunctionDef) or node.name != 'worker':
            continue
        for statement in ast.walk(node):
            if isinstance(statement, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'command' for t in statement.targets):
                command = statement.value
                if not isinstance(command, ast.List):
                    raise ValueError('Generation command changed')
                expected = {'--steps': ('10', '50'), '--height': ('384', '928'), '--width': ('576', '768')}
                changed = set()
                for i, item in enumerate(command.elts[:-1]):
                    if isinstance(item, ast.Constant) and item.value in expected:
                        old, new = expected[item.value]
                        if not isinstance(command.elts[i + 1], ast.Constant) or command.elts[i + 1].value != old:
                            raise ValueError('Original generation profile changed')
                        command.elts[i + 1] = ast.Constant(new)
                        changed.add(item.value)
                if changed != set(expected):
                    raise ValueError('Incomplete generation profile override')
                counts['command'] += 1

        class WorkerChecks(ast.NodeTransformer):
            def visit_Tuple(self, value):
                if ast.dump(value, include_attributes=False) == ast.dump(ast.parse('(576,384)', mode='eval').body, include_attributes=False):
                    counts['frame_shape'] += 1
                    return ast.copy_location(ast.parse('(768,928)', mode='eval').body, value)
                return self.generic_visit(value)

            def visit_Constant(self, value):
                if type(value.value) is int and value.value == 9:
                    counts['native_calls'] += 1
                    return ast.copy_location(ast.Constant(EXPECTED_FORWARDS), value)
                return value

            def visit_For(self, value):
                # Avoid three large optional PNGs in the nearly full personal AFS.
                if (isinstance(value.target, ast.Tuple)
                        and [getattr(v, 'id', None) for v in value.target.elts] == ['label', 'index']
                        and ast.literal_eval(value.iter) == (('first', 0), ('middle', 62), ('last', 123))):
                    counts['png_preview_loop'] += 1
                    return ast.copy_location(ast.Pass(), value)
                return self.generic_visit(value)
        WorkerChecks().visit(node)

    class OutputNames(ast.NodeTransformer):
        def visit_Constant(self, value):
            mapping = ({'official-example.mp4': 'reference-768x928-s50.mp4',
                        'h3-official-example-': 'h3-reference-768x928-s50-'}
                       if backend == 'reference' else
                       {'native-competition-cache.mp4': 'native-cache-768x928-s50.mp4',
                        'h3-native-competition-cache-': 'h3-native-cache-768x928-s50-'})
            if isinstance(value.value, str) and value.value in mapping:
                return ast.copy_location(ast.Constant(mapping[value.value]), value)
            return value
    tree = OutputNames().visit(tree)
    expected = {'parameters': 1, 'command': 1, 'frame_shape': 1,
                'native_calls': 0 if backend == 'reference' else 1,
                'timeout': 1, 'png_preview_loop': 1}
    if counts != expected:
        raise ValueError('Launcher profile anchors changed: ' + repr(counts))
    return ast.fix_missing_locations(tree), counts


def install_profile_checks():
    tree, counts = profile_checks_tree()
    module = types.ModuleType('h3_competition_checks')
    module.__file__ = str(ROOT / 'h3_competition_checks.py')
    exec(compile(tree, '<isolated-profile-checks>', 'exec'), module.__dict__)
    sys.modules['h3_competition_checks'] = module
    return counts


def build(backend, entry_file):
    checked_sources()
    if backend not in ('reference', 'native'):
        raise ValueError('Unknown backend')
    install_profile_checks()
    if backend == 'reference':
        tree, counts = adapt_tree(ast.parse((ROOT / 'h3_official_example.py').read_text()), backend)
        scope = {'__name__': 'h3_profile_reference', '__file__': str(entry_file)}
        exec(compile(tree, '<reference-profile-launcher>', 'exec'), scope)
        return scope['main'], tree, counts
    main = runpy.run_path(str(ROOT / 'h3_competition_cache_20261107.py'), run_name='profile_cache')['load_candidate']()
    scope = main.__globals__
    original = scope['transformed_official']

    def transform(source):
        return adapt_tree(original(source), backend)[0]

    tree, counts = adapt_tree(original((ROOT / 'h3_official_example.py').read_text()), backend)
    scope.update(__file__=str(entry_file), transformed_official=transform)
    return main, tree, counts


def main(backend, entry_file):
    function, tree, counts = build(backend, Path(entry_file).resolve())
    if len(sys.argv) < 2:
        print(json.dumps({'mode': 'plan_only', 'backend': backend, 'profile': PROFILE,
                          'expected_transformer_calls': EXPECTED_FORWARDS,
                          'profile_anchor_checks': counts, 'new_model_math': False,
                          'new_gpu_jobs': 0, 'omitted_diagnostics': 'Three optional PNG previews only'}))
        return
    if sys.argv[1] not in ('--platform', '--worker'):
        raise ValueError('Only plan_only, --platform, --worker accepted')
    return function()
