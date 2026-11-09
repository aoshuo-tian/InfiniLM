"""Single-request competition-aligned local profile. No arguments prints a plan only."""
import ast
import hashlib
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parent
FAST_SHA = '18f2f68f52fe8b7d54b0ccb9977907719149c70acd99c8e5fca4f454d8411e66'


def load_candidate():
    fast = ROOT / 'h3_native_bridge_fast_20261107.py'
    if hashlib.sha256(fast.read_bytes()).hexdigest() != FAST_SHA:
        raise ValueError('Stable source-device bridge launcher changed')
    main = runpy.run_path(str(fast), run_name='competition_dependency')['load_candidate']()
    scope = main.__globals__
    original_transform = scope['transformed_official']

    def transform(source):
        tree = original_transform(source)
        tree.body.insert(0, ast.ImportFrom(module='h3_competition_checks', names=[
            ast.alias(name=name) for name in
            ('run_single_entry', 'validate_media', 'annotate_result', 'evaluation_status')], level=0))
        counts = dict.fromkeys(('entry', 'gate', 'save', 'worker_timeout'), 0)

        class Adapt(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                if node.name == 'save':
                    node.body.insert(0, ast.parse('annotate_result(value)').body[0])
                    counts['save'] += 1
                return self.generic_visit(node)

            def visit_Assign(self, node):
                if any(isinstance(t, ast.Name) and t.id == 'PARAMETERS' for t in node.targets):
                    parameters = ast.literal_eval(node.value)
                    if parameters['prompt'] != 'A dog falls asleep':
                        raise ValueError('Unified competition prompt differs')
                    parameters.update(platform_seconds=600, worker_seconds=480, auto_retry=False)
                    node.value = ast.parse(repr(parameters), mode='eval').body
                if any(isinstance(t, ast.Name) and t.id == 'SCOPE' for t in node.targets):
                    node.value = ast.Constant('Single native FL2VA request, A dog falls asleep; local comparison profile; official similarity and timing protocol unconfirmed')
                if (len(node.targets) == 1 and ast.unparse(node.targets[0]) == "report['passed']"
                        and isinstance(node.value, ast.Constant) and node.value.value is True):
                    counts['gate'] += 1
                    return ast.parse("report['basic_validation'] = validate_media(video)\n"
                                     "report['evaluation'] = evaluation_status(report['basic_validation']['passed'])\n"
                                     "if not report['basic_validation']['passed']:\n"
                                     "    raise ValueError('Basic media validation failed')\n"
                                     "report['passed'] = True").body
                return self.generic_visit(node)

            def visit_Call(self, node):
                if ast.unparse(node.func) == "ns['main']":
                    counts['entry'] += 1
                    return ast.copy_location(ast.Call(func=ast.Name(id='run_single_entry', ctx=ast.Load()),
                        args=[ast.Name(id=name, ctx=ast.Load()) for name in ('ns', 'report', 'output')], keywords=[]), node)
                return self.generic_visit(node)

            def visit_Constant(self, node):
                if type(node.value) is int and node.value == 1500:
                    counts['worker_timeout'] += 1
                    return ast.copy_location(ast.Constant(480), node)
                replacements = {'native-bridge-fast.mp4': 'native-competition-single.mp4',
                                'h3-native-bridge-fast-': 'h3-native-competition-single-',
                                'H3_NATIVE_BRIDGE_FAST_RESULT': 'H3_COMPETITION_SINGLE_RESULT'}
                if isinstance(node.value, str) and node.value in replacements:
                    return ast.copy_location(ast.Constant(replacements[node.value]), node)
                return node

        tree = ast.fix_missing_locations(Adapt().visit(tree))
        if any(value != 1 for value in counts.values()):
            raise ValueError('Single-request launcher anchors changed: ' + repr(counts))
        return tree

    scope.update(__file__=str(Path(__file__).resolve()), transformed_official=transform)
    return main


if __name__ == '__main__':
    load_candidate()()
