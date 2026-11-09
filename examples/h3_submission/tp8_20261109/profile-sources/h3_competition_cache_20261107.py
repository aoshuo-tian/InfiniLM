"""Single FL2VA request with a bounded, growable native conditioner KV cache."""
import ast
import hashlib
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parent
SINGLE_SHA = '9bc32e15baaa7308033b4a8559324b3b775803d70c95d959044eb88e0fe7354b'


def load_candidate():
    dependency = ROOT / 'h3_competition_single_20261107.py'
    if hashlib.sha256(dependency.read_bytes()).hexdigest() != SINGLE_SHA:
        raise ValueError('Competition single-request dependency changed')
    main = runpy.run_path(str(dependency), run_name='cache_dependency')['load_candidate']()
    scope = main.__globals__
    original = scope['transformed_official']

    def transform(source):
        tree = original(source)
        count = 0
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module == 'h3_competition_checks':
                names = [name for name in node.names if name.name != 'run_single_entry']
                count += len(node.names) - len(names)
                node.names = names
        if count != 1:
            raise ValueError('Single-request import anchor changed')
        tree.body.insert(0, ast.ImportFrom(module='h3_native_cache_budget', names=[
            ast.alias(name='run_cache_entry', asname='run_single_entry')], level=0))

        class Rename(ast.NodeTransformer):
            def visit_Constant(self, node):
                names = {'native-competition-single.mp4': 'native-competition-cache.mp4',
                         'h3-native-competition-single-': 'h3-native-competition-cache-',
                         'H3_COMPETITION_SINGLE_RESULT': 'H3_COMPETITION_CACHE_RESULT'}
                if isinstance(node.value, str) and node.value in names:
                    return ast.copy_location(ast.Constant(names[node.value]), node)
                return node
        return ast.fix_missing_locations(Rename().visit(tree))

    scope.update(__file__=str(Path(__file__).resolve()), transformed_official=transform)
    return main


if __name__ == '__main__':
    load_candidate()()
