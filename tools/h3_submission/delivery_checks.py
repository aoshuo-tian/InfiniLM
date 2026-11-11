"""Host regression tests for packaging repairs; no models or DCU execution."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_rebuilt
import runtime_links
import prepare_flash

HERE = Path(__file__).resolve().parent


class DeliveryChecks(unittest.TestCase):
    def fixture(self, root):
        lib = root / 'core/lib'
        lib.mkdir(parents=True)
        ext = root / 'lm/_infinilm.cpython-311-x86_64-linux-gnu.so'
        ext.parent.mkdir(parents=True)
        ext.write_bytes(b'fixture-extension')
        package = root / 'runtime/python/infinilm/lib'
        package.mkdir(parents=True)
        (root / 'runtime/python/infinicore').mkdir(parents=True)
        (package / '__init__.py').write_bytes(b'import _infinilm\n')
        return dict(lib=str(lib), extension=str(ext))

    def test_real_lm_package_is_retained(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            state = self.fixture(root)
            initializer = root / 'runtime/python/infinilm/lib/__init__.py'
            before = initializer.read_bytes()
            links = runtime_links.link_plan(root, state)
            self.assertEqual(links[1][0].parent, initializer.parent)
            self.assertNotEqual(links[1][0], initializer.parent)
            self.assertEqual(initializer.read_bytes(), before)

    def test_foreign_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            state = self.fixture(root)
            state['extension'] = str(root.parent / 'foreign.so')
            with self.assertRaises(ValueError):
                runtime_links.link_plan(root, state)

    def test_collision_fails_before_any_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            state = self.fixture(root)
            links = runtime_links.link_plan(root, state)
            links[1][0].write_bytes(b'do-not-overwrite')
            with patch.object(runtime_links, 'ensure_relative_link') as mutate:
                with self.assertRaises(ValueError):
                    runtime_links.connect(root, state)
                mutate.assert_not_called()
            self.assertEqual(links[1][0].read_bytes(), b'do-not-overwrite')

    @unittest.skipIf(os.name == 'nt', 'Target Linux symlinks require Windows privileges unavailable here')
    def test_relative_links_survive_relocation_and_repeat(self):
        import shutil
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve() / 'original'
            state = self.fixture(root)
            runtime_links.connect(root, state)
            runtime_links.connect(root, state)
            moved = root.parent / 'moved'
            shutil.copytree(root, moved, symlinks=True)
            self.assertEqual((moved / 'runtime/python/infinilm/lib' / Path(state['extension']).name).read_bytes(), b'fixture-extension')
            self.assertEqual((moved / 'runtime/python/infinicore/lib').resolve(), moved / 'core/lib')

    def test_reference_and_native_keep_fixed_configuration(self):
        for backend in ('native', 'reference'):
            scope, _ = run_rebuilt.make_worker(backend)
            p = scope['PARAMETERS']
            self.assertEqual((p['width'], p['height'], p['steps'], p['num_frames'], p['fps'], p['seed']), (768, 928, 50, 124, 24, 42))
            self.assertEqual(p['prompt'], 'A dog falls asleep')
            if backend == 'native':
                self.assertEqual(p['native_text_gpus'], list(range(8)))
                self.assertEqual(p['native_transformer_gpus'], list(range(8)))
            else:
                self.assertEqual(p['legacy_text_gpus'], [0, 1, 2, 3])
                self.assertEqual(p['legacy_transformer_gpus'], [4, 5, 6, 7])

    def test_both_worker_computations_unchanged(self):
        run_rebuilt.make_worker()
        import h3_profile_768x928 as base
        import h3_tp8_profile as tp8
        for backend in ('native', 'reference'):
            _, tree, _ = base.build(backend, base.ROOT / 'reviewer_worker.py')
            if backend == 'native':
                tree, _ = tp8.tp8_tree(tree)
            worker = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'worker')
            expected = copy.deepcopy(worker)
            values = {n.targets[0].id: copy.deepcopy(n.value) for n in ast.walk(worker)
                      if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                      and n.targets[0].id in ('home', 'descriptor', 'allowed', 'image')}
            guard = next(copy.deepcopy(n) for n in ast.walk(worker) if isinstance(n, ast.If)
                         and ast.unparse(n.test) == "descriptor.parent != home / 'h3-official-flash-cache'")
            adapted, _ = run_rebuilt.worker_tree(tree)

            class Restore(ast.NodeTransformer):
                def visit_Assign(self, node):
                    if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in values:
                        node.value = copy.deepcopy(values[node.targets[0].id])
                    return self.generic_visit(node)
                def visit_Expr(self, node):
                    if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'check_descriptor':
                        return copy.deepcopy(guard)
                    return self.generic_visit(node)

            actual = next(n for n in adapted.body if isinstance(n, ast.FunctionDef) and n.name == 'worker')
            self.assertEqual(ast.dump(expected, include_attributes=False),
                             ast.dump(Restore().visit(actual), include_attributes=False))

    def test_flash_pin_and_incomplete_wheel_rejection(self):
        self.assertEqual(len(prepare_flash.WHEEL_SHA256), 64)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            wheel = root / 'incomplete.whl'
            wheel.write_bytes(b'incomplete')
            with self.assertRaises(ValueError):
                prepare_flash.prepare(wheel, root / 'output')
            self.assertFalse((root / 'output').exists())

    def test_original_compare_and_generator_identities(self):
        repo = HERE.parents[1]
        self.assertEqual(hashlib.sha256((repo / 'examples/minimax_h3_compare.py').read_bytes()).hexdigest(),
            'e143c28e4b3acb885662e0aa238aa00ca17cbcc82077f465f2ce57d0f1225590')
        self.assertEqual(hashlib.sha256((repo / 'examples/minimax_h3.py').read_bytes()).hexdigest(),
            'ca54a18a73a7d760d644ca4eb2c3b20678f88dfdbcfa26a62dcd3de617b31af4')


if __name__ == '__main__':
    unittest.main(verbosity=2)
