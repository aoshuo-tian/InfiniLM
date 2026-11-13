"""Host checks for build inputs, metadata-only setup and offline closure."""
import ast
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

import build_support
import offline_wheels

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


class Dist:
    def __init__(self, version, requires=()):
        self.version, self.requires = version, requires


class SourceBuildChecks(unittest.TestCase):
    def test_all_public_qk_sources_and_local_headers_exist(self):
        manifest = build_support.validate(HERE / 'support/dependencies')
        self.assertGreater(len(manifest), 30)

    def test_missing_qk_inputs_fail_before_compile(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, '12'):
                build_support.validate(temp)

    def test_core_staged_abi_paths_are_resolved(self):
        import shutil
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shutil.copytree(HERE / 'support/dependencies', root / 'dependencies')
            generated = root / 'sources/Core/build/ninetoothed-sample'
            generated.mkdir(parents=True)
            (generated / 'real-path-shape.cpp').write_text('#include "../../../../dependencies/abi/ninetoothed.h"\n')
            build_support.validate(root / 'dependencies', root / 'sources/Core')
            (root / 'dependencies/abi/ninetoothed.h').unlink()
            with self.assertRaises(ValueError):
                build_support.validate(root / 'dependencies', root / 'sources/Core')

    def test_setup_metadata_does_not_build_and_discovers_subpackages(self):
        with patch('setuptools.setup') as setup, patch('subprocess.run', side_effect=AssertionError('metadata must not build')):
            previous = Path.cwd()
            try:
                os.chdir(REPO)
                namespace = runpy.run_path(str(REPO / 'setup.py'))
            finally:
                os.chdir(previous)
        kwargs = setup.call_args.kwargs
        self.assertNotIn('egg_info', kwargs['cmdclass'])
        self.assertIn('infinilm.models', kwargs['packages'])
        expected = [p.parent.relative_to(REPO / 'python').as_posix().replace('/', '.')
                    for p in (REPO / 'python/infinilm').rglob('__init__.py')]
        self.assertEqual(set(expected), set(kwargs['packages']))
        with patch('subprocess.run') as run:
            namespace['build_cpp_module']()
        self.assertEqual([c.args[0] for c in run.call_args_list],
                         [['xmake', 'build', '_infinilm'], ['xmake', 'install', '_infinilm']])

    def test_offline_closure_preserves_extras_and_stops_at_protected_base(self):
        installed = {
            'a': Dist('1', ['b[x]>=2', 'torch>=2']),
            'b': Dist('2', ['c==3; extra=="x"', 'ignored==1; extra=="y"']),
            'c': Dist('3', ['a>=1']),
            'torch': Dist('2.9.0+das.opt1.dtk2604', ['not-downloaded==99']),
            'numpy': Dist('1.25.0')}
        result = offline_wheels.closure(['a==1', 'numpy'], installed.__getitem__)
        self.assertEqual(set(result), {'a', 'b', 'c', 'torch', 'numpy'})

    def test_offline_conflict_is_not_silently_ignored(self):
        with self.assertRaises(ValueError):
            offline_wheels.closure(['a>=2'], lambda n: Dist('1'))

    def test_later_extra_request_revisits_dependency(self):
        installed = {'a': Dist('1', ['b', 'c']), 'b': Dist('1', ['d;extra=="x"']),
                     'c': Dist('1', ['b[x]']), 'd': Dist('1')}
        self.assertIn('d', offline_wheels.closure(['a'], installed.__getitem__))

    def test_offline_plan_has_no_files_or_downloads(self):
        with tempfile.TemporaryDirectory() as temp, patch('subprocess.run', side_effect=AssertionError('plan is read-only')):
            output = Path(temp) / 'new'
            record = offline_wheels.collect(output, {'torch': '2.9.0+das.opt1.dtk2604', 'numpy': '1.25.0', 'a': '1'})
            self.assertFalse(output.exists())
            self.assertFalse(record['download'])

    def test_wheel_metadata_is_read_without_importing_its_code(self):
        import zipfile
        with tempfile.TemporaryDirectory() as temp:
            wheel = Path(temp) / 'fixture.whl'
            with zipfile.ZipFile(wheel, 'w') as archive:
                archive.writestr('fixture_dist-1.dist-info/METADATA', 'Name: fixture-dist\nVersion: 1\n')
                archive.writestr('fixture_dist/__init__.py', 'raise AssertionError("do not import")')
            self.assertEqual(offline_wheels.wheel_identity(wheel), ('fixture-dist', '1'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
