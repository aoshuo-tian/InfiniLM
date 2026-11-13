"""Control/identity tests with tiny fixtures; no GPU/model or package installs."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

HERE=Path(__file__).resolve().parent
TOOLS=HERE
sys.path.insert(0,str(TOOLS))
import dependency_probe as dep
import flash_reuse as flash
import install_dependencies as inference
import install_comparison as comparison


def ready():
    return dict(passed=True,missing_or_version_mismatch=[],import_errors=[],versions={'torchvision':'0.24.0+cpu'},module_paths={})


class FlashTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=HERE)
        self.root=Path(self.tmp.name)
        self.elf=b'\x7fELF\x02\x01'+b'\0'*12+(62).to_bytes(2,'little')+b'fixture only'
        self.library=self.root/'existing.so'
        self.library.write_bytes(self.elf)
        self.pins=patch.multiple(flash,SO_BYTES=len(self.elf),SO_SHA256=hashlib.sha256(self.elf).hexdigest())
        self.pins.start()
        self.addCleanup(self.pins.stop)
        self.addCleanup(self.tmp.cleanup)

    def test_existing_library_needs_no_wheel_copy_or_import(self):
        with patch.object(flash.zipfile,'ZipFile',side_effect=AssertionError('No wheel')):
            record=flash.prepare(None,self.root/'receipt',flash_so=self.library)
        self.assertTrue(record['reused_existing'])
        self.assertFalse(record['wheel_required'])
        self.assertFalse(record['gpu_kernel_executed'])
        self.assertEqual(list((self.root/'receipt').iterdir()),[self.root/'receipt/prepare.json'])

    def test_valid_cache_can_be_reused_with_missing_wheel(self):
        before=flash.prepare(None,self.root/'receipt',flash_so=self.library)
        after=flash.prepare(self.root/'does-not-exist.whl',self.root/'receipt')
        self.assertEqual(before,after)

    def test_wrong_identity_rejected_without_creating_output(self):
        self.library.write_bytes(self.elf[:-1]+b'X')
        with self.assertRaises(ValueError):
            flash.prepare(None,self.root/'receipt',flash_so=self.library)
        self.assertFalse((self.root/'receipt').exists())

    def test_discovery_rejects_wrong_file_and_reuses_matching_file(self):
        wrong=self.root/'wrong.so'; wrong.write_bytes(b'bad')
        with patch.object(flash,'candidates',return_value=[str(wrong),str(self.library)]):
            record=flash.prepare(None,self.root/'receipt',discover=True)
        self.assertEqual(record['flash_so'],str(self.library.resolve()))

    def test_missing_both_is_clear_error_without_install(self):
        with patch.object(flash,'candidates',return_value=[]):
            with self.assertRaisesRegex(ValueError,'没有安装或下载'):
                flash.prepare(None,self.root/'receipt',discover=True)

    def test_existing_foreign_output_is_not_overwritten(self):
        out=self.root/'receipt'; out.mkdir(); (out/'keep').write_bytes(b'keep')
        with self.assertRaises(ValueError):
            flash.prepare(None,out,flash_so=self.library)
        self.assertEqual((out/'keep').read_bytes(),b'keep')

    def test_wheel_fallback_verifies_and_extracts_single_library(self):
        wheel=self.root/'vendor.whl'
        with zipfile.ZipFile(wheel,'w') as archive:
            archive.writestr(flash.SO_FILENAME,self.elf)
        with patch.multiple(flash,WHEEL_BYTES=wheel.stat().st_size,WHEEL_SHA256=flash.sha(wheel)):
            record=flash.prepare(wheel,self.root/'receipt')
        self.assertFalse(record['reused_existing'])
        self.assertEqual(Path(record['flash_so']).read_bytes(),self.elf)

    def test_elf_architecture_must_match_even_when_test_hash_matches(self):
        other=self.elf[:18]+(183).to_bytes(2,'little')+self.elf[20:]
        self.library.write_bytes(other)
        with patch.object(flash,'SO_SHA256',hashlib.sha256(other).hexdigest()):
            with self.assertRaisesRegex(ValueError,'ELF64 x86_64'):
                flash.verify_so(self.library)

    def test_wrong_wheel_hash_rejected_before_extraction(self):
        wheel=self.root/'wrong.whl'; wheel.write_bytes(b'wrong wheel')
        with self.assertRaisesRegex(ValueError,'wheel size/SHA256 mismatch'):
            flash.prepare(wheel,self.root/'receipt')
        self.assertFalse((self.root/'receipt').exists())


class DependencyTests(unittest.TestCase):
    def fixture(self):
        temp=tempfile.TemporaryDirectory(dir=HERE); self.addCleanup(temp.cleanup)
        work=Path(temp.name)
        (work/'venv/bin').mkdir(parents=True); (work/'venv/bin/python').write_bytes(b'fixture not executable')
        (work/'logs').mkdir()
        base=dict(prefix=str(work/'venv'),torch='2.9.0+das.opt1.dtk2604',numpy='1.25.0',torch_path='/system',numpy_path='/system')
        (work/'dependency-base-identities.json').write_text(json.dumps(base))
        return work,base

    def test_ready_inference_never_calls_pip_or_needs_wheelhouse(self):
        work,base=self.fixture()
        with patch.object(sys,'argv',['install','--work',str(work)]),patch.object(sys,'platform','linux'), \
             patch.object(inference.subprocess,'check_output',return_value=json.dumps(base)), \
             patch.object(inference,'dependency_probe',return_value=ready()), \
             patch.object(inference.subprocess,'run',side_effect=AssertionError('Must reuse')):
            inference.main()
        record=json.loads((work/'dependency-install.json').read_text())
        self.assertEqual(record['mode'],'reused_existing')
        self.assertEqual((work/'reviewer-base-constraints.txt').read_text(),
                         'torch=='+base['torch']+'\nnumpy=='+base['numpy']+'\n')

    def test_ready_comparison_never_calls_pip(self):
        work,base=self.fixture()
        with patch.object(sys,'argv',['comparison','--work',str(work)]),patch.object(sys,'platform','linux'), \
             patch.object(comparison.subprocess,'check_output',return_value=json.dumps(base)), \
             patch.object(comparison,'dependency_probe',return_value=ready()), \
             patch.object(comparison.subprocess,'run',side_effect=AssertionError('Must reuse')):
            comparison.main()
        self.assertEqual(json.loads((work/'comparison-dependency-install.json').read_text())['mode'],'reused_existing')

    def test_missing_without_offline_inputs_never_tries_network(self):
        work,base=self.fixture(); report=ready(); report.update(passed=False,missing_or_version_mismatch=['janus==2.0.0'])
        with patch.object(sys,'argv',['install','--work',str(work)]),patch.object(sys,'platform','linux'), \
             patch.object(inference.subprocess,'check_output',return_value=json.dumps(base)), \
             patch.object(inference,'dependency_probe',return_value=report), \
             patch.object(inference.subprocess,'run',side_effect=AssertionError('No network')):
            with self.assertRaises(SystemExit):
                inference.main()
        self.assertFalse((work/'dependency-install.json').exists())

    def test_api_failure_does_not_trigger_blind_reinstallation(self):
        work,base=self.fixture(); report=ready(); report.update(passed=False,import_errors=[dict(error='bad lazy API')])
        with patch.object(sys,'argv',['install','--work',str(work),'--allow-online']),patch.object(sys,'platform','linux'), \
             patch.object(inference.subprocess,'check_output',return_value=json.dumps(base)), \
             patch.object(inference,'dependency_probe',return_value=report), \
             patch.object(inference.subprocess,'run',side_effect=AssertionError('No install')):
            with self.assertRaises(SystemExit):
                inference.main()

    def test_offline_missing_dependency_reuses_existing_torchvision(self):
        work,base=self.fixture(); wheelhouse=work/'wheels'; wheelhouse.mkdir()
        missing=ready(); missing.update(passed=False,missing_or_version_mismatch=['janus==2.0.0'])
        with patch.object(sys,'argv',['install','--work',str(work),'--wheelhouse',str(wheelhouse)]),patch.object(sys,'platform','linux'), \
             patch.object(inference.subprocess,'check_output',return_value=json.dumps(base)), \
             patch.object(inference,'dependency_probe',side_effect=[missing,ready()]), \
             patch.object(inference.subprocess,'run') as run:
            inference.main()
        install=run.call_args_list[0].args[0]
        self.assertIn('--no-index',install)
        self.assertIn('--find-links',install)
        self.assertFalse(any('download' in call.args[0] for call in run.call_args_list))
        self.assertFalse(any('torchvision' in str(call.args[0]) for call in run.call_args_list))

    def test_base_torch_is_guarded_before_dependency_reuse(self):
        work,base=self.fixture(); base['torch']='unexpected'
        with patch.object(sys,'argv',['install','--work',str(work)]),patch.object(sys,'platform','linux'), \
             patch.object(inference.subprocess,'check_output',return_value=json.dumps(base)), \
             patch.object(inference,'dependency_probe',side_effect=AssertionError('Base first')):
            with self.assertRaises(SystemExit):
                inference.main()

    def test_check_only_does_not_accept_old_receipt_when_api_is_broken(self):
        work,base=self.fixture(); (work/'dependency-install.json').write_text('{}')
        failed=ready(); failed.update(passed=False,import_errors=[dict(error='broken')])
        with patch.object(sys,'argv',['install','--work',str(work),'--check-only']),patch.object(sys,'platform','linux'), \
             patch.object(inference.subprocess,'check_output',return_value=json.dumps(base)), \
             patch.object(inference,'dependency_probe',return_value=failed), \
             patch.object(inference.subprocess,'run',side_effect=AssertionError('No install')):
            with self.assertRaises(SystemExit):
                inference.main()

    def test_real_metadata_probe_for_installed_packaging(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            requirements=Path(tmp)/'req.txt'; requirements.write_text('packaging>=20\n')
            record=dep.inspect([requirements])
            self.assertTrue(record['passed'])
            self.assertIn('packaging',record['versions'])

    def test_missing_distribution_reports_name_instead_of_installing(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            requirements=Path(tmp)/'req.txt'; requirements.write_text('h3-nonexistent-fixture-package==1\n')
            record=dep.inspect([requirements])
            self.assertEqual(record['missing_or_version_mismatch'],['h3-nonexistent-fixture-package==1'])
            self.assertFalse(record['passed'])

    def test_pip_target_and_user_config_cannot_redirect_offline_install(self):
        with patch.dict(os.environ,{'PIP_TARGET':'/outside','PIP_USER':'1','PIP_EXTRA_INDEX_URL':'https://unused.invalid'}):
            env=inference.safe_pip_environment(Path('/private'))
        self.assertNotIn('PIP_TARGET',env)
        self.assertNotIn('PIP_USER',env)
        self.assertNotIn('PIP_EXTRA_INDEX_URL',env)
        self.assertEqual(env['PIP_CONFIG_FILE'],os.devnull)


if __name__=='__main__':
    unittest.main(verbosity=2)
