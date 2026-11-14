"""租期编排的主机夹具检查，不执行编译、模型、设备或联网请求。"""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
import fabric_review_session as app


class SessionChecks(unittest.TestCase):
    def setUp(self):
        parent = Path(__file__).resolve().parents[3]/'fabric-session-fixtures'
        parent.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=parent)
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def fixture(self, outcome='pass'):
        base = self.root/'base'
        (base/'sessions').mkdir(parents=True)
        config = dict(base=str(base),source_sha256='f'*64,afs_group=base.stat().st_gid,
                      assets=dict(H3_MODEL='fixture-model',H3_WHEELHOUSE='fixture-wheels',H3_FLASH_SO='fixture-flash'))
        args = types.SimpleNamespace(command='cpu',lease_end='2030-01-01T12:00:00+08:00',reserve_seconds=180,
                                     checkpoint_mib=512,scratch_parent=self.root)
        calls = []
        def runner(command,log,**kwargs):
            name = log.stem
            calls.append((name,command))
            log.write_text('fixture\n')
            root = Path(os.environ['H3_REPRO_ROOT'])
            work = root/'build'
            if name == 'prepare':
                work.mkdir(parents=True)
            if name == 'framework':
                app.write_json(work/'build-origin.json',dict(initial_clean_host_framework=True))
                if outcome == 'pause':
                    return dict(exit_code=124,timed_out=True,seconds=1)
                libs = {}
                for i in range(6):
                    path = work/('fixture'+str(i)+'.so')
                    path.write_bytes(bytes([i]))
                    libs[str(path)] = app.sha(path)
                app.write_json(work/'build-status.json',dict(passed=True,clean_host_framework=True,libraries=libs))
            if name == 'ccl':
                app.write_json(work/'ccl-build-status.json',dict(passed=True,complete=True,libraries={}))
            if name == 'dependencies' and outcome == 'fail':
                log.write_text('fatal error: fixture initial failure\n')
                return dict(exit_code=1,timed_out=False,seconds=1)
            return dict(exit_code=0,timed_out=False,seconds=1)
        def pack(root,output,source_sha256,**kwargs):
            output.write_bytes(b'checkpoint fixture')
            return dict(path=str(output),sha256=app.sha(output),bytes=output.stat().st_size)
        return base,args,config,runner,pack,calls

    def execute_fixture(self,fixture,save_failure=False):
        base,args,config,runner,pack,calls = fixture
        with patch.object(app,'configuration',return_value=config),patch.object(app,'run_logged',side_effect=runner),patch.object(app,'pack_checkpoint',side_effect=OSError('fixture save failure') if save_failure else pack),patch.dict(os.environ):
            return app.execute(args,base)

    def test_ready_requires_saved_checkpoint_and_six_libraries(self):
        fixture = self.fixture()
        self.assertEqual(self.execute_fixture(fixture),0)
        latest = app.load(fixture[0]/'sessions/latest-cpu.json')
        self.assertEqual(latest['state'],'ready')
        report = app.load(Path(latest['run'])/'execution-record.json')
        self.assertEqual(len(report['libraries']),6)
        self.assertTrue(Path(latest['checkpoint']['path']).is_file())

    def test_budget_timeout_saves_and_returns_75(self):
        fixture = self.fixture('pause')
        self.assertEqual(self.execute_fixture(fixture),75)
        self.assertEqual(app.load(fixture[0]/'sessions/latest-cpu.json')['state'],'paused')
        self.assertNotIn('ccl',[name for name,_ in fixture[5]])

    def test_real_error_stops_before_compiling(self):
        fixture = self.fixture('fail')
        with self.assertRaisesRegex(RuntimeError,'dependencies'):
            self.execute_fixture(fixture)
        latest = app.load(fixture[0]/'sessions/latest-cpu.json')
        self.assertEqual(latest['state'],'failed')
        self.assertNotIn('framework',[name for name,_ in fixture[5]])
        self.assertIn('fixture initial failure',app.load(Path(latest['run'])/'execution-record.json')['first_error']['text'])

    def test_save_failure_cannot_publish_ready(self):
        fixture = self.fixture()
        with self.assertRaisesRegex(OSError,'save failure'):
            self.execute_fixture(fixture,save_failure=True)
        self.assertEqual(app.load(fixture[0]/'sessions/latest-cpu.json')['state'],'failed')

    def test_failed_attempt_not_automatically_resumed(self):
        fixture = self.fixture()
        app.write_json(fixture[0]/'sessions/latest-cpu.json',dict(state='failed'))
        with self.assertRaisesRegex(RuntimeError,'失败不自动重试'):
            self.execute_fixture(fixture)
        self.assertEqual(fixture[5],[])

    def test_timezone_and_budget_required(self):
        fixture = self.fixture()
        fixture[1].lease_end = '2030-01-01T12:00:00'
        with self.assertRaisesRegex(RuntimeError,'时区'):
            self.execute_fixture(fixture)
        fixture[1].lease_end = '2000-01-01T12:00:00+08:00'
        with self.assertRaisesRegex(RuntimeError,'租期'):
            self.execute_fixture(fixture)

    def test_framework_uses_public_python_without_nested_timeout(self):
        fixture = self.fixture()
        self.execute_fixture(fixture)
        commands = dict(fixture[5])
        self.assertTrue(any(x.endswith('build_framework.py') for x in commands['framework']))
        self.assertTrue(any(x.endswith('build_ccl.py') for x in commands['ccl']))
        self.assertNotIn('timeout',commands['framework'])
        self.assertIn('exec python',commands['framework'][2])
        self.assertTrue(any('sources/InfiniLM/tools/h3_submission/build_framework.py' in str(x).replace('\\','/') for x in commands['framework']))

    def test_final_evidence_failure_marks_attempt_failed(self):
        fixture = self.fixture()
        calls = []
        original = app.evidence
        def evidence(work,run):
            calls.append(1)
            if len(calls) == 2:
                raise OSError('fixture final evidence failure')
            return original(work,run)
        with patch.object(app,'evidence',side_effect=evidence):
            with self.assertRaisesRegex(OSError,'final evidence'):
                self.execute_fixture(fixture)
        self.assertEqual(app.load(fixture[0]/'sessions/latest-cpu.json')['state'],'failed')


if __name__ == '__main__':
    unittest.main(verbosity=2)
