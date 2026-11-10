"""Exercise actual progress, timeout, resume and archive guards without DCU."""
import ast
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import errno

import build_checkpoint as checkpoint
from build_monitor import run_logged, write_json

HERE = Path(__file__).resolve().parent


class BuildControlChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_live_output_is_reported_before_process_finishes(self):
        records = []
        report = run_logged([sys.executable, '-u', '-c',
            'import time;print("compiling-real-object",flush=True);time.sleep(.5)'],
            self.root / 'output.log', heartbeat=.05, on_progress=records.append)
        self.assertEqual(report['exit_code'], 0)
        self.assertTrue(any(r['status'] == 'running' and r.get('log_bytes', 0) > 0 for r in records))
        self.assertIn('compiling-real-object', report['tail'])

    def test_timeout_has_terminal_record_and_dead_child(self):
        records = []
        report = run_logged([sys.executable, '-c', 'import time;time.sleep(30)'],
            self.root / 'timeout.log', timeout=.15, heartbeat=.05, new_session=True,
            on_progress=records.append)
        self.assertEqual(report['exit_code'], 124)
        self.assertTrue(report['timed_out'])
        self.assertEqual(records[-1]['status'], 'timed_out')
        self.assertIsNotNone(report['terminated_returncode'])

    @unittest.skipUnless(os.name == 'posix', 'Process-group descendant check needs Linux')
    def test_timeout_also_stops_compiler_descendant(self):
        marker = self.root / 'escaped'
        child = 'import time;from pathlib import Path;time.sleep(1);Path(' + repr(str(marker)) + ').write_text("bad")'
        parent = 'import subprocess,sys,time;subprocess.Popen([sys.executable,"-c",' + repr(child) + ']);time.sleep(30)'
        run_logged([sys.executable, '-c', parent], self.root / 'group.log', timeout=.2, new_session=True)
        import time
        time.sleep(1.2)
        self.assertFalse(marker.exists())

    def make_build(self):
        root = self.root / 'old'
        (root / 'build/sources/Core/build/.objs').mkdir(parents=True)
        (root / 'build/build-origin.json').write_text('{"initial_clean_host_framework":true}')
        (root / 'build/sources/Core/build/.objs/one.o').write_bytes(b'object-content')
        (root / 'build/setup.sh').write_text('export H3_REVIEW_WORK=' + str(root / 'build'))
        (root / 'build/venv').mkdir()
        (root / 'build/venv/large-dependency').write_bytes(b'not duplicated in the object checkpoint')
        (root / 'session-report.json').write_text(json.dumps(dict(root=str(root), source_sha256='source', stages=[])))
        return root

    def test_checkpoint_roundtrip_keeps_object_and_relocates_metadata(self):
        root = self.make_build()
        archive = self.root / 'checkpoint.tar.gz'
        report = checkpoint.pack_checkpoint(root, archive, 'source')
        restored = self.root / 'new'
        receipt = checkpoint.restore_checkpoint(archive, report['sha256'], 'source', restored)
        self.assertEqual((restored / 'build/sources/Core/build/.objs/one.o').read_bytes(), b'object-content')
        self.assertIn(str(restored), (restored / 'build/setup.sh').read_text())
        self.assertFalse(report['successful_runtime'])
        self.assertFalse((restored / 'build/venv').exists())
        self.assertTrue(receipt['incremental_target_validation_pending'])

    def test_checkpoint_wrong_sha_or_source_is_rejected(self):
        root = self.make_build()
        archive = self.root / 'checkpoint.tar.gz'
        report = checkpoint.pack_checkpoint(root, archive, 'source')
        for digest, source in [('0'*64, 'source'), (report['sha256'], 'changed')]:
            with self.assertRaises(ValueError):
                checkpoint.restore_checkpoint(archive, digest, source, self.root / 'rejected')

    def test_checkpoint_limit_keeps_previous_complete_file(self):
        root = self.make_build()
        archive = self.root / 'checkpoint.tar.gz'
        archive.write_bytes(b'previous-complete-checkpoint')
        with self.assertRaises(RuntimeError):
            checkpoint.pack_checkpoint(root, archive, 'source', maximum=32)
        self.assertEqual(archive.read_bytes(), b'previous-complete-checkpoint')

    def test_cross_filesystem_publication_verifies_copy_before_replace(self):
        source, destination = self.root / 'scratch-file', self.root / 'published'
        source.write_bytes(b'new verified data')
        destination.write_bytes(b'previous complete data')
        original = Path.replace
        def cross_device(path, target):
            if path == source:
                raise OSError(errno.EXDEV, 'fixture different filesystem')
            return original(path, target)
        with patch.object(Path, 'replace', cross_device):
            checkpoint.publish_file(source, destination)
        self.assertEqual(destination.read_bytes(), b'new verified data')
        self.assertFalse(source.exists())
        self.assertFalse(list(self.root.glob('.h3-publish-*')))

    def test_checkpoint_refuses_legacy_without_clean_origin(self):
        root = self.root / 'legacy'
        root.mkdir()
        with self.assertRaises(RuntimeError):
            checkpoint.pack_checkpoint(root, self.root / 'archive.tar.gz', 'source')

    def test_member_tampering_and_escaping_path_are_rejected(self):
        root = self.make_build()
        archive = self.root / 'checkpoint.tar.gz'
        checkpoint.pack_checkpoint(root, archive, 'source')
        with tarfile.open(archive) as tar:
            payload = [(m, tar.extractfile(m).read()) for m in tar if m.isfile()]
        broken = self.root / 'tampered.tar.gz'
        with tarfile.open(broken, 'w:gz') as tar:
            for member, data in payload:
                if member.name.endswith('one.o'):
                    data = b'tampered-data!'
                    member.size = len(data)
                tar.addfile(member, io.BytesIO(data))
        with self.assertRaises(ValueError):
            checkpoint.restore_checkpoint(broken, checkpoint.sha(broken), 'source', self.root / 'broken')
        for path in ['../escape', '/absolute', 'a/../escape', 'a\\..\\escape', 'C:/escape', 'a//b']:
            with self.assertRaises(ValueError):
                checkpoint.safe_name(path)

    def test_verified_completed_command_is_not_reexecuted(self):
        tree = ast.parse((HERE / 'build_framework.py').read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        namespace = dict(Path=Path, json=json, hashlib=hashlib, run_logged=run_logged,
                         write_json=write_json, time=__import__('time'), w=self.root,
                         status={}, args_resume=False)
        (self.root / 'logs').mkdir()
        exec(compile(ast.Module(body=functions, type_ignores=[]), '<framework-functions>', 'exec'), namespace)
        output = self.root / 'one.o'
        command = [sys.executable, '-c', 'from pathlib import Path;Path(' + repr(str(output)) + ').write_bytes(b"object")', '-o', str(output)]
        namespace['run']('one.log', command)
        before = output.stat().st_mtime_ns
        namespace['args_resume'] = True
        namespace['run']('one.log', command)
        self.assertEqual(output.stat().st_mtime_ns, before)
        output.write_bytes(b'altered')
        with self.assertRaises(AssertionError):
            namespace['run']('one.log', command)

    @unittest.skipUnless(sys.platform == 'linux' and sys.version_info[:2] == (3, 11),
                         'Driver contract needs the target Linux Python 3.11')
    def test_driver_records_timeout_and_exports_incomplete_checkpoint(self):
        prepare = '''import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--core');p.add_argument('--work');a=p.parse_args()
w=Path(a.work);w.mkdir();(w/'logs').mkdir();(w/'setup.sh').write_text('')
(w/'build-origin.json').write_text(json.dumps(dict(initial_clean_host_framework=True)))
'''
        payload = {
            'repos/InfiniLM/tools/h3_submission/prepare_build.py': prepare.encode(),
            'repos/InfiniLM/tools/h3_submission/install_dependencies.py': b'print("fixture dependency stage")',
            'repos/InfiniLM/tools/h3_submission/build_framework.py': b'import time;print("fixture compiling",flush=True);time.sleep(30)',
        }
        manifest = dict(commits={'InfiniCore':'test-fixture','InfiniLM':'test-fixture'},
                        files={name:hashlib.sha256(data).hexdigest() for name,data in payload.items()})
        archive = self.root / 'fixture-source.tar.gz'
        with tarfile.open(archive, 'w:gz') as tar:
            for name,data in [*payload.items(), ('reviewer-source-manifest.json',json.dumps(manifest).encode())]:
                info=tarfile.TarInfo(name);info.size=len(data);tar.addfile(info,io.BytesIO(data))
        output = self.root / 'fixture-runtime.tar.gz'
        saved = self.root / 'fixture-checkpoint.tar.gz'
        result = subprocess.run([sys.executable,str(HERE/'build_and_pack.py'),
            '--source-bundle',str(archive),'--source-sha256',checkpoint.sha(archive),
            '--output',str(output),'--seconds','4','--checkpoint-reserve','1',
            '--checkpoint-output',str(saved)],cwd=self.root,capture_output=True,text=True,timeout=15)
        self.assertNotEqual(result.returncode,0)
        report=json.loads(output.with_suffix(output.suffix+'.json').read_text())
        self.assertTrue(report['complete'])
        self.assertFalse(report['passed'])
        self.assertEqual(report['stages'][-1]['label'],'framework')
        self.assertEqual(report['stages'][-1]['exit_code'],124)
        self.assertTrue(report['stages'][-1]['timed_out'])
        self.assertTrue(report['checkpoint']['portable_checkpoint_created'])
        self.assertFalse(output.exists())
        self.assertIn('REVIEWER_STAGE framework 124 timed_out',result.stdout)

    def test_source_fingerprint_ignores_objects_but_detects_source_changes(self):
        tree = ast.parse((HERE / 'build_framework.py').read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        core, lm = self.root / 'sources/Core', self.root / 'sources/LM'
        for folder in [core, lm, self.root / 'dependencies', self.root / 'tooling']:
            folder.mkdir(parents=True)
        source = core / 'unit.cc'
        source.write_text('int first;')
        namespace = dict(Path=Path, hashlib=hashlib, w=self.root, c=core, lm=lm)
        exec(compile(ast.Module(body=functions, type_ignores=[]), '<framework-functions>', 'exec'), namespace)
        before = namespace['inputs']()
        (core / 'build/.objs').mkdir(parents=True)
        (core / 'build/.objs/unit.o').write_bytes(b'compiled')
        self.assertEqual(before, namespace['inputs']())
        (self.root / 'tooling/xmake-global').mkdir()
        (self.root / 'tooling/xmake-global/config.json').write_text('{"cache":true}')
        self.assertEqual(before, namespace['inputs']())
        source.write_text('int changed;')
        self.assertNotEqual(before, namespace['inputs']())


if __name__ == '__main__':
    unittest.main(verbosity=2)
