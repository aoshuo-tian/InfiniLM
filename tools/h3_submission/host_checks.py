"""Host contract checks; does not establish DCU build or video correctness."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import run_rebuilt as candidate

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CORE = REPO.parent / 'InfiniCore'
results = []


def check(name, condition):
    if not condition:
        raise AssertionError(name)
    results.append(name)


namespace, counts = candidate.make_worker()
params = namespace['PARAMETERS']
check('shared_tp8_profile', params['native_text_gpus'] == list(range(8))
      and params['native_transformer_gpus'] == list(range(8))
      and params['prompt'] == 'A dog falls asleep' and params['seed'] == 42
      and params['width'] == 768 and params['height'] == 928
      and params['steps'] == 50 and params['num_frames'] == 124 and params['fps'] == 24)
check('old_platform_restorer_not_exported', 'main' not in namespace
      and 'resolve_current_flash' not in namespace)
check('host_plan_does_not_import_torch', 'torch' not in sys.modules)

# Restore only the five permitted path adapters and compare the whole worker AST.
# This detects any unintended change in generation, VAE, casts or scheduler code.
import h3_profile_768x928 as base
import h3_tp8_profile as tp8
sources = REPO / 'examples/h3_submission/tp8_20261109/profile-sources'
function, tree, _ = base.build('native', sources / 'reviewer_worker.py')
tree, _ = tp8.tp8_tree(tree)
original = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'worker')
assignments = {x.targets[0].id: copy.deepcopy(x.value) for x in ast.walk(original)
               if isinstance(x, ast.Assign) and len(x.targets) == 1
               and isinstance(x.targets[0], ast.Name)
               and x.targets[0].id in ('home', 'descriptor', 'allowed', 'image')}
guard = next(copy.deepcopy(x) for x in ast.walk(original) if isinstance(x, ast.If)
             and ast.unparse(x.test) == "descriptor.parent != home / 'h3-official-flash-cache'")
adapted, _ = candidate.worker_tree(copy.deepcopy(tree))


class Restore(ast.NodeTransformer):
    def visit_Assign(self, node):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in assignments:
                node.value = copy.deepcopy(assignments[node.targets[0].id])
        return self.generic_visit(node)

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
            if node.value.func.id == 'check_descriptor':
                return copy.deepcopy(guard)
        return self.generic_visit(node)


new_worker = next(x for x in adapted.body if isinstance(x, ast.FunctionDef) and x.name == 'worker')
restored = Restore().visit(copy.deepcopy(new_worker))
check('worker_computation_unchanged', ast.dump(original, include_attributes=False)
      == ast.dump(restored, include_attributes=False))
check('worker_has_no_afs_environment_lookup', 'IFAB_AFS_HOME' not in ast.unparse(new_worker))

sdk = HERE / 'support/tooling/xmake'
manifest = json.loads((sdk.parent / 'xmake-sdk-manifest.json').read_text())
check('xmake_sdk_identity', all(hashlib.sha256((sdk / rel).read_bytes()).hexdigest() == digest
      for rel, digest in manifest['files'].items()))

with tempfile.TemporaryDirectory() as temporary:
    work = Path(temporary)
    provenance = json.loads((REPO / 'h3_submission/ccl-build-provenance.json').read_text())
    flags = ['--ccl=n' if x == '--ccl=y' else x for x in provenance['flags']]
    (work / 'state.json').write_text(json.dumps(dict(core=str(CORE), xmake='xmake')))
    status = dict(status={'core-configure.log': {'command': ['xmake', 'f', '-y', '-m', 'release', *flags]}})
    (work / 'build-status.json').write_text(json.dumps(status))
    env = dict(os.environ, H3_HEAD56_CLEAN_WORK=str(work))
    output = subprocess.check_output([sys.executable, str(HERE / 'build_ccl.py'), '--plan'], env=env, text=True)
    plan = json.loads(output)
    check('ccl_plan_matches_historical_commands', plan['commands'][0][5:] == provenance['flags']
          and plan['commands'][1] == ['xmake', 'build', '-j', '4', '-v', 'infiniccl']
          and plan['historical_command_recovered'] is True)
    status['status']['core-configure.log']['command'].remove('--hygon-arch=gfx936')
    (work / 'build-status.json').write_text(json.dumps(status))
    bad = subprocess.run([sys.executable, str(HERE / 'build_ccl.py'), '--plan'], env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    check('altered_ccl_build_profile_rejected', bad.returncode != 0)

bad = subprocess.run([sys.executable, str(HERE / 'prepare_build.py'), '--core', str(CORE),
                      '--work', str(CORE / 'nested-build'), '--plan'],
                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
check('recursive_staging_rejected', bad.returncode != 0 and not (CORE / 'nested-build').exists())
print(json.dumps(dict(passed=True, checks=results, gpu_executed=False), indent=2))
