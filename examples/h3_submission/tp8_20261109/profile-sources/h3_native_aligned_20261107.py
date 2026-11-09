"""Prepare native 4+4 TP regression against the verified official legacy example.
Default is plan-only. No candidate kernels, VAE patch, or MATH fallback.
"""
import ast
import copy
import difflib
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
OFFICIAL_SHA = '79e062c4e56a8672f64079f1d184d5916c6ce918c14505c96b8bcad34b8a5771'
ENTRY_SHA = 'ca54a18a73a7d760d644ca4eb2c3b20678f88dfdbcfa26a62dcd3de617b31af4'
ADAPTER_SHA = '5e5fc07bf0c02925c553603285fd163c44c973eb6c3de71349cf6c7677e3e4e5'
PARAMETERS = {'prompt': 'A dog falls asleep', 'seed': 42, 'steps': 10,
    'height': 384, 'width': 576, 'num_frames': 124, 'fps': 24,
    'gpus': list(range(8)), 'legacy': False, 'native_text_gpus': [0,1,2,3],
    'native_transformer_gpus': [4,5,6,7], 'platform_seconds': 1800,
    'worker_seconds': 1500, 'auto_retry': False}
SCOPE = 'Native InfiniLM 4+4 TP, official media/sampling, BF16 VAE, vendor Flash for Torch components; quality regression, not speed benchmark'

def digest(data):
    return hashlib.sha256(data).hexdigest()

def patch_native_entry(data):
    if digest(data) != ENTRY_SHA:
        raise ValueError('Frozen original entry SHA mismatch')
    text = data.decode()
    tree = ast.parse(text)
    load = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'load_pipeline')
    targets = {'MiniMaxH3ConditionEncoder': [0,1,2,3], 'MiniMaxH3TransformerRunner': [4,5,6,7]}
    changes = []
    for node in ast.walk(load):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in targets:
            kw = next(k for k in node.keywords if k.arg == 'tp_device_ids')
            if not isinstance(kw.value, ast.Name) or kw.value.id != 'devices':
                raise ValueError('Native constructor anchor changed')
            changes.append((kw.value, node.func.id, targets[node.func.id]))
    if sorted(name for _,name,_ in changes) != sorted(targets):
        raise ValueError('Expected exactly one constructor of each native component')
    lines = text.splitlines(keepends=True)
    for node, name, group in sorted(changes, key=lambda item:(item[0].lineno,item[0].col_offset), reverse=True):
        if node.lineno != node.end_lineno:
            raise ValueError('Unexpected multiline device argument')
        line = lines[node.lineno-1]
        if line[node.col_offset:node.end_col_offset] != 'devices':
            raise ValueError('Source coordinate mismatch')
        lines[node.lineno-1] = line[:node.col_offset] + repr(group) + line[node.end_col_offset:]
    modified = ''.join(lines)
    # Revert only the two expressions and demand the complete AST match.
    reverted = ast.parse(modified)
    count = 0
    for node in ast.walk(next(n for n in reverted.body if isinstance(n,ast.FunctionDef) and n.name=='load_pipeline')):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in targets:
            kw = next(k for k in node.keywords if k.arg=='tp_device_ids')
            if ast.literal_eval(kw.value) != targets[node.func.id]:
                raise ValueError('Device list mismatch')
            kw.value = ast.Name(id='devices',ctx=ast.Load());count += 1
    if count != 2 or ast.dump(reverted,include_attributes=False) != ast.dump(tree,include_attributes=False):
        raise ValueError('Changes escaped the two device expressions')
    compile(modified, 'native_split_entry.py', 'exec')
    return modified, {'original_sha256':ENTRY_SHA,'patched_sha256':digest(modified.encode()),
        'changes':[{'constructor':name,'line':node.lineno,'tp_device_ids':group} for node,name,group in changes],
        'only_two_device_expressions_changed':True, 'device_validated':False}

def load_native_entry(entry, output):
    text, record = patch_native_entry(entry.read_bytes())
    patched = entry.with_name('minimax_h3_native_split_20261107.py')
    patched.write_bytes(text.encode())
    (output/'native-entry-patch.json').write_text(json.dumps(record,indent=2)+'\n')
    (output/'native-entry.patch').write_text(''.join(difflib.unified_diff(
        entry.read_text().splitlines(keepends=True),text.splitlines(keepends=True),
        fromfile='frozen_entry.py',tofile=patched.name)))
    import runpy
    return runpy.run_path(str(patched),run_name='h3_native_aligned_entry')

def native_placement(module, allowed, label):
    engine = module.engine if label=='text' else module.runner.engine
    actual = list(engine.distributed_config.tp_device_ids)
    if actual != allowed:
        raise ValueError('Actual native TP configuration differs: '+repr((label,actual,allowed)))
    return {'tp_device_ids':actual,'engine_dtype':str(engine.dtype),
        'model_type':engine.model_type,'evidence':'Read back from live engine DistConfig; not a per-parameter placement enumeration'}

def transformed_official(source):
    replacements = [
        ("'no_model_source_patch':True", "'no_model_source_patch':False,'isolated_patch':'two native TP device expressions only','experimental_mha_rope_enabled':False"),
        ("ns=runpy.run_path(str(entry),run_name='h3_official_guide_entry')", "ns=load_native_entry(entry,output)"),
        ("if kwargs.get('legacy') is not True or kwargs.get('legacy_text_devices')!=[0,1,2,3] or kwargs.get('legacy_transformer_devices')!=[4,5,6,7]:", "if kwargs.get('legacy') is not False or list(args[2])!=list(range(8)) or kwargs.get('legacy_text_devices') or kwargs.get('legacy_transformer_devices'):"),
        ("raise ValueError('Official 4+4 CLI grouping changed')", "raise ValueError('Native all-device CLI or backend changed')"),
        ("if not isinstance(encoder,ns['Qwen3VLForConditionalGeneration']) or not isinstance(transformer,ns['MiniMaxH3Transformer3DModel']):", "if type(encoder).__name__!='MiniMaxH3ConditionEncoder' or type(transformer).__name__!='MiniMaxH3DiffusersAdapter':"),
        ("raise ValueError('Official legacy implementation changed')", "raise ValueError('Native implementation changed')"),
        ("'text':placement(encoder,[0,1,2,3],'text')", "'text':native_placement(encoder,[0,1,2,3],'text')"),
        ("'transformer':placement(transformer,[4,5,6,7],'transformer')", "'transformer':native_placement(transformer,[4,5,6,7],'transformer')"),
        ("report['conditioner_attn_implementation']=getattr(encoder.config,'_attn_implementation',None)", "report['conditioner_attn_implementation']='native_infiniLM'; audit_pipeline(pipe,encoder,transformer,report)"),
        ("transformer.set_attention_backend('_native_flash')", "# Native attention stays in the pinned runtime; no Torch backend setter applies here."),
        ("report['transformer_attention_backend']='_native_flash'", "report['transformer_attention_backend']='native_frozen_runtime'; report['torch_components_attention_backend']='forced_vendor_flash'"),
        ("'--first-frame',str(image),'--gpus','0,1,2,3,4,5,6,7','--legacy',\n            '--legacy-text-gpus','0,1,2,3','--legacy-transformer-gpus','4,5,6,7',", "'--first-frame',str(image),'--gpus','0,1,2,3,4,5,6,7',"),
        ("video=output/'official-example.mp4'", "video=output/'native-aligned.mp4'"),
        ("'h3-official-example-'", "'h3-native-aligned-'"),
        ("with sdpa_kernel(SDPBackend.FLASH_ATTENTION):ns['main']()", "with sdpa_kernel(SDPBackend.FLASH_ATTENTION):ns['main']()\n        if any(report['calls'].get(k)!=9 for k in ('transformer','video_scheduler','audio_scheduler')):raise ValueError('Unexpected denoising call counts')"),
        ("H3_OFFICIAL_RESULT", "H3_NATIVE_ALIGNED_RESULT"),
    ]
    for before,after in replacements:
        if source.count(before)!=1:
            raise ValueError('Official launcher anchor changed: '+before)
        source = source.replace(before,after)
    tree=ast.parse(source)
    for node in tree.body:
        if isinstance(node,ast.Assign):
            if any(isinstance(t,ast.Name) and t.id=='PARAMETERS' for t in node.targets):
                node.value=ast.parse(repr(PARAMETERS),mode='eval').body
            if any(isinstance(t,ast.Name) and t.id=='SCOPE' for t in node.targets):
                node.value=ast.Constant(SCOPE)
    return ast.fix_missing_locations(tree)

def audit_pipeline(pipe,encoder,transformer,report):
    import functools
    import inspect
    import torch
    module = sys.modules[type(encoder).__module__]
    if digest(Path(module.__file__).read_bytes()) != ADAPTER_SHA:
        raise ValueError('Native adapter source changed')
    report['native_adapter_sha256']=ADAPTER_SHA
    report['vae_execution_device']=str(pipe._execution_device)
    dtypes={name:sorted({str(p.dtype) for p in getattr(pipe,name).parameters() if p.is_floating_point()}) for name in ('vae','audio_vae')}
    if any(v!=['torch.bfloat16'] for v in dtypes.values()):
        raise ValueError('BF16 reference VAE contract changed: '+repr(dtypes))
    report['scheduler_initial_state']={}
    for label, scheduler in (('video',pipe.scheduler),('audio',pipe.audio_scheduler)):
        report['scheduler_initial_state'][label]={'class':type(scheduler).__name__,'config':dict(scheduler.config)}
        original=scheduler.step
        @functools.wraps(original)
        def observe_step(*args,_original=original,_label=label,**kwargs):
            bound=inspect.signature(_original).bind_partial(*args,**kwargs).arguments
            t=bound.get('timestep')
            value=t.detach().cpu().tolist() if torch.is_tensor(t) else t
            report.setdefault('scheduler_step_timesteps',{}).setdefault(_label,[]).append(value)
            return _original(*args,**kwargs)
        scheduler.step=observe_step
    original=transformer.forward
    @functools.wraps(original)
    def observe_transformer(*args,**kwargs):
        bound=inspect.signature(original).bind_partial(*args,**kwargs).arguments
        if 'first_transformer_inputs' not in report:
            record={}
            for name,value in bound.items():
                if torch.is_tensor(value):
                    cpu=value.detach().contiguous().cpu()
                    record[name]={'shape':list(cpu.shape),'dtype':str(cpu.dtype),'device':str(value.device),
                        'sha256_raw':digest(cpu.reshape(-1).view(torch.uint8).numpy().tobytes())}
                    if cpu.numel()<=256:record[name]['values']=cpu.tolist()
                    if cpu.is_floating_point() and not torch.isfinite(cpu).all().item():raise ValueError('Nonfinite first transformer input: '+name)
            report['first_transformer_inputs']=record
        return original(*args,**kwargs)
    transformer.forward=observe_transformer

def main():
    official=ROOT/'h3_official_example.py'
    source=official.read_bytes()
    if digest(source)!=OFFICIAL_SHA:raise ValueError('Verified official launcher SHA mismatch')
    tree=transformed_official(source.decode())
    scope={'__file__':str(Path(__file__).resolve()),'__name__':'h3_native_aligned_launcher',
        'load_native_entry':load_native_entry,'native_placement':native_placement,'audit_pipeline':audit_pipeline}
    exec(compile(tree,str(official),'exec'),scope)
    if len(sys.argv)>1 and sys.argv[1] not in ('--platform','--worker'):
        raise ValueError('Only plan-only (no arguments), --platform or --worker supported')
    scope['main']()

if __name__=='__main__':main()
