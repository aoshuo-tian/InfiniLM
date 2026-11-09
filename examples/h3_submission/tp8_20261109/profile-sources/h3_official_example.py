"""Official guide legacy FL2VA, exact media/prompt/4+4 configuration; isolated offline launcher."""
import ast
import functools
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time

ROOT=Path(__file__).resolve().parent
TEAM_SHA='23a73d5c2dafaa35fbbbd917709aee9dfefd1d5c81e6cee9c62e83470039e386'
ENTRY_SHA='ca54a18a73a7d760d644ca4eb2c3b20678f88dfdbcfa26a62dcd3de617b31af4'
IMAGE_SHA='8dacfbb2bf2ca5593c8b33222b74e7c265380c8441a41bc09718083fe1ca3fd8'
FLASH_WHEEL_SHA='8546e46bcbfcd346757c67a0ff180ace2a49716d9724d52ee870d04b4c3b41d1'
PARAMETERS={'prompt':'A dog falls asleep','seed':42,'steps':10,'height':384,'width':576,
    'num_frames':124,'fps':24,'gpus':list(range(8)),'legacy':True,
    'legacy_text_gpus':[0,1,2,3],'legacy_transformer_gpus':[4,5,6,7],
    'platform_seconds':1800,'worker_seconds':1500,'auto_retry':False}
SCOPE='Official guide legacy 4+4 FL2VA video; exact prompt/media/sampling; vendor Torch Flash SDPA'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4194304),b''):h.update(block)
    return h.hexdigest()

def save(path,value):
    if path.name=='result.json':value.update(parameters=PARAMETERS,scope=SCOPE,visual_quality='unreviewed')
    data=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
    path.write_bytes(data)
    if path.read_bytes()!=data:raise IOError('Result readback mismatch')

def checked(path,digest):
    if sha(path)!=digest:raise ValueError('Source/artifact identity mismatch: '+str(path))

def resolve_current_flash():
    """Called in the selected GPU image, before runtime restoration or weight loading."""
    import importlib.metadata as md
    import time
    home=Path(os.environ['IFAB_AFS_HOME']);cache=home/'h3-official-flash-cache';cache.mkdir(exist_ok=True)
    record={'passed':False,'scope':'Selected GPU image metadata and files only; no framework or model import',
        'gpu_kernel_executed':False,'source':'selected GPU image pre-existing library'}
    prepared=cache/'prepare.json'
    if prepared.is_file():
        prior=json.loads(prepared.read_text())
        if prior.get('passed') and prior.get('wheel_sha256')==FLASH_WHEEL_SHA:
            checked(prior['flash_so'],prior['flash_so_sha256']);return prepared
    roots={Path('/usr/local/lib/python3.11/site-packages'),Path('/opt/dtk/lib')}
    try:
        distribution=md.distribution('flash-attn')
        roots.add(Path(distribution.locate_file('')))
        record['distribution_version']=distribution.version
    except md.PackageNotFoundError:record['distribution_version']=None
    candidates=set()
    for root in roots:
        if root.is_dir():
            for pattern in ('flash_attn_2_cuda*.so','torch/lib/flash_attn_2_cuda*.so'):
                candidates.update(p.resolve() for p in root.glob(pattern) if p.is_file())
    record['candidates']=[str(p) for p in sorted(candidates)]
    if len(candidates)==1:
        so=next(iter(candidates));record.update(passed=True,flash_so=str(so),flash_so_sha256=sha(so),flash_so_bytes=so.stat().st_size)
    else:record['error']='No unique existing vendor Flash library in selected GPU image; no download, model load or MATH fallback'
    target=cache/('gpu-discovery-'+str(time.time_ns())+'.json');save(target,record)
    print('H3_OFFICIAL_GPU_FLASH_DISCOVERY',json.dumps(record),flush=True)
    if not record['passed']:raise RuntimeError(record['error']+'; record='+str(target))
    return target

def worker(work,output,model_path):
    report={'passed':False,'stage':'flash-artifact','parameters':PARAMETERS,'scope':SCOPE,
        'visual_quality':'unreviewed','calls':{},'no_model_source_patch':True}
    started=time.monotonic()
    def persist():save(output/'generation-metrics.json',report)
    try:
        home=Path(os.environ['IFAB_AFS_HOME'])
        descriptor=Path(os.environ['H3_OFFICIAL_FLASH_DESCRIPTOR'])
        if descriptor.parent!=home/'h3-official-flash-cache':raise ValueError('Unexpected Flash descriptor')
        flash=json.loads(descriptor.read_text())
        if not flash.get('passed'):
            raise ValueError('Verified vendor Flash cache not ready; model loading prohibited')
        so=Path(flash['flash_so'])
        allowed=(home/'h3-official-flash-cache',Path('/usr/local/lib/python3.11/site-packages'),Path('/opt/dtk/lib'))
        if not any(so.resolve().is_relative_to(p.resolve()) for p in allowed) or not so.name.startswith('flash_attn_2_cuda'):
            raise ValueError('Unexpected Flash path')
        checked(so,flash['flash_so_sha256']);os.environ['FA_SO_PATH']=str(so)
        report['flash']={k:flash.get(k) for k in ('source','wheel_sha256','distribution_version','flash_so','flash_so_sha256','flash_so_bytes')}
        import torch
        import torch.nn.functional as F
        from torch.nn.attention import sdpa_kernel,SDPBackend
        if not torch.version.hip or torch.cuda.device_count()!=8:
            raise ValueError('Expected one authorized eight-DCU node')
        report['environment']={'torch':torch.__version__,'hip':torch.version.hip,
            'devices':[torch.cuda.get_device_name(i) for i in range(8)]}
        report['stage']='forced-flash-preflight';persist()
        report['flash_smoke']=[]
        # No model is loaded until every allocated device executes the guide's forced Flash test.
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            if torch.backends.cuda.math_sdp_enabled() or torch.backends.cuda.mem_efficient_sdp_enabled():
                raise ValueError('Unexpected fallback enabled')
            for device in range(8):
                for dimension in (64,128):
                    q=torch.randn(2,8,128,dimension,device=f'cuda:{device}',dtype=torch.bfloat16)
                    y=F.scaled_dot_product_attention(q,q,q)
                    if not torch.isfinite(y).all().item():raise ValueError('Nonfinite Flash smoke')
                    torch.cuda.synchronize(device)
                    report['flash_smoke'].append({'device':device,'head_dim':dimension,'passed':True})
            del q,y
        report['flash_fallback_disabled']=True
        print('H3_OFFICIAL_FLASH_PREFLIGHT_PASS',json.dumps(report['flash_smoke']),flush=True)
        state=json.loads((work/'state.json').read_text())
        entry=Path(state['lm'])/'examples/minimax_h3.py';checked(entry,ENTRY_SHA)
        image=ROOT/'user-puppy-clear-8dacfbb2.jpg';checked(image,IMAGE_SHA)
        report.update(entry=str(entry),entry_sha256=ENTRY_SHA,image_sha256=IMAGE_SHA)
        import importlib.metadata as md
        report['dependencies']={k:md.version(k) for k in ('torch','transformers','diffusers','accelerate','av','torchvision')}
        ns=runpy.run_path(str(entry),run_name='h3_official_guide_entry')
        original_load=ns['load_pipeline']
        def placement(module,allowed,label):
            counts={}
            for parameter in module.parameters():
                key=str(parameter.device);counts[key]=counts.get(key,0)+parameter.numel()
            seen={int(k.split(':')[1]) for k in counts if k.startswith('cuda:')}
            if not seen or not seen<=set(allowed) or any(not k.startswith('cuda:') for k in counts):
                raise ValueError('Unexpected '+label+' device placement: '+repr(counts))
            return {'parameter_elements_by_device':counts,'hf_device_map':getattr(module,'hf_device_map',None),
                'allowed':allowed,'actual_cuda_devices':sorted(seen)}
        @functools.wraps(original_load)
        def audited_load(*args,**kwargs):
            if kwargs.get('legacy') is not True or kwargs.get('legacy_text_devices')!=[0,1,2,3] or kwargs.get('legacy_transformer_devices')!=[4,5,6,7]:
                raise ValueError('Official 4+4 CLI grouping changed')
            pipe,encoder,transformer=original_load(*args,**kwargs)
            if not isinstance(encoder,ns['Qwen3VLForConditionalGeneration']) or not isinstance(transformer,ns['MiniMaxH3Transformer3DModel']):
                raise ValueError('Official legacy implementation changed')
            report['placement']={'text':placement(encoder,[0,1,2,3],'text'),
                'transformer':placement(transformer,[4,5,6,7],'transformer')}
            report['conditioner_attn_implementation']=getattr(encoder.config,'_attn_implementation',None)
            report['vae_weight_dtypes']={name:sorted({str(p.dtype) for p in getattr(pipe,name).parameters() if p.is_floating_point()}) for name in ('vae','audio_vae')}
            # Explicit native Flash route uses vendor Torch, not the incompatible Python flash-attn dispatch API.
            transformer.set_attention_backend('_native_flash')
            report['transformer_attention_backend']='_native_flash'
            for owner,name,label in ((transformer,'forward','transformer'),(pipe.scheduler,'step','video_scheduler'),
                    (pipe.audio_scheduler,'step','audio_scheduler'),(pipe.vae,'encode','video_encode'),
                    (pipe.vae,'decode','video_decode'),(pipe.audio_vae,'decode','audio_decode')):
                original=getattr(owner,name)
                def wrap(original=original,label=label):
                    @functools.wraps(original)
                    def invoke(*a,**kw):
                        report['calls'][label]=report['calls'].get(label,0)+1
                        if torch.backends.cuda.math_sdp_enabled() or torch.backends.cuda.mem_efficient_sdp_enabled():
                            raise ValueError('Flash-only context lost: '+label)
                        return original(*a,**kw)
                    return invoke
                setattr(owner,name,wrap())
            report['stage']='models-loaded';persist()
            print('H3_OFFICIAL_PLACEMENT',json.dumps(report['placement']),flush=True)
            return pipe,encoder,transformer
        ns['main'].__globals__['load_pipeline']=audited_load
        video=output/'official-example.mp4'
        command=[str(entry),'fl2va','--model-path',str(model_path),'--text',PARAMETERS['prompt'],
            '--first-frame',str(image),'--gpus','0,1,2,3,4,5,6,7','--legacy',
            '--legacy-text-gpus','0,1,2,3','--legacy-transformer-gpus','4,5,6,7',
            '--num-frames','124','--steps','10','--seed','42','--height','384','--width','576','--output',str(video)]
        report['command']=command;report['stage']='official-full-generation';persist();sys.argv=command
        generation_started=time.monotonic()
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):ns['main']()
        report['entry_seconds']=time.monotonic()-generation_started
        if not video.is_file():raise ValueError('No official video output')
        import av
        from PIL import Image,ImageDraw
        frames=[]
        with av.open(str(video)) as container:
            stream=container.streams.video[0]
            report['media']={'fps':float(stream.average_rate),'width':stream.width,'height':stream.height,
                'video_codec':stream.codec_context.name,'audio_streams':len(container.streams.audio)}
            pts=[]
            for frame in container.decode(stream):
                frames.append(frame.to_image().convert('RGB'));pts.append(frame.pts)
        if len(frames)!=124 or report['media']['fps']!=24 or any(im.size!=(576,384) for im in frames):
            raise ValueError('Official media contract mismatch')
        if any(a is None or b is None or b<=a for a,b in zip(pts,pts[1:])):raise ValueError('Video timestamps invalid')
        report['media'].update(frames=124,duration_seconds=124/24,bytes=video.stat().st_size,sha256=sha(video))
        audio_samples=0
        with av.open(str(video)) as container:
            if not container.streams.audio:raise ValueError('Official audio missing')
            for frame in container.decode(audio=0):audio_samples+=frame.samples
        if not audio_samples:raise ValueError('Audio decode empty')
        report['media']['decoded_audio_samples']=audio_samples
        sheet=Image.new('RGB',(768,420),'white');draw=ImageDraw.Draw(sheet)
        for index,frame_no in enumerate((0,11,22,33,44,55,66,77,88,99,110,123)):
            im=frames[frame_no].resize((192,128));x=index%4*192;y=index//4*140
            sheet.paste(im,(x,y));draw.text((x+3,y+128),str(frame_no),fill='black')
        sheet.save(output/'contact-sheet.jpg',quality=92)
        for label,index in (('first',0),('middle',62),('last',123)):frames[index].save(output/('preview-'+label+'.png'))
        report['passed']=True;report['stage']='video-saved-and-decoded'
    except Exception:
        import traceback
        report['error']=traceback.format_exc();print(report['error'],flush=True)
    report['worker_seconds']=time.monotonic()-started;persist()
    print('H3_OFFICIAL_RESULT',json.dumps(report,allow_nan=False),flush=True)
    return 0 if report['passed'] else 1

def transformed(source):
    class Adapter(ast.NodeTransformer):
        def __init__(self):self.counts=dict.fromkeys(('worker','save','parameters','prefix','timeout','image'),0)
        def visit_FunctionDef(self,node):
            if node.name in ('worker','save'):self.counts[node.name]+=1;node.name='_unused_'+node.name
            return self.generic_visit(node)
        def visit_Assign(self,node):
            if any(isinstance(t,ast.Name) and t.id=='PARAMETERS' for t in node.targets):
                self.counts['parameters']+=1;node.value=ast.parse(repr(PARAMETERS),mode='eval').body
            return self.generic_visit(node)
        def visit_Constant(self,node):
            pairs={1450:('timeout',1500),'h3-fl2va-baseline-':('prefix','h3-official-example-'),
                'user-puppy-first-frame.jpg':('image','user-puppy-clear-8dacfbb2.jpg')}
            if node.value in pairs:
                label,value=pairs[node.value];self.counts[label]+=1
                return ast.copy_location(ast.Constant(value),node)
            return node
    adapter=Adapter();tree=ast.fix_missing_locations(adapter.visit(ast.parse(source)))
    if any(v!=1 for v in adapter.counts.values()):raise ValueError('Offline environment adapter changed: '+repr(adapter.counts))
    return tree

def main():
    if len(sys.argv)<2 or sys.argv[1] not in ('--platform','--worker'):
        print(json.dumps({'mode':'plan_only','scope':SCOPE,**PARAMETERS}));return
    team=ROOT/'fl2va_puppy_math_baseline.py';checked(team,TEAM_SHA)
    checked(ROOT/'user-puppy-clear-8dacfbb2.jpg',IMAGE_SHA)
    if sys.argv[1]=='--platform':
        os.environ['H3_OFFICIAL_FLASH_DESCRIPTOR']=str(resolve_current_flash())
    scope={'__file__':str(Path(__file__).resolve()),'__name__':'__main__','worker':worker,'save':save}
    exec(compile(transformed(team.read_text()),str(team),'exec'),scope)

if __name__=='__main__':main()
