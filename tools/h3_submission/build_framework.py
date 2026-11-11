from pathlib import Path
import os,json,subprocess,time,hashlib,traceback,shlex,shutil,argparse
from build_monitor import run_logged, write_json
from runtime_links import connect
w=Path(os.environ['H3_HEAD56_CLEAN_WORK']);s=json.loads((w/'state.json').read_text());c=Path(s['core']);lm=Path(s['lm']);lib=Path(s['lib']);status={};xmake=s['xmake']
parser=argparse.ArgumentParser();parser.add_argument('--resume-python-binding',action='store_true');parser.add_argument('--resume',action='store_true');parser.add_argument('--verify-completed',action='store_true');args=parser.parse_args()
binding_resume=args.resume_python_binding
resume=args.resume or binding_resume
if binding_resume:
 previous=json.loads((w/'build-status.json').read_text());assert not previous['passed'] and 'Python.h: No such file' in previous['error']
 status=previous['status'];assert status['core-clean-build.log']['exit_code']==0
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,v):write_json(w/n,v)

def inputs():
 result={}
 for base in [c,lm,w/'dependencies',w/'tooling']:
  for p in base.rglob('*'):
   if not p.is_file() or p.is_symlink():continue
   rel=p.relative_to(w)
   under=p.relative_to(base)
   if base in (c,lm) and under.parts[0]=='build' and (len(under.parts)<2 or not under.parts[1].startswith('ninetoothed')):continue
   if base==w/'dependencies' and 'build' in under.parts:continue
   if base==w/'tooling' and under.parts[0]=='xmake-global':continue
   if any(x in ('.xmake','__pycache__','.git') for x in rel.parts) or p.suffix in ('.o','.a','.so','.pyc','.d'):continue
   result[rel.as_posix()]=sha(p)
 return result

def outputs(name,command):
 if '-o' in command:return [Path(command[command.index('-o')+1])]
 if command[0]=='ar':return [Path(command[2])]
 if name=='core-clean-build.log':return list(lib.glob('*.so'))
 if name=='core-python-build.log':return list((c/'build').rglob('*_infinicore*.so'))
 if name=='lm-clean-build.log':return list((lm/'build').rglob('_infinilm*.so'))
 return []

def run(name,args,cwd=None):
 command=list(map(str,args));previous=status.get(name,{})
 if args_resume and previous.get('exit_code')==0 and previous.get('outputs'):
  assert previous['command']==command,('Resume command changed',name)
  for rel,digest in previous['outputs'].items():assert sha(w/rel)==digest,('Resume output changed',rel)
  print('BUILD_REUSE_VERIFIED',name,flush=True);return
 def progress(record):
  status[name]=record;save('build-progress.json',status)
 record=run_logged(command,w/'logs'/name,cwd=cwd,on_progress=progress)
 if record['exit_code']==0:
  produced=outputs(name,command)
  if name in ('core-clean-build.log','core-python-build.log','lm-clean-build.log'):assert produced,('No expected output',name)
  record['outputs']={p.relative_to(w).as_posix():sha(p) for p in produced}
 status[name]=record;save('build-progress.json',status)
 print('BUILD_STAGE',name,record['exit_code'],round(record['seconds'],2),flush=True)
 if record['exit_code']:raise RuntimeError(name+'\n'+record['tail'])

def precreate_object_dirs(root,targets):
 for reltarget in targets:
  for source in list(root.rglob('*.cc'))+list(root.rglob('*.cpp'))+list(root.rglob('*.c'))+list(root.rglob('*.cu')):
   if 'build' in source.relative_to(root).parts and not source.relative_to(root).parts[0]=='build':continue
   if '.objs' in source.parts or '.xmake' in source.parts:continue
   (root/'build/.objs'/reltarget/'linux/x86_64/release'/source.relative_to(root).parent).mkdir(parents=True,exist_ok=True)

try:
 assert json.loads((w/'restoration-status.json').read_text())['passed']
 if args.verify_completed:
  origin=json.loads((w/'build-origin.json').read_text());assert origin.get('initial_clean_host_framework') and origin['inputs']==inputs()
  completed=json.loads((w/'build-status.json').read_text());assert completed['passed'] and completed['clean_host_framework']
  ccl_started=(w/'ccl-build-status.json').exists()
  for path,digest in completed['libraries'].items():
   if ccl_started and Path(path).name in ('libinfinirt.so','libinfiniccl.so'):continue
   assert sha(path)==digest,('Completed framework output changed',path)
  print('COMPLETED_FRAMEWORK_REUSE_VERIFIED',flush=True);raise SystemExit(0)
 args_resume=args.resume
 if args_resume:
  origin=json.loads((w/'build-origin.json').read_text())
  assert origin.get('initial_clean_host_framework') is True
  assert origin['inputs']==inputs(),'Build inputs changed; resume refused'
  status=json.loads((w/'build-progress.json').read_text()) if (w/'build-progress.json').exists() else {}
 if not resume:
  assert not (c/'build/.objs').exists() and not (lm/'build').exists()
  for base in [c,lm,w/'dependencies']:assert not any(p.suffix in ['.o','.so','.a'] for p in base.rglob('*') if p.is_file()),base
  save('build-origin.json',dict(initial_clean_host_framework=True,inputs=inputs(),gpu_executed=False))
 for kind in ([] if binding_resume else ['qknorm','qkrope']):
  g=w/'dependencies'/kind/'generated';b=w/'dependencies'/kind/'build';b.mkdir(exist_ok=args_resume);objects=[]
  for i,p in enumerate(sorted(g.glob('*.cpp'))):
   obj=b/f'kernel-{i}.o';run(f'{kind}-host-{i}.log',['g++','-c','-fPIC','-O3','-D__HIP_PLATFORM_AMD__','-I/opt/dtk/include','-I'+str(g),p,'-o',obj]);objects.append(obj)
  obj=b/'dispatch.o';run(kind+'-dispatch.log',['gcc','-c','-std=c11','-fPIC','-O3','-D__HIP_PLATFORM_AMD__','-I/opt/dtk/include','-I'+str(g),g/(kind+'.c'),'-o',obj]);objects.append(obj)
  run(kind+'-archive.log',['ar','rcs',b/('libh3-'+kind+'-aot.a'),*objects])
 flags=['--hygon-dcu=y','--hygon-arch=gfx936','--ninetoothed=y','--ninetoothed-flash-attention-only=y','--ninetoothed-cast=y','--ninetoothed-hygon-mha-large=y','--ninetoothed-hygon-mha-inner32=y','--ninetoothed-hygon-mha-head56=y','--ninetoothed-hygon-embedding=y','--ninetoothed-hygon-swiglu-fp32=y','--ninetoothed-hygon-adaln=y','--ninetoothed-hygon-projection-swiglu=y','--ninetoothed-hygon-gated-residual=y','--ninetoothed-hygon-dist=y','--cpu=y','--omp=y','--aten=n','--nv-gpu=n','--cuda=/opt/dtk/cuda/cuda','--ccache=n','--graph=n','--ccl=n','--use-vendor-ops=n']
 if not binding_resume:
  run('core-configure.log',[xmake,'f','-y','-m','release',*flags],c)
  precreate_object_dirs(c,['infiniop-cpu','infiniop-hygon','infinirt-cpu','infinirt-hygon','infini-utils','infinirt','infiniop','infiniccl','infinicore_cpp_api','_infinicore','ninetoothed-hygon-adaln-aot','ninetoothed-hygon-projection-swiglu-aot','ninetoothed-hygon-gated-residual-aot','ninetoothed-hygon-dist-aot'])
  run('core-clean-build.log',[xmake,'build','-j','4','-v','infinicore_cpp_api'],c)
 run('core-python-build.log',[xmake,'build','-j','4','-v','_infinicore'],c)
 run('lm-configure.log',[xmake,'f','-y','-m','release','--cxx11-abi=1','--ccache=n','--ninetoothed-hygon-adaln=y','--ninetoothed-hygon-projection-swiglu=y','--ninetoothed-hygon-gated-residual=y','--ninetoothed-hygon-qknorm=y','--ninetoothed-hygon-qkrope=n'],lm)
 precreate_object_dirs(lm,['_infinilm'])
 run('lm-clean-build.log',[xmake,'build','-j','4','-v','_infinilm'],lm)
 exts=list((lm/'build').rglob('_infinilm*.so'));assert len(exts)==1;s['extension']=str(exts[0]);save('state.json',s)
 save('runtime-links.json',connect(w,s))
 run('native-extension-import.log',[s['python'],Path(__file__).with_name('runtime_links.py'),'--verify-native',w])
 counts={name:len(list((p/'build/.objs').rglob('*.o'))) for name,p in [('Core',c),('LM',lm)]};assert counts['Core']>1000 and counts['LM']>100,counts
 text=(w/'logs/core-clean-build.log').read_text();assert 'ENABLE_NINETOOTHED_HYGON_MHA_INNER32' in text and 'h3_long_inner32_f16_m128n32' in text and 'ENABLE_NINETOOTHED_HYGON_MHA_HEAD56' in text and 'h3_flash_attention_large.c.o' in text and 'ninetoothed-dist/h3_dist_' in text
 for p,h in json.loads((w/'protected-before.json').read_text()).items():assert sha(p)==h,p
 save('build-status.json',dict(complete=True,passed=True,status=status,new_object_counts=counts,clean_host_framework=True,resumed=args_resume,reused_embedded_device_code=True,full_package_import_validated=False,gpu_execution_validated=False,libraries={str(p):sha(p) for p in [*lib.glob('*.so'),exts[0]]}))
 print('REVIEW_RELEASE_CLEAN_BUILD_PASS',counts,flush=True)
except Exception:
 save('build-status.json',dict(complete=True,passed=False,status=status,error=traceback.format_exc()));raise
