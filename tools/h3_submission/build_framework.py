from pathlib import Path
import os,json,subprocess,time,hashlib,traceback,shlex,shutil,argparse
w=Path(os.environ['H3_HEAD56_CLEAN_WORK']);s=json.loads((w/'state.json').read_text());c=Path(s['core']);lm=Path(s['lm']);lib=Path(s['lib']);status={};xmake=s['xmake']
parser=argparse.ArgumentParser();parser.add_argument('--resume-python-binding',action='store_true');args=parser.parse_args()
resume=args.resume_python_binding
if resume:
 previous=json.loads((w/'build-status.json').read_text());assert not previous['passed'] and 'Python.h: No such file' in previous['error']
 status=previous['status'];assert status['core-clean-build.log']['exit_code']==0
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,v):(w/n).write_text(json.dumps(v,indent=2))
def run(name,args,cwd=None):
 t=time.monotonic()
 with (w/'logs'/name).open('w') as f:r=subprocess.run(list(map(str,args)),cwd=cwd,stdout=f,stderr=subprocess.STDOUT)
 status[name]=dict(command=list(map(str,args)),cwd=str(cwd) if cwd else None,exit_code=r.returncode,seconds=time.monotonic()-t);save('build-progress.json',status);print('BUILD_STAGE',name,r.returncode,round(time.monotonic()-t,2),flush=True)
 if r.returncode:raise RuntimeError(name+'\n'+(w/'logs'/name).read_text()[-5000:])

def precreate_object_dirs(root,targets):
 for reltarget in targets:
  for source in list(root.rglob('*.cc'))+list(root.rglob('*.cpp'))+list(root.rglob('*.c'))+list(root.rglob('*.cu')):
   if 'build' in source.relative_to(root).parts and not source.relative_to(root).parts[0]=='build':continue
   if '.objs' in source.parts or '.xmake' in source.parts:continue
   (root/'build/.objs'/reltarget/'linux/x86_64/release'/source.relative_to(root).parent).mkdir(parents=True,exist_ok=True)

try:
 assert json.loads((w/'restoration-status.json').read_text())['passed']
 if not resume:
  assert not (c/'build/.objs').exists() and not (lm/'build').exists()
  for base in [c,lm,w/'dependencies']:assert not any(p.suffix in ['.o','.so','.a'] for p in base.rglob('*') if p.is_file()),base
 for kind in ([] if resume else ['qknorm','qkrope']):
  g=w/'dependencies'/kind/'generated';b=w/'dependencies'/kind/'build';b.mkdir();objects=[]
  for i,p in enumerate(sorted(g.glob('*.cpp'))):
   obj=b/f'kernel-{i}.o';run(f'{kind}-host-{i}.log',['g++','-c','-fPIC','-O3','-D__HIP_PLATFORM_AMD__','-I/opt/dtk/include','-I'+str(g),p,'-o',obj]);objects.append(obj)
  obj=b/'dispatch.o';run(kind+'-dispatch.log',['gcc','-c','-std=c11','-fPIC','-O3','-D__HIP_PLATFORM_AMD__','-I/opt/dtk/include','-I'+str(g),g/(kind+'.c'),'-o',obj]);objects.append(obj)
  run(kind+'-archive.log',['ar','rcs',b/('libh3-'+kind+'-aot.a'),*objects])
 flags=['--hygon-dcu=y','--hygon-arch=gfx936','--ninetoothed=y','--ninetoothed-flash-attention-only=y','--ninetoothed-cast=y','--ninetoothed-hygon-mha-large=y','--ninetoothed-hygon-mha-inner32=y','--ninetoothed-hygon-mha-head56=y','--ninetoothed-hygon-embedding=y','--ninetoothed-hygon-swiglu-fp32=y','--ninetoothed-hygon-adaln=y','--ninetoothed-hygon-projection-swiglu=y','--ninetoothed-hygon-gated-residual=y','--ninetoothed-hygon-dist=y','--cpu=y','--omp=y','--aten=n','--nv-gpu=n','--cuda=/opt/dtk/cuda/cuda','--ccache=n','--graph=n','--ccl=n','--use-vendor-ops=n']
 if not resume:
  run('core-configure.log',[xmake,'f','-y','-m','release',*flags],c)
  precreate_object_dirs(c,['infiniop-cpu','infiniop-hygon','infinirt-cpu','infinirt-hygon','infini-utils','infinirt','infiniop','infiniccl','infinicore_cpp_api','_infinicore','ninetoothed-hygon-adaln-aot','ninetoothed-hygon-projection-swiglu-aot','ninetoothed-hygon-gated-residual-aot','ninetoothed-hygon-dist-aot'])
  run('core-clean-build.log',[xmake,'build','-j','4','-v','infinicore_cpp_api'],c)
 run('core-python-build.log',[xmake,'build','-j','4','-v','_infinicore'],c)
 run('lm-configure.log',[xmake,'f','-y','-m','release','--cxx11-abi=1','--ccache=n','--ninetoothed-hygon-adaln=y','--ninetoothed-hygon-projection-swiglu=y','--ninetoothed-hygon-gated-residual=y','--ninetoothed-hygon-qknorm=y','--ninetoothed-hygon-qkrope=n'],lm)
 precreate_object_dirs(lm,['_infinilm'])
 run('lm-clean-build.log',[xmake,'build','-j','4','-v','_infinilm'],lm)
 exts=list((lm/'build').rglob('_infinilm*.so'));assert len(exts)==1;s['extension']=str(exts[0]);save('state.json',s)
 for name,bindir in [('infinicore',lib),('infinilm',exts[0].parent)]:
  p=w/'runtime/python'/name/'lib';assert not p.exists();p.symlink_to(bindir,target_is_directory=True)
 run('ordinary-python-import.log',[s['python'],'-c','import infinicore,infinilm,json;from pathlib import Path; p=sorted({l.split()[-1] for l in Path("/proc/self/maps").read_text().splitlines() if any(x in l for x in ["libinfinicore_cpp_api.so","libinfiniop.so","libinfinirt.so","_infinilm."])}); print(json.dumps(dict(core=infinicore.__file__,lm=infinilm.__file__,maps=p)))'])
 counts={name:len(list((p/'build/.objs').rglob('*.o'))) for name,p in [('Core',c),('LM',lm)]};assert counts['Core']>1000 and counts['LM']>100,counts
 text=(w/'logs/core-clean-build.log').read_text();assert 'ENABLE_NINETOOTHED_HYGON_MHA_INNER32' in text and 'h3_long_inner32_f16_m128n32' in text and 'ENABLE_NINETOOTHED_HYGON_MHA_HEAD56' in text and 'h3_flash_attention_large.c.o' in text and 'ninetoothed-dist/h3_dist_' in text
 for p,h in json.loads((w/'protected-before.json').read_text()).items():assert sha(p)==h,p
 save('build-status.json',dict(complete=True,passed=True,status=status,new_object_counts=counts,clean_host_framework=True,reused_embedded_device_code=True,libraries={str(p):sha(p) for p in [*lib.glob('*.so'),exts[0]]}))
 print('REVIEW_RELEASE_CLEAN_BUILD_PASS',counts,flush=True)
except Exception:
 save('build-status.json',dict(complete=True,passed=False,status=status,error=traceback.format_exc()));raise
