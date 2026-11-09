"""All new device math is expressed and generated through NineToothed."""
from pathlib import Path
import os,json,time,hashlib,importlib.util
import ninetoothed as nt
import ninetoothed.language as ntl
from ninetoothed import Tensor
from triton.runtime import driver
w=Path(os.environ['QK_NORM_WORK'])
out=w/'generated'
def reduction(prefix,rows):
    s=f'    {prefix}f=ntl.cast(ntl.reshape({prefix},({rows},128)),ntl.float32)\n    {prefix}ss={prefix}f*{prefix}f\n'
    for n in [64,32,16,8,4,2,1]:s+=f'    {prefix}l,{prefix}r=ntl.split(ntl.reshape({prefix}ss,({rows},{n},2)))\n    {prefix}ss={prefix}l+{prefix}r\n'
    s+=f'    {prefix}inv=ntl.expand_dims(ntl.rsqrt(ntl.reshape({prefix}ss,({rows},))*0.0078125+ntl.cast(e{prefix},ntl.float32)),1)\n'
    s+=f'    {prefix}weighted={prefix}f*ntl.cast(ntl.reshape(g{prefix},({rows},128)),ntl.float32)\n    {prefix}rounded={prefix}weighted*{prefix}inv\n'
    return s
def application(rows,dc):
    name=f'apply_qknorm_{rows}_{dc}';s=f'def {name}(oq,ok,q,k,gq,gk,eq,ek):\n'
    for p in ['q','k']:
        s+=reduction(p,rows)
        if dc==0:
            s+=f'''    {p}res=ntl.fma({p}weighted,{p}inv,-{p}rounded)
    {p}bits=ntl.cast({p}rounded,ntl.int32,bitcast=True)
    {p}abs={p}bits&2147483647
    {p}exp=({p}abs>>23)&255
    {p}discard=ntl.maximum(13,ntl.minimum(24,126-{p}exp))
    {p}sig=({p}abs&8388607)|8388608
    {p}mid=({p}sig&((1<<{p}discard)-1))==(1<<({p}discard-1))
    {p}direction=ntl.where(({p}bits<0)==({p}res<0),1,-1)
    {p}correct=ntl.cast({p}bits+{p}direction,ntl.float32,bitcast=True)
    {p}rounded=ntl.where({p}mid&({p}res!=0)&({p}exp>=102)&({p}exp<=142),{p}correct,{p}rounded)
'''
        s+=f'    o{p}=ntl.reshape({p}rounded,(1,{rows},128))\n'
    return name,s
def main():
    target=driver.active.get_current_target();assert (target.backend,target.arch)==('hip','gfx936'),target
    records=[];expressions='import ninetoothed.language as ntl\n'
    for rows in [4,8,16]:
        expressions+=f'def heads{rows}(oq,ok,q,k,gq,gk,eq,ek):\n    return (*[t.tile((1,{rows},128)) for t in (oq,ok,q,k,gq,gk)],eq,ek)\n\n'
        for dc in [0,1]:expressions+=application(rows,dc)[1]+'\n'
    expr_path=w/'scripts/expressions.py';expr_path.write_text(expressions)
    module_spec=importlib.util.spec_from_file_location('qk_expressions',expr_path);module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    for rows in [4,8,16]:
        arrange_name=f'heads{rows}';arrange_source=f'def {arrange_name}(oq,ok,q,k,gq,gk,eq,ek):\n    return (*[t.tile((1,{rows},128)) for t in (oq,ok,q,k,gq,gk)],eq,ek)\n'
        for dc,label,dt in [(0,'f16',nt.float16),(1,'bf16',nt.bfloat16)]:
            app_name,app_source=application(rows,dc)
            name=f'h3_qknorm_{label}_r{rows}';print('GENERATE',name,flush=True);start=time.monotonic()
            args=tuple(Tensor(3,dtype=dt,other=0) for _ in range(6))+(Tensor(0,dtype=nt.float32),Tensor(0,dtype=nt.float32))
            nt.make(getattr(module,arrange_name),getattr(module,app_name),args,caller='cuda',kernel_name=name,output_dir=out,num_warps=4,num_stages=1)
            records.append(dict(name=name,dc=dc,rows=rows,seconds=time.monotonic()-start));(w/'generation-progress.json').write_text(json.dumps(records,indent=2))
    (w/'references/generated-expressions.py').write_text(expressions)
    params='NineToothedStream stream,NineToothedTensor oq,NineToothedTensor ok,NineToothedTensor q,NineToothedTensor k,NineToothedTensor gq,NineToothedTensor gk,NineToothedTensor eq,NineToothedTensor ek,int dc,int rows'
    header='#pragma once\n'+'\n'.join(f'#include "{r["name"]}.h"' for r in records)+'\n#ifdef __cplusplus\nextern "C" {\n#endif\nint qknorm_launch('+params+');\n#ifdef __cplusplus\n}\n#endif\n'
    source='#include "qknorm.h"\nint qknorm_launch('+params+'){\n'
    for r in records:source+=f'if(dc=={r["dc"]}&&rows=={r["rows"]})return launch_{r["name"]}(stream,oq,ok,q,k,gq,gk,eq,ek);\n'
    source+='return 1;\n}\n';(out/'qknorm.h').write_text(header);(out/'qknorm.c').write_text(source)
    (w/'generation.json').write_text(json.dumps(dict(passed=True,target=str(target),nine=str(Path(nt.__file__).resolve()),records=records,files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}),indent=2));print('GENERATION_PASS',len(records),flush=True)
if __name__=='__main__':main()
