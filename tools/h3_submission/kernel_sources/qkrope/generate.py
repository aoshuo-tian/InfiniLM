"""Q/K RMSNorm + H3 partial three-axis NeoX M-RoPE, through NineToothed."""
from pathlib import Path
import os,json,time,hashlib,importlib.util
import ninetoothed as nt
from ninetoothed import Tensor
from triton.runtime import driver
w=Path(os.environ['QK_ROPE_WORK']);out=w/'generated'

def main():
    target=driver.active.get_current_target();assert (target.backend,target.arch)==('hip','gfx936')
    spec=importlib.util.spec_from_file_location('previous_qk_generator',w/'references/qknorm_generate.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    source='import ninetoothed.language as ntl\n';jobs=[]
    for rows in [8,16]:
        source+=f'''def arrange_{rows}(oq,ok,q,k,gq,gk,cache,pos,eq,ek,rd):
    oq=oq.tile((1,{rows},128))
    ok=ok.tile((1,{rows},128))
    q=q.tile((1,{rows},128))
    k=k.tile((1,{rows},128))
    gq=gq.tile((1,{rows},128))
    gk=gk.tile((1,{rows},128))
    cache=cache.flatten().tile((1,)).unsqueeze(1).unsqueeze(2).expand(oq.shape)
    pos=pos.flatten().tile((1,)).unsqueeze(1).unsqueeze(2).expand(oq.shape)
    return oq,ok,q,k,gq,gk,cache,pos,eq,ek,rd

'''
        for dc,label,dt in [(0,'f16',nt.float16),(1,'bf16',nt.bfloat16)]:
            name=f'apply_{rows}_{dc}';body=old.application(rows,dc)[1].split('\n',1)[1]
            for p in ['q','k']:body=body.replace(f'    o{p}=ntl.reshape({p}rounded,(1,{rows},128))\n',f'    {p}norm=ntl.cast(ntl.cast({p}rounded,ntl.{"float16" if dc==0 else "bfloat16"}),ntl.float32)\n')
            source+=f'def {name}(oq,ok,q,k,gq,gk,cache,pos,eq,ek,rd):\n'+body
            source+=f'''    rotary=ntl.cast(rd,ntl.int32)
    half=rotary//2
    col=ntl.arange(0,128)
    partner=ntl.where(col<rotary,ntl.where(col<half,col+half,col-half),col)
    gather_index=ntl.broadcast_to(ntl.expand_dims(partner,0),({rows},128))
    qp=ntl.gather(qnorm,gather_index,1)
    kp=ntl.gather(knorm,gather_index,1)
    token=ntl.reshape(oq.offsets(0),(1,1))
    freq=ntl.expand_dims(ntl.where(col<half,col,col-half),0)
    section=half//3
    axis=ntl.where(freq<section,0,ntl.where(freq<2*section,1,2))
    valid=(token<q.source.shape[0])&(ntl.expand_dims(col,0)<rotary)
    position=ntl.load(pos.source.data_ptr()+axis*pos.source.stride(0)+token*pos.source.stride(1),mask=valid,other=0)
    position=ntl.where((position>=0)&(position<cache.source.shape[0]),position,0)
    offset=position*cache.source.stride(0)+freq
    cs=ntl.cast(ntl.load(cache.source.data_ptr()+offset,mask=valid,other=1),ntl.float32)
    sn=ntl.cast(ntl.load(cache.source.data_ptr()+offset+half,mask=valid,other=0),ntl.float32)
    signed_sn=ntl.where(ntl.expand_dims(col,0)<half,-sn,sn)
'''
            if dc==0:
                for p in ['q','k']:
                    source+=f'''    {p}a={p}norm*cs
    {p}b={p}p*signed_sn
    {p}r={p}a+{p}b
    {p}v={p}r-{p}a
    {p}err=({p}a-({p}r-{p}v))+({p}b-{p}v)
    {p}rbits=ntl.cast({p}r,ntl.int32,bitcast=True)
    {p}rabs={p}rbits&2147483647
    {p}rexp=({p}rabs>>23)&255
    {p}rdiscard=ntl.maximum(13,ntl.minimum(24,126-{p}rexp))
    {p}rsig=({p}rabs&8388607)|8388608
    {p}rmid=({p}rsig&((1<<{p}rdiscard)-1))==(1<<({p}rdiscard-1))
    {p}rdirection=ntl.where(({p}rbits<0)==({p}err<0),1,-1)
    {p}rcorrect=ntl.cast({p}rbits+{p}rdirection,ntl.float32,bitcast=True)
    {p}r=ntl.where({p}rmid&({p}err!=0)&({p}rexp>=102)&({p}rexp<=142),{p}rcorrect,{p}r)
'''
            else:source+='    qr=ntl.fma(qnorm,cs,qp*signed_sn)\n    kr=ntl.fma(knorm,cs,kp*signed_sn)\n'
            source+=f'''    oq=ntl.reshape(ntl.where(ntl.expand_dims(col,0)<rotary,qr,qnorm),(1,{rows},128))
    ok=ntl.reshape(ntl.where(ntl.expand_dims(col,0)<rotary,kr,knorm),(1,{rows},128))

'''
            jobs.append(dict(rows=rows,dc=dc,label=label,dt=dt,application=name))
    p=w/'scripts/expressions.py';p.write_text(source)
    spec=importlib.util.spec_from_file_location('qk_rope_expressions',p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    records=[]
    for job in jobs:
        # Start with the actual native runner's int64 position IDs. I32 retains baseline fallback.
        name=f'h3_qkrope_{job["label"]}_r{job["rows"]}_i64';print('GENERATE',name,flush=True);start=time.monotonic()
        args=tuple(Tensor(3,dtype=job['dt'],other=0) for _ in range(6))+(Tensor(2,dtype=job['dt'],other=0),Tensor(2,dtype=nt.int64,other=0),Tensor(0,dtype=nt.float32),Tensor(0,dtype=nt.float32),Tensor(0,dtype=nt.float32))
        nt.make(getattr(module,'arrange_'+str(job['rows'])),getattr(module,job['application']),args,caller='cuda',kernel_name=name,output_dir=out,num_warps=4,num_stages=1)
        records.append(dict(name=name,rows=job['rows'],dc=job['dc'],seconds=time.monotonic()-start))
        (w/'generation-progress.json').write_text(json.dumps(records,indent=2))
    params='NineToothedStream stream,NineToothedTensor oq,NineToothedTensor ok,NineToothedTensor q,NineToothedTensor k,NineToothedTensor gq,NineToothedTensor gk,NineToothedTensor cache,NineToothedTensor pos,NineToothedTensor eq,NineToothedTensor ek,NineToothedTensor rd,int dc,int rows'
    header='#pragma once\n'+'\n'.join('#include "'+r['name']+'.h"' for r in records)+'\n#ifdef __cplusplus\nextern "C" {\n#endif\nint qkrope_launch('+params+');\n#ifdef __cplusplus\n}\n#endif\n'
    cpp='#include "qkrope.h"\nint qkrope_launch('+params+'){\n'
    for r in records:cpp+=f'if(dc=={r["dc"]}&&rows=={r["rows"]})return launch_{r["name"]}(stream,oq,ok,q,k,gq,gk,cache,pos,eq,ek,rd);\n'
    cpp+='return 1;\n}\n';(out/'qkrope.h').write_text(header);(out/'qkrope.c').write_text(cpp)
    (w/'generation.json').write_text(json.dumps(dict(passed=True,target=str(target),math='Preserve rounded RMSNorm intermediate; FP16 rotation uses TwoSum error and half-midpoint correction to match original direct-to-half rounding. BF16 uses original FMA order.',records=records,files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}),indent=2));print('GENERATION_PASS',len(records),flush=True)
if __name__=='__main__':main()
