#include "qkrope.h"
int qkrope_launch(NineToothedStream stream,NineToothedTensor oq,NineToothedTensor ok,NineToothedTensor q,NineToothedTensor k,NineToothedTensor gq,NineToothedTensor gk,NineToothedTensor cache,NineToothedTensor pos,NineToothedTensor eq,NineToothedTensor ek,NineToothedTensor rd,int dc,int rows){
if(dc==0&&rows==8)return launch_h3_qkrope_f16_r8_i64(stream,oq,ok,q,k,gq,gk,cache,pos,eq,ek,rd);
if(dc==1&&rows==8)return launch_h3_qkrope_bf16_r8_i64(stream,oq,ok,q,k,gq,gk,cache,pos,eq,ek,rd);
if(dc==0&&rows==16)return launch_h3_qkrope_f16_r16_i64(stream,oq,ok,q,k,gq,gk,cache,pos,eq,ek,rd);
if(dc==1&&rows==16)return launch_h3_qkrope_bf16_r16_i64(stream,oq,ok,q,k,gq,gk,cache,pos,eq,ek,rd);
return 1;
}
