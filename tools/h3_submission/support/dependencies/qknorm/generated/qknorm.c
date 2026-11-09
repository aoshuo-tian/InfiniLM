#include "qknorm.h"
int qknorm_launch(NineToothedStream stream,NineToothedTensor oq,NineToothedTensor ok,NineToothedTensor q,NineToothedTensor k,NineToothedTensor gq,NineToothedTensor gk,NineToothedTensor eq,NineToothedTensor ek,int dc,int rows){
if(dc==0&&rows==4)return launch_h3_qknorm_f16_r4(stream,oq,ok,q,k,gq,gk,eq,ek);
if(dc==1&&rows==4)return launch_h3_qknorm_bf16_r4(stream,oq,ok,q,k,gq,gk,eq,ek);
if(dc==0&&rows==8)return launch_h3_qknorm_f16_r8(stream,oq,ok,q,k,gq,gk,eq,ek);
if(dc==1&&rows==8)return launch_h3_qknorm_bf16_r8(stream,oq,ok,q,k,gq,gk,eq,ek);
if(dc==0&&rows==16)return launch_h3_qknorm_f16_r16(stream,oq,ok,q,k,gq,gk,eq,ek);
if(dc==1&&rows==16)return launch_h3_qknorm_bf16_r16(stream,oq,ok,q,k,gq,gk,eq,ek);
return 1;
}
