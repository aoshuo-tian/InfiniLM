#pragma once
#include "h3_qknorm_f16_r4.h"
#include "h3_qknorm_bf16_r4.h"
#include "h3_qknorm_f16_r8.h"
#include "h3_qknorm_bf16_r8.h"
#include "h3_qknorm_f16_r16.h"
#include "h3_qknorm_bf16_r16.h"
#ifdef __cplusplus
extern "C" {
#endif
int qknorm_launch(NineToothedStream stream,NineToothedTensor oq,NineToothedTensor ok,NineToothedTensor q,NineToothedTensor k,NineToothedTensor gq,NineToothedTensor gk,NineToothedTensor eq,NineToothedTensor ek,int dc,int rows);
#ifdef __cplusplus
}
#endif
