#pragma once
#include "h3_qkrope_f16_r8_i64.h"
#include "h3_qkrope_bf16_r8_i64.h"
#include "h3_qkrope_f16_r16_i64.h"
#include "h3_qkrope_bf16_r16_i64.h"
#ifdef __cplusplus
extern "C" {
#endif
int qkrope_launch(NineToothedStream stream,NineToothedTensor oq,NineToothedTensor ok,NineToothedTensor q,NineToothedTensor k,NineToothedTensor gq,NineToothedTensor gk,NineToothedTensor cache,NineToothedTensor pos,NineToothedTensor eq,NineToothedTensor ek,NineToothedTensor rd,int dc,int rows);
#ifdef __cplusplus
}
#endif
