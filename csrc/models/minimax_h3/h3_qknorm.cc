#include "h3_qknorm.hpp"
#include "qknorm.h"
#include <infinicore/context/context.hpp>
#include <array>
#include <climits>
#include <cmath>
#include <stdexcept>
using namespace infinicore;
namespace h3qk {
namespace {
struct Meta {uint64_t shape[3];int64_t strides[3];};
bool valid(const Tensor&t){
 if(t->ndim()!=3||t->size(2)!=128||!t->size(0)||!t->size(1)||t->size(0)>INT_MAX||t->size(1)>INT_MAX/128)return false;
 auto st=t->strides();return st[2]==1&&st[1]==128&&st[0]>=int64_t(t->size(1)*128)&&st[0]<=INT_MAX&&st[0]%16==0&&(t->size(0)-1)*uint64_t(st[0])+t->size(1)*128<=INT_MAX&&reinterpret_cast<uintptr_t>(t->data())%16==0;
}
bool weight(const Tensor&w,DataType dt,Device device){return w->device()==device&&w->dtype()==dt&&w->ndim()==1&&w->size(0)==128&&w->strides()[0]==1&&reinterpret_cast<uintptr_t>(w->data())%16==0;}
bool eligible(const Tensor&q,const Tensor&k,const Tensor&gq,const Tensor&gk,float eq,float ek){
 auto dt=q->dtype();return !context::isGraphRecording()&&q->device().getType()==Device::Type::HYGON&&k->device()==q->device()&&q->shape()==k->shape()&&k->dtype()==dt&&(dt==DataType::F16||dt==DataType::BF16)&&valid(q)&&valid(k)&&weight(gq,dt,q->device())&&weight(gk,dt,q->device())&&std::isfinite(eq)&&eq>0&&std::isfinite(ek)&&ek>0;
}
Meta meta(const Tensor&t){auto s=t->strides();return {{t->size(0),t->size(1),t->size(2)},{s[0],s[1],s[2]}};}
}
void into(const Tensor&oq,const Tensor&ok,const Tensor&q,const Tensor&k,const Tensor&gq,const Tensor&gk,float eq,float ek,int rows){
 if(!eligible(q,k,gq,gk,eq,ek)||!valid(oq)||!valid(ok)||oq->shape()!=q->shape()||ok->shape()!=q->shape()||oq->dtype()!=q->dtype()||ok->dtype()!=q->dtype()||oq->device()!=q->device()||ok->device()!=q->device()||(rows!=4&&rows!=8&&rows!=16))throw std::invalid_argument("qknorm unsupported arguments");
 // Validate disjoint output ranges before any device write. Q/K may share read-only QKV storage.
 auto range=[](const Tensor&t){auto st=t->strides();auto b=reinterpret_cast<uintptr_t>(t->data());uint64_t n=1;for(size_t i=0;i<t->ndim();++i)n+=(t->size(i)-1)*st[i];return std::pair<uintptr_t,uintptr_t>{b,b+n*2};};
 auto overlap=[&](const Tensor&a,const Tensor&b){auto x=range(a),y=range(b);return x.first<y.second&&y.first<x.second;};
 if(overlap(oq,ok)||overlap(oq,q)||overlap(oq,k)||overlap(ok,q)||overlap(ok,k)||overlap(oq,gq)||overlap(oq,gk)||overlap(ok,gq)||overlap(ok,gk))throw std::invalid_argument("qknorm output overlaps input/output");
 context::setDevice(q->device());std::array<Meta,4> m={meta(oq),meta(ok),meta(q),meta(k)};Meta g={{q->size(0),q->size(1),128},{0,0,1}};
 NineToothedTensor a{const_cast<std::byte*>(oq->data()),m[0].shape,m[0].strides},b{const_cast<std::byte*>(ok->data()),m[1].shape,m[1].strides},c{const_cast<std::byte*>(q->data()),m[2].shape,m[2].strides},d{const_cast<std::byte*>(k->data()),m[3].shape,m[3].strides},e{const_cast<std::byte*>(gq->data()),g.shape,g.strides},f{const_cast<std::byte*>(gk->data()),g.shape,g.strides};
 double eqd=eq,ekd=ek;NineToothedTensor epq{&eqd,nullptr,nullptr},epk{&ekd,nullptr,nullptr};int rc=qknorm_launch(context::getStream(),a,b,c,d,e,f,epq,epk,q->dtype()==DataType::F16?0:1,rows);if(rc)throw std::runtime_error("qknorm launch failed "+std::to_string(rc));
}
std::pair<Tensor,Tensor> normalize(const nn::RMSNorm*nq,const nn::RMSNorm*nk,const Tensor&q,const Tensor&k,bool prefer,int rows){
 // H56 qualifies from 65 tokens. TP2/4/8 local heads are only
 // performance-qualified from 1024 here; keep shorter shards on the old pair.
 if(!prefer||!eligible(q,k,nq->weight(),nk->weight(),float(nq->eps()),float(nk->eps()))
    || (q->size(1)<56 && q->size(0)<1024))return {nq->forward(q),nk->forward(k)};
 auto oq=Tensor::empty(q->shape(),q->dtype(),q->device()),ok=Tensor::empty(k->shape(),k->dtype(),k->device());into(oq,ok,q,k,nq->weight(),nk->weight(),float(nq->eps()),float(nk->eps()),rows);return {oq,ok};
}
}
