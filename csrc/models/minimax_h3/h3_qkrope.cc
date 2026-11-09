#include "h3_qkrope.hpp"
#include "h3_qknorm.hpp"
#include "qkrope.h"
#include <infinicore/context/context.hpp>
#include <infinicore/ops/mrope.hpp>
#include <array>
#include <climits>
#include <cmath>
#include <stdexcept>
using namespace infinicore;
namespace h3qr {
namespace {
bool data(const Tensor&t){if(t->ndim()!=3||t->size(2)!=128||!t->size(0)||!t->size(1))return false;auto s=t->strides();return s[2]==1&&s[1]==128&&s[0]>=int64_t(t->size(1)*128)&&s[0]<=INT_MAX&&s[0]%16==0&&reinterpret_cast<uintptr_t>(t->data())%16==0&&(t->size(0)-1)*uint64_t(s[0])+t->size(1)*128<=INT_MAX;}
bool weight(const Tensor&t,const Tensor&q){return t->device()==q->device()&&t->dtype()==q->dtype()&&t->ndim()==1&&t->size(0)==128&&t->strides()[0]==1&&reinterpret_cast<uintptr_t>(t->data())%16==0;}
struct Meta{uint64_t shape[3];int64_t strides[3];};
Meta meta(const Tensor&t){auto s=t->strides();return {{t->size(0),t->size(1),t->size(2)},{s[0],s[1],s[2]}};}
}
bool eligible(const Tensor&q,const Tensor&k,const Tensor&gq,const Tensor&gk,const Tensor&cache,const Tensor&pos,float eq,float ek){
 if(context::isGraphRecording()||q->device().getType()!=Device::Type::HYGON||(q->dtype()!=DataType::F16&&q->dtype()!=DataType::BF16)||q->device()!=k->device()||q->dtype()!=k->dtype()||q->shape()!=k->shape()||!data(q)||!data(k)||!weight(gq,q)||!weight(gk,q)||!std::isfinite(eq)||eq<=0||!std::isfinite(ek)||ek<=0)return false;
 if(cache->device()!=q->device()||cache->dtype()!=q->dtype()||cache->ndim()!=2||!cache->size(0)||cache->size(1)>128||cache->size(1)<6||cache->size(1)%6||cache->strides()[1]!=1||cache->strides()[0]<int64_t(cache->size(1))||(cache->size(0)-1)*uint64_t(cache->strides()[0])+cache->size(1)>INT_MAX)return false;
 if(pos->device()!=q->device()||pos->dtype()!=DataType::I64||reinterpret_cast<uintptr_t>(cache->data())%16||reinterpret_cast<uintptr_t>(pos->data())%16)return false;
 return (pos->ndim()==1&&pos->size(0)==q->size(0)&&pos->strides()[0]>0&&uint64_t(pos->strides()[0])*q->size(0)<=INT_MAX)||(pos->ndim()==2&&pos->size(0)==3&&pos->size(1)==q->size(0)&&pos->strides()[0]>=0&&pos->strides()[1]>0&&uint64_t(pos->strides()[0])*2+uint64_t(pos->strides()[1])*q->size(0)<=INT_MAX);
}
void into(const Tensor&oq,const Tensor&ok,const Tensor&q,const Tensor&k,const Tensor&gq,const Tensor&gk,const Tensor&cache,const Tensor&pos,float eq,float ek,int rows){
 if(!eligible(q,k,gq,gk,cache,pos,eq,ek)||!data(oq)||!data(ok)||oq->shape()!=q->shape()||ok->shape()!=q->shape()||oq->dtype()!=q->dtype()||ok->dtype()!=q->dtype()||oq->device()!=q->device()||ok->device()!=q->device()||(rows!=8&&rows!=16))throw std::invalid_argument("qkrope unsupported arguments");
 auto range=[](const Tensor&t){auto s=t->strides();uint64_t n=1;for(size_t i=0;i<t->ndim();++i)n+=(t->size(i)-1)*s[i];auto b=reinterpret_cast<uintptr_t>(t->data());return std::pair<uintptr_t,uintptr_t>{b,b+n*(t->dtype()==DataType::I64?8:2)};};
 auto overlaps=[&](const Tensor&a,const Tensor&b){auto x=range(a),y=range(b);return x.first<y.second&&y.first<x.second;};
 if(overlaps(oq,ok))throw std::invalid_argument("qkrope overlapping outputs");
 for(const auto&t:{q,k,gq,gk,cache,pos})if(overlaps(oq,t)||overlaps(ok,t))throw std::invalid_argument("qkrope output overlaps input");
 context::setDevice(q->device());std::array<Meta,4>m={meta(oq),meta(ok),meta(q),meta(k)};Meta g={{q->size(0),q->size(1),128},{0,0,1}};
 NineToothedTensor a{const_cast<std::byte*>(oq->data()),m[0].shape,m[0].strides},b{const_cast<std::byte*>(ok->data()),m[1].shape,m[1].strides},c{const_cast<std::byte*>(q->data()),m[2].shape,m[2].strides},d{const_cast<std::byte*>(k->data()),m[3].shape,m[3].strides},e{const_cast<std::byte*>(gq->data()),g.shape,g.strides},f{const_cast<std::byte*>(gk->data()),g.shape,g.strides};
 uint64_t cs[2]={cache->size(0),cache->size(1)},ps[2]={3,q->size(0)};int64_t ct[2]={cache->strides()[0],1},pt[2]={pos->ndim()==1?0:pos->strides()[0],pos->ndim()==1?pos->strides()[0]:pos->strides()[1]};
 NineToothedTensor ca{const_cast<std::byte*>(cache->data()),cs,ct},po{const_cast<std::byte*>(pos->data()),ps,pt};double eqd=eq,ekd=ek,rdd=cache->size(1);NineToothedTensor epq{&eqd,nullptr,nullptr},epk{&ekd,nullptr,nullptr},rd{&rdd,nullptr,nullptr};
 int rc=qkrope_launch(context::getStream(),a,b,c,d,e,f,ca,po,epq,epk,rd,q->dtype()==DataType::F16?0:1,rows);if(rc)throw std::runtime_error("qkrope launch failed "+std::to_string(rc));
}
std::pair<Tensor,Tensor> forward(const nn::RMSNorm*nq,const nn::RMSNorm*nk,const Tensor&q,const Tensor&k,const Tensor&cache,const Tensor&pos,int rows){
 if(rows!=0&&rows!=1&&eligible(q,k,nq->weight(),nk->weight(),cache,pos,float(nq->eps()),float(nk->eps()))){auto a=Tensor::empty(q->shape(),q->dtype(),q->device()),b=Tensor::empty(k->shape(),k->dtype(),k->device());into(a,b,q,k,nq->weight(),nk->weight(),cache,pos,float(nq->eps()),float(nk->eps()),rows);return {a,b};}
 auto pair=h3qk::normalize(nq,nk,q,k,rows!=0&&q->size(0)>=65,16);Size t=q->size(0),h=q->size(1),hd=q->size(2),rd=cache->size(1),half=rd/2;auto out=op::mrope(pair.first->view({t,h*hd}),pair.second->view({t,h*hd}),cache->narrow({{1,0,half}}),cache->narrow({{1,half,half}}),pos,hd,rd,half/3,half/3,half/3,false);return {out.first->view(q->shape()),out.second->view(k->shape())};
}
}
