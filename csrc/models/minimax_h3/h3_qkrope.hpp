#pragma once
#include <infinicore/nn/rmsnorm.hpp>
#include <utility>
namespace h3qr {
bool eligible(const infinicore::Tensor&,const infinicore::Tensor&,const infinicore::Tensor&,const infinicore::Tensor&,const infinicore::Tensor&,const infinicore::Tensor&,float,float);
std::pair<infinicore::Tensor,infinicore::Tensor> forward(const infinicore::nn::RMSNorm*,const infinicore::nn::RMSNorm*,const infinicore::Tensor&,const infinicore::Tensor&,const infinicore::Tensor&,const infinicore::Tensor&,int=16);
void into(const infinicore::Tensor&,const infinicore::Tensor&,const infinicore::Tensor&,const infinicore::Tensor&,const infinicore::Tensor&,const infinicore::Tensor&,const infinicore::Tensor&,const infinicore::Tensor&,float,float,int);
}
