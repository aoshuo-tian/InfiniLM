#pragma once
#include <infinicore/nn/rmsnorm.hpp>
#include <utility>
namespace h3qk {
std::pair<infinicore::Tensor,infinicore::Tensor> normalize(const infinicore::nn::RMSNorm*,const infinicore::nn::RMSNorm*,const infinicore::Tensor&,const infinicore::Tensor&,bool=true,int=8);
void into(const infinicore::Tensor&,const infinicore::Tensor&,const infinicore::Tensor&,const infinicore::Tensor&,const infinicore::Tensor&,const infinicore::Tensor&,float,float,int);
}
