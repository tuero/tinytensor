// distribution.h
// Element-wise distribution runner

#ifndef TINYTENSOR_BACKEND_CPU_DISTRIBUTION_H_
#define TINYTENSOR_BACKEND_CPU_DISTRIBUTION_H_

#include <tt/concepts.h>
#include <tt/random.h>
#include <tt/tensor.h>

#include "tensor/backend/common/distribution.h"

namespace tinytensor::cpu {

template <common::distribution::DistributionOpT Op, typename... Params>
    requires IsAllOf<Tensor, Params...>
auto dist_runner(Generator &gen, const Params &...params) -> Tensor;

template <common::distribution::DistributionOpT Op, typename... Params>
    requires IsAllOf<Tensor, Params...>
void dist_inplace_runner(Tensor &tensor, Generator &gen, const Params &...params);

}    // namespace tinytensor::cpu

#endif    // TINYTENSOR_BACKEND_CPU_DISTRIBUTION_H_
