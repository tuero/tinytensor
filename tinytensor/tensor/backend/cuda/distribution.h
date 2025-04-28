// distribution.h
// Element-wise distribution runner

#ifndef TINYTENSOR_BACKEND_CUDA_DISTRIBUTION_H_
#define TINYTENSOR_BACKEND_CUDA_DISTRIBUTION_H_

#include <tt/concepts.h>
#include <tt/random.h>
#include <tt/tensor.h>

#include "tensor/backend/common/distribution.h"

namespace tinytensor::cuda {

// @NOTE: Below has a link error on clang + cuda
// possible related issue: https://github.com/llvm/llvm-project/issues/62134

// template <common::distribution::DistributionOpT Op, typename... Params>
//     requires IsAllOf<Tensor, Params...>
// auto dist_runner(Generator &gen, const Params &...params) -> Tensor;

// template <common::distribution::DistributionOpT Op, typename... Params>
//     requires IsAllOf<Tensor, Params...>
// void dist_inplace_runner(Tensor &tensor, Generator &gen, const Params &...params);

// ------------------

// @NOTE: Fix is to explicitly list out each combination of params for external linking

template <common::distribution::DistributionOpT Op>
auto dist_runner(Generator &gen, const Tensor &p1) -> Tensor;

template <common::distribution::DistributionOpT Op>
auto dist_runner(Generator &gen, const Tensor &p1, const Tensor &p2) -> Tensor;

template <common::distribution::DistributionOpT Op>
void dist_inplace_runner(Tensor &tensor, Generator &gen, const Tensor &p1);

template <common::distribution::DistributionOpT Op>
void dist_inplace_runner(Tensor &tensor, Generator &gen, const Tensor &p1, const Tensor &p2);

}    // namespace tinytensor::cuda

#endif    // TINYTENSOR_BACKEND_CUDA_DISTRIBUTION_H_
