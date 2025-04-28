// reduce.h
// Reduction runner

#ifndef TINYTENSOR_BACKEND_CUDA_REDUCE_H_
#define TINYTENSOR_BACKEND_CUDA_REDUCE_H_

#include <tt/tensor.h>

#include "tensor/backend/common/reduce.h"

namespace tinytensor::cuda {

template <common::reduce::ReduceOpT Op>
auto reduce_all_runner(const Tensor &tensor) -> Tensor;

template <common::reduce::ReduceOpT Op>
auto reduce_dim_runner(const Tensor &tensor, int dim) -> Tensor;

}    // namespace tinytensor::cuda

#endif    // TINYTENSOR_BACKEND_CUDA_REDUCE_H_
