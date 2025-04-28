// misc.h
// Element-wise misc runner

#ifndef TINYTENSOR_BACKEND_CUDA_MISC_H_
#define TINYTENSOR_BACKEND_CUDA_MISC_H_

#include <tt/tensor.h>

namespace tinytensor::cuda {

auto where_runner(const Tensor &cond, const Tensor &lhs, const Tensor &rhs) -> Tensor;
auto gather_runner(const Tensor &input, const Tensor &indices, int dim) -> Tensor;

}    // namespace tinytensor::cuda

#endif    // TINYTENSOR_BACKEND_CUDA_MISC_H_
