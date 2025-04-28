// matmul.h
// Matmul runner

#ifndef TINYTENSOR_BACKEND_CUDA_MATMUL_H_
#define TINYTENSOR_BACKEND_CUDA_MATMUL_H_

#include <tt/tensor.h>

namespace tinytensor::cuda {

auto batched_matmul_runner(const Tensor &lhs, const Tensor &rhs) -> Tensor;

}    // namespace tinytensor::cuda

#endif    // TINYTENSOR_BACKEND_CUDA_MATMUL_H_
