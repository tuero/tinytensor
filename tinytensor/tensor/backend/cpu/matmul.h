// matmul.h
// Matmul runner

#ifndef TINYTENSOR_BACKEND_CPU_MATMUL_H_
#define TINYTENSOR_BACKEND_CPU_MATMUL_H_

#include <tt/tensor.h>

namespace tinytensor::cpu {

auto batched_matmul_runner(const Tensor &lhs, const Tensor &rhs) -> Tensor;

}    // namespace tinytensor::cpu

#endif    // TINYTENSOR_BACKEND_CPU_MATMUL_H_
