// misc.h
// Element-wise misc runner

#ifndef TINYTENSOR_BACKEND_CPU_MISC_H_
#define TINYTENSOR_BACKEND_CPU_MISC_H_

#include <tt/tensor.h>

namespace tinytensor::cpu {

auto where_runner(const Tensor &cond, const Tensor &lhs, const Tensor &rhs) -> Tensor;
auto gather_runner(const Tensor &input, const Tensor &indices, int dim) -> Tensor;

}    // namespace tinytensor::cpu

#endif    // TINYTENSOR_BACKEND_CPU_MISC_H_
