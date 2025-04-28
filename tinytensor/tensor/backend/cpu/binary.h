// binary.h
// Element-wise binary runner

#ifndef TINYTENSOR_BACKEND_CPU_BINARY_H_
#define TINYTENSOR_BACKEND_CPU_BINARY_H_

#include <tt/tensor.h>

#include "tensor/backend/common/binary.h"

namespace tinytensor::cpu {

template <common::binary::BinaryOpT Op>
auto binary_runner(const Tensor &lhs, const Tensor &rhs) -> Tensor;

template <common::binary::BinaryOpT Op>
void binary_inplace_runner(Tensor &lhs, const Tensor &rhs);

}    // namespace tinytensor::cpu

#endif    // TINYTENSOR_BACKEND_CPU_BINARY_H_
