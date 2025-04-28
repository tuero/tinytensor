// unary.h
// Element-wise unary runner

#ifndef TINYTENSOR_BACKEND_CPU_UNARY_H_
#define TINYTENSOR_BACKEND_CPU_UNARY_H_

#include <tt/tensor.h>

#include "tensor/backend/common/unary.h"

namespace tinytensor::cpu {

template <common::unary::UnaryOpT Op, typename... Params>
auto unary_runner(const Tensor &tensor, Params... params) -> Tensor;

template <common::unary::UnaryOpT Op, typename... Params>
void unary_runner_inplace(Tensor &tensor, Params... params);

}    // namespace tinytensor::cpu

#endif    // TINYTENSOR_BACKEND_CPU_UNARY_H_
