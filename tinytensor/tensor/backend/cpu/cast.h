// cast.h
// Cast from one type to another

#ifndef TINYTENSOR_BACKEND_CPU_CAST_H_
#define TINYTENSOR_BACKEND_CPU_CAST_H_

#include <tt/scalar.h>
#include <tt/tensor.h>

namespace tinytensor::cpu {

auto cast_runner(const Tensor &tensor, ScalarType dtype) -> Tensor;

}    // namespace tinytensor::cpu

#endif    // TINYTENSOR_BACKEND_CPU_CAST_H_
