// cast.h
// Cast from one type to another

#ifndef TINYTENSOR_BACKEND_CUDA_CAST_H_
#define TINYTENSOR_BACKEND_CUDA_CAST_H_

#include <tt/scalar.h>
#include <tt/tensor.h>

namespace tinytensor::cuda {

auto cast_runner(const Tensor &tensor, ScalarType dtype) -> Tensor;

}    // namespace tinytensor::cuda

#endif    // TINYTENSOR_BACKEND_CUDA_CAST_H_
