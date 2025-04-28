// clamp.h
// Element-wise clamp runner

#ifndef TINYTENSOR_BACKEND_CUDA_CLAMP_H_
#define TINYTENSOR_BACKEND_CUDA_CLAMP_H_

#include <tt/tensor.h>

namespace tinytensor::cuda {

void clamp_inplace_runner(Tensor &tensor, const Tensor &min, const Tensor &max);

}    // namespace tinytensor::cuda

#endif    // TINYTENSOR_BACKEND_CUDA_CLAMP_H_
