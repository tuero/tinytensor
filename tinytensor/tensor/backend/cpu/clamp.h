// clamp.h
// Element-wise clamp runner

#ifndef TINYTENSOR_BACKEND_CPU_CLAMP_H_
#define TINYTENSOR_BACKEND_CPU_CLAMP_H_

#include <tt/tensor.h>

namespace tinytensor::cpu {

void clamp_inplace_runner(Tensor &tensor, const Tensor &min, const Tensor &max);

}    // namespace tinytensor::cpu

#endif    // TINYTENSOR_BACKEND_CPU_CLAMP_H_
