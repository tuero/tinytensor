// assign.h
// Assign runner

#ifndef TINYTENSOR_BACKEND_CUDA_ASSIGN_H_
#define TINYTENSOR_BACKEND_CUDA_ASSIGN_H_

#include <tt/scalar.h>
#include <tt/tensor.h>

namespace tinytensor::cuda {

void assign_runner(Tensor &lhs, const Tensor &rhs);

}    // namespace tinytensor::cuda

#endif    // TINYTENSOR_BACKEND_CUDA_ASSIGN_H_
