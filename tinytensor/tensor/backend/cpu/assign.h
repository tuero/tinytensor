// assign.h
// Assign runner

#ifndef TINYTENSOR_BACKEND_CPU_ASSIGN_H_
#define TINYTENSOR_BACKEND_CPU_ASSIGN_H_

#include <tt/tensor.h>

namespace tinytensor::cpu {

void assign_runner(Tensor &lhs, const Tensor &rhs);

}    // namespace tinytensor::cpu

#endif    // TINYTENSOR_BACKEND_CPU_ASSIGN_H_
