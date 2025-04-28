// assign.hpp
// Assignment kernels

#ifndef TINYTENSOR_BACKEND_CPU_KERNEL_ASSIGN_H_
#define TINYTENSOR_BACKEND_CPU_KERNEL_ASSIGN_H_

#include <tt/scalar.h>

#include "tensor/backend/common/util.h"
#include "tensor/backend/cpu/data_types.h"

namespace tinytensor::cpu::kernel::assign {

template <typename T>
void assign_kernel(DataInfo<T> lhs, const DataInfo<const T> rhs, int N) {
    for (int i = 0; i < N; ++i) {
        const auto lhs_idx = to_flat_index(i, lhs.shape, lhs.stride, lhs.offset);
        const auto rhs_idx = to_flat_index(i, rhs.shape, rhs.stride, rhs.offset);
        lhs.data[lhs_idx] = rhs.data[rhs_idx];
    }
}

}    // namespace tinytensor::cpu::kernel::assign

#endif    // TINYTENSOR_BACKEND_CPU_KERNEL_ASSIGN_H_
