// clamp.hpp
// Element-wise clamp kernel

#ifndef TINYTENSOR_BACKEND_CPU_KERNEL_CLAMP_H_
#define TINYTENSOR_BACKEND_CPU_KERNEL_CLAMP_H_

#include <tt/scalar.h>

#include "tensor/backend/common/util.h"
#include "tensor/backend/cpu/data_types.h"

#include <cmath>

namespace tinytensor::cpu::kernel::clamp {

template <typename T>
void clamp_kernel(DataInfo<T> array, const DataInfo<const T> min, const DataInfo<const T> max, int N) {
    for (int i = 0; i < N; ++i) {
        const auto idx = to_flat_index(i, array.shape, array.stride, array.offset);
        const auto min_idx = to_flat_index(i, min.shape, min.stride, min.offset);
        const auto max_idx = to_flat_index(i, max.shape, max.stride, max.offset);
        array.data[idx] = std::min(std::max(array.data[idx], min.data[min_idx]), max.data[max_idx]);
    }
}

}    // namespace tinytensor::cpu::kernel::clamp

#endif    // TINYTENSOR_BACKEND_CPU_KERNEL_CLAMP_H_
