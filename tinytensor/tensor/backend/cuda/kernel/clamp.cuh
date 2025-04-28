// clamp.cuh
// Element-wise clamp kernel

#ifndef TINYTENSOR_BACKEND_CUDA_KERNEL_CLAMP_H_
#define TINYTENSOR_BACKEND_CUDA_KERNEL_CLAMP_H_

#include <tt/scalar.h>

#include "tensor/backend/common/util.h"
#include "tensor/backend/cuda/config.cuh"
#include "tensor/backend/cuda/data_types.cuh"

#include <nvfunctional>

namespace tinytensor::cuda::kernel::clamp {

template <typename T>
__global__ void clamp_kernel(DataInfo<T> array, const DataInfo<const T> min, const DataInfo<const T> max, int N) {
    const auto i = GLOBAL_FLAT_THREAD_IDX;
    if (i < N) {
        const auto idx = to_flat_index(i, array.shape, array.stride, array.offset);
        const auto min_idx = to_flat_index(i, min.shape, min.stride, min.offset);
        const auto max_idx = to_flat_index(i, max.shape, max.stride, max.offset);
        array.data[idx] = std::min(std::max(array.data[idx], min.data[min_idx]), max.data[max_idx]);
    }
}

}    // namespace tinytensor::cuda::kernel::clamp

#endif    // TINYTENSOR_BACKEND_CUDA_KERNEL_CLAMP_H_
