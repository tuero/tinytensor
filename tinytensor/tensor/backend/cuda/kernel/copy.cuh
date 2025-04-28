#ifndef TINYTENSOR_BACKEND_CUDA_KERNEL_COPY_H_
#define TINYTENSOR_BACKEND_CUDA_KERNEL_COPY_H_

#include "tensor/backend/common/util.h"
#include "tensor/backend/cuda/config.cuh"
#include "tensor/backend/cuda/data_types.cuh"

namespace tinytensor::cuda::kernel::copy {

template <typename T>
__global__ void binary_kernel(const DataInfo<const T> src, DataInfo<T> dest, int N) {
    const auto i = GLOBAL_FLAT_THREAD_IDX;
    if (i < N) {
        const auto src_idx = to_flat_index(i, src.shape, src.stride, src.offset);
        dest.data[i] = src.data[src_idx];
    }
}

}    // namespace tinytensor::cuda::kernel::copy

#endif    // TINYTENSOR_BACKEND_CUDA_KERNEL_COPY_H_
