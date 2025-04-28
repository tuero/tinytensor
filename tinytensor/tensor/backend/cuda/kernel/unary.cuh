// unary.cuh
// Element-wise unary kernel

#ifndef TINYTENSOR_BACKEND_CUDA_KERNEL_UNARY_H_
#define TINYTENSOR_BACKEND_CUDA_KERNEL_UNARY_H_

#include "tensor/backend/common/util.h"
#include "tensor/backend/cuda/config.cuh"
#include "tensor/backend/cuda/data_types.cuh"

#include <cstddef>

namespace tinytensor::cuda::kernel::unary {

template <bool CastBeforeOp, typename T, typename R, typename OP, typename... Args>
__global__ void unary_kernel(const DataInfo<const T> di, DeviceSpan<R> res, OP op, int N, Args... args) {
    const auto i = GLOBAL_FLAT_THREAD_IDX;
    if (i < N) {
        const auto idx = static_cast<std::size_t>(to_flat_index(i, di.shape, di.stride, di.offset));
        // Some ops need to cast input before op, some after
        if constexpr (CastBeforeOp) {
            res[i] = op()(static_cast<R>(di.data[idx]), args...);
        } else {
            res[i] = static_cast<R>(op()(di.data[idx], args...));
        }
    }
}

template <typename T, typename OP, typename... Args>
__global__ void unary_kernel_inplace(DataInfo<T> di, OP op, int N, Args... args) {
    const auto i = GLOBAL_FLAT_THREAD_IDX;
    if (i < N) {
        const auto idx = to_flat_index(i, di.shape, di.stride, di.offset);
        di.data[idx] = op()(di.data[idx], args...);
    }
}

}    // namespace tinytensor::cuda::kernel::unary

#endif    // TINYTENSOR_BACKEND_CUDA_KERNEL_UNARY_H_
