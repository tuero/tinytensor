// assign.cuh
// Assignment kernels

#ifndef TINYTENSOR_BACKEND_CUDA_KERNEL_ASSIGN_H_
#define TINYTENSOR_BACKEND_CUDA_KERNEL_ASSIGN_H_

#include <tt/scalar.h>

#include "tensor/backend/common/util.h"
#include "tensor/backend/cuda/config.cuh"
#include "tensor/backend/cuda/data_types.cuh"

namespace tinytensor::cuda::kernel::assign {

template <typename T>
__global__ void assign_kernel(DataInfo<T> lhs, const DataInfo<const T> rhs, int N) {
    const auto i = GLOBAL_FLAT_THREAD_IDX;
    if (i < N) {
        const auto lhs_idx = to_flat_index(i, lhs.shape, lhs.stride, lhs.offset);
        const auto rhs_idx = to_flat_index(i, rhs.shape, rhs.stride, rhs.offset);
        lhs.data[lhs_idx] = rhs.data[rhs_idx];
    }
}

}    // namespace tinytensor::cuda::kernel::assign

#endif    // TINYTENSOR_BACKEND_CUDA_KERNEL_BINARY_H_
