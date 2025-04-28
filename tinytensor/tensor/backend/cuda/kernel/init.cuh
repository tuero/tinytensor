// init.cuh
// Initialization kernel runner

#ifndef TINYTENSOR_BACKEND_CUDA_KERNEL_INIT_H_
#define TINYTENSOR_BACKEND_CUDA_KERNEL_INIT_H_

#include "tensor/backend/cuda/config.cuh"

namespace tinytensor::cuda::kernel::init {

// Each thread sets from the same value
template <typename T>
__global__ void init_full_kernel(T *data, const T value, int N) {
    const auto i = GLOBAL_FLAT_THREAD_IDX;
    if (i < N) {
        data[i] = value;    // NOLINT(*-pointer-arithmetic)
    }
}

// Threads set sequentially by thread idx
template <typename T>
__global__ void init_arange_kernel(T *data, int N) {
    const auto i = GLOBAL_FLAT_THREAD_IDX;
    if (i < N) {
        data[i] = static_cast<T>(i);    // NOLINT(*-pointer-arithmetic)
    }
}

}    // namespace tinytensor::cuda::kernel::init

#endif    // TINYTENSOR_BACKEND_CUDA_KERNEL_INIT_H_
