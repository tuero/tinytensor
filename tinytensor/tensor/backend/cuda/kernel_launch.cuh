// kernel_launch.cuh
// Kernel launch helper

#ifndef TINYTENSOR_CUDA_KERNEL_LAUNCH_H_
#define TINYTENSOR_CUDA_KERNEL_LAUNCH_H_

#include <tt/exception.h>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#include <string>

namespace tinytensor::cuda {

// Launch a kernel
template <typename Kernel, typename... Args>
void launch(const Kernel &kernel, const dim3 &grid_dim, const dim3 &block_dim, Args &&...args) {
    kernel<<<grid_dim, block_dim>>>(std::forward<Args>(args)...);
    const auto status = cudaGetLastError();
    if (status != cudaSuccess) {
        TT_ERROR("Kernel error: " + std::string(cudaGetErrorString(status)));
    }
}

}    // namespace tinytensor::cuda

#endif    // TINYTENSOR_STORAGE_CUDA_H_
