// distribution.cuh
// Element-wise distribution kernel

#ifndef TINYTENSOR_BACKEND_CUDA_KERNEL_DISTRIBUTION_H_
#define TINYTENSOR_BACKEND_CUDA_KERNEL_DISTRIBUTION_H_

#include <tt/concepts.h>
#include <tt/random.h>
#include <tt/scalar.h>

#include "tensor/backend/common/util.h"
#include "tensor/backend/cuda/config.cuh"
#include "tensor/backend/cuda/data_types.cuh"

#include <cstdint>
#include <nvfunctional>

namespace tinytensor::cuda::kernel::distribution {

template <typename T, typename OP, typename... TS>
__global__ void
    variadic_param_kernel(const DeviceSpan<const uint64_t> gen_states, DataInfo<T> res, OP op, int N, TS... params) {
    const auto i = GLOBAL_FLAT_THREAD_IDX;
    if (i < N) {
        Generator gen = Generator::from_state(gen_states[i]);
        const auto idx_res = to_flat_index(i, res.shape, res.stride, res.offset);
        res.data[idx_res] = op()(params[i]..., gen);
    }
}

}    // namespace tinytensor::cuda::kernel::distribution

#endif    // TINYTENSOR_BACKEND_CUDA_KERNEL_DISTRIBUTION_H_
