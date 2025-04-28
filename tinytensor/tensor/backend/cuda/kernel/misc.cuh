// misc.cuh
// Element-wise distribution kernel

#ifndef TINYTENSOR_BACKEND_CUDA_KERNEL_MISC_H_
#define TINYTENSOR_BACKEND_CUDA_KERNEL_MISC_H_

#include <tt/concepts.h>
#include <tt/scalar.h>

#include "tensor/backend/common/misc.h"
#include "tensor/backend/common/util.h"
#include "tensor/backend/cuda/config.cuh"
#include "tensor/backend/cuda/data_types.cuh"

#include <nvfunctional>

namespace tinytensor::cuda::kernel::misc {

using namespace tinytensor::common::misc;

template <typename T>
__global__ void where_kernel(
    DeviceSpan<T> res,
    int N,
    const DataInfo<const to_ctype_t<kBool>> cond,
    const DataInfo<const T> lhs,
    const DataInfo<const T> rhs
) {
    using KernelOp = typename OpFactory<T, MiscOpT::where>::KernelOp;
    KernelOp op{};
    const auto i = GLOBAL_FLAT_THREAD_IDX;
    if (i < N) {
        const auto lhs_idx = to_flat_index(i, lhs.shape, lhs.stride, lhs.offset);
        const auto rhs_idx = to_flat_index(i, rhs.shape, rhs.stride, rhs.offset);
        const auto cond_idx = to_flat_index(i, cond.shape, cond.stride, cond.offset);
        res[i] = op()(cond.data[cond_idx], lhs.data[lhs_idx], rhs.data[rhs_idx]);
    }
}

template <typename T>
__global__ void gather_kernel(
    DeviceSpan<T> res,
    int N,
    const DataInfo<const T> input,
    const DataInfo<const int> indices,
    const DeviceSpan<const int> res_shape,
    int dim
) {
    const auto i = GLOBAL_FLAT_THREAD_IDX;
    if (i < N) {
        const auto base_input_idx = to_flat_index(i, res_shape, input.stride, input.offset);
        const auto gather_idx = to_flat_index(i, indices.shape, indices.stride, indices.offset);
        const auto offset = input.stride[dim] * indices.data[gather_idx];
        res[i] = input.data[base_input_idx + offset];
    }
}

}    // namespace tinytensor::cuda::kernel::misc

#endif    // TINYTENSOR_BACKEND_CUDA_KERNEL_MISC_H_
