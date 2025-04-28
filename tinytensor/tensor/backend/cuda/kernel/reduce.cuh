// reduce.cuh
// Reduction kernel

#ifndef TINYTENSOR_BACKEND_CUDA_KERNEL_REDUCE_H_
#define TINYTENSOR_BACKEND_CUDA_KERNEL_REDUCE_H_

#include "tensor/backend/common/util.h"
#include "tensor/backend/cuda/config.cuh"
#include "tensor/backend/cuda/data_types.cuh"

#include <cassert>
#include <cstddef>
#include <cstdio>

namespace tinytensor::cuda::kernel::reduce {

// Each thread represents a non-reduced element, which is reducing over the given dim
template <typename T, typename R, typename OP>
__global__ void reduce_dim_kernel(const DataInfo<const T> v, DeviceSpan<R> res, OP op, int dim, int N) {
    using VT = OP::VT;
    using VIT = OP::VIT;
    const auto i = GLOBAL_FLAT_THREAD_IDX;
    if (i < N) {
        auto out_val = OP::padding_value;
        int out_idx = -1;
        const auto start_idx = to_flat_index(i, v.shape, v.stride, v.offset, dim);
        for (int j = 0; j < v.shape[dim]; ++j) {
            const auto idx = to_flat_index(start_idx + j * v.stride[dim], v.shape, v.stride, v.offset);
            const VIT out_val_idx{out_val, out_idx};
            const VIT curr_val_idx{static_cast<VT>(v.data[idx]), j};
            out_idx = op.index()(curr_val_idx, out_val_idx);
            out_val = op.value()(curr_val_idx, out_val_idx);
        }
        res[i] = static_cast<R>(op.res()({out_val, out_idx}));
    }
}

// Fetch elements into a partial reduction array
// With decreasing stride, fold the right half into the left half
// Once a single warp of elements remain, use a wrap reduction
// Half the threads are idle on first iteration (the right half elements),
// so we perform the first reduction using half the number of threads, keeping all threads acitive
// initially
constexpr unsigned int MASK = 0xffffffff;
template <typename T, typename R, typename OP>
__global__ void reduce_all_kernel(const DataInfo<const T> v, DeviceSpan<R> res, OP op, std::size_t N) {
    using VT = OP::VT;
    using VIT = OP::VIT;
    volatile __shared__ VT partial_reduction_val[THREADS_PER_BLOCK];     // NOLINT(*-avoid-c-arrays)
    volatile __shared__ int partial_reduction_idx[THREADS_PER_BLOCK];    // NOLINT(*-avoid-c-arrays)
    const auto tid = threadIdx.x;

    // Number of blocks halved so we can perform first reduction with two loads
    // This keeps all threads active on first loop iteration
    const int i = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x * 2) + static_cast<int>(threadIdx.x);
    const auto idx1 = static_cast<std::size_t>(to_flat_index(i, v.shape, v.stride, v.offset));
    const auto idx2 = static_cast<std::size_t>(to_flat_index(i + blockDim.x, v.shape, v.stride, v.offset));

    // Padding value ensures reducing off the bounds results in values which don't change the result
    const auto val1 = i < static_cast<int>(N) ? static_cast<VT>(v.data[idx1]) : OP::padding_value;
    const int raw_idx1 = i < static_cast<int>(N) ? i : -1;
    const auto val2 = (i + blockDim.x) < N ? static_cast<VT>(v.data[idx2]) : OP::padding_value;
    const int raw_idx2 = (i + blockDim.x) < N ? static_cast<int>(i + blockDim.x) : -1;
    {
        const VIT val_idx1{val1, raw_idx1};
        const VIT val_idx2{val2, raw_idx2};
        // Ensure tiebreaks go towards smaller indices
        partial_reduction_val[tid] = op.value()(val_idx2, val_idx1);    // NOLINT(*-array-index)
        partial_reduction_idx[tid] = op.index()(val_idx2, val_idx1);    // NOLINT(*-array-index)
    }

    __syncthreads();

    // Fold each half using decreasing stride
    for (unsigned int s = blockDim.x / 2; s > WARP_SIZE; s >>= 1) {
        if (threadIdx.x < s) {
            // NOLINTBEGIN
            const VIT _val_idx1 = {partial_reduction_val[tid], partial_reduction_idx[tid]};
            const VIT _val_idx2 = {partial_reduction_val[tid + s], partial_reduction_idx[tid + s]};
            partial_reduction_val[tid] = op.value()(_val_idx1, _val_idx2);
            partial_reduction_idx[tid] = op.index()(_val_idx1, _val_idx2);
            // NOLINTEND
        }
        __syncthreads();
    }

    // Warp reduction
    // Since CUDA9, its no longer guranteed that threads in warp are lockstep, need to sync
    // NOLINTBEGIN
    const VIT val_idx1{partial_reduction_val[tid], partial_reduction_idx[tid]};
    const VIT val_idx2{partial_reduction_val[tid + WARP_SIZE], partial_reduction_idx[tid + WARP_SIZE]};
    // NOLINTEND
    VT out_val = op.value()(val_idx1, val_idx2);
    int out_idx = op.index()(val_idx1, val_idx2);
    if (threadIdx.x < WARP_SIZE) {
        for (unsigned int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            const VIT out_val_idx{out_val, out_idx};
            // shfl_down_sync not defined for 8/16 bit int
            // In param will upcast, need to explicitly downcast result
            const VIT curr_val_idx{
                static_cast<VT>(__shfl_down_sync(MASK, out_val, offset)),
                __shfl_down_sync(MASK, out_idx, offset)
            };
            out_idx = op.index()(curr_val_idx, out_val_idx);
            out_val = op.value()(curr_val_idx, out_val_idx);
        }
    }

    // Store grid reduction
    if (tid == 0) {
        res[static_cast<int>(blockIdx.x)] = static_cast<R>(op.res()({out_val, out_idx}));
    }
}

}    // namespace tinytensor::cuda::kernel::reduce

#endif    // TINYTENSOR_BACKEND_CUDA_KERNEL_REDUCE_H_
