// index.cuh
// Indexing kernels

#ifndef TINYTENSOR_BACKEND_CUDA_KERNEL_INDEX_H_
#define TINYTENSOR_BACKEND_CUDA_KERNEL_INDEX_H_

#include <tt/scalar.h>

#include "tensor/backend/common/util.h"
#include "tensor/backend/cuda/config.cuh"
#include "tensor/backend/cuda/data_types.cuh"

#include <cstdint>

namespace tinytensor::cuda::kernel::index {

constexpr int SECTION_SIZE = THREADS_PER_BLOCK;

template <typename T>
__global__ void kernel_mask_small(
    const DataInfo<const T> input_info,
    const DataInfo<const uint8_t> mask_info,
    DataInfo<T> res_info,
    int N
) {
    // Each thread loads its value into shared memory
    __shared__ int PREFIX_SUM[SECTION_SIZE];    // NOLINT(*-avoid-c-arrays)
    int i = GLOBAL_FLAT_THREAD_IDX;
    int mask_idx = to_flat_index(i, mask_info.shape, mask_info.stride, mask_info.offset);

    // NOLINTNEXTLINE(*-array-index)
    PREFIX_SUM[threadIdx.x] = (i < N && mask_info.data[mask_idx] == 0) ? 1 : 0;

    // In increasing stride, accumulate
    for (int s = 1; s < static_cast<int>(blockDim.x); s *= 2) {
        __syncthreads();
        int value = 0;
        if (threadIdx.x + s < SECTION_SIZE) {
            value = PREFIX_SUM[threadIdx.x];    // NOLINT(*-array-index)
        }
        __syncthreads();
        if (threadIdx.x + s < SECTION_SIZE) {
            PREFIX_SUM[threadIdx.x + s] += value;    // NOLINT(*-array-index)
        }
    }
    __syncthreads();

    // PREFIX_SUM indicates how many indices to move left, as long its a masked value
    if (i < N) {
        int input_idx = to_flat_index(i, input_info.shape, input_info.stride, input_info.offset);
        T val = input_info.data[input_idx];
        auto mask = mask_info.data[mask_idx];
        int res_idx = i - PREFIX_SUM[threadIdx.x];    // NOLINT(*-array-index)
        res_idx = to_flat_index(res_idx, res_info.shape, res_info.stride, res_info.offset);
        if (mask > 0) {
            res_info.data[res_idx] = val;
        }
    }
}

// Insert indices of input into res
template <typename T>
__global__ void index_indices_kernel(
    const DataInfo<const T> input_info,
    const DataInfo<const to_ctype_t<kDefaultInt>> indices_info,
    DataInfo<T> res_info,
    int N
) {
    int i = GLOBAL_FLAT_THREAD_IDX;
    if (i < N) {
        int indices_idx = to_flat_index(i, indices_info.shape, indices_info.stride, indices_info.offset);
        auto idx = indices_info.data[indices_idx];
        int input_idx = to_flat_index(idx, input_info.shape, input_info.stride, input_info.offset);
        int res_idx = to_flat_index(i, res_info.shape, res_info.stride, res_info.offset);
        res_info.data[res_idx] = input_info.data[input_idx];
    }
}

template <typename T>
__global__ void kernel_input_put_mask_small(
    DataInfo<T> input_info,
    const DataInfo<const T> values_info,
    const DataInfo<const uint8_t> mask_info,
    int N
) {
    // Each thread loads its value into shared memory
    __shared__ int PREFIX_SUM[SECTION_SIZE];    // NOLINT(*-avoid-c-arrays)
    int i = GLOBAL_FLAT_THREAD_IDX;
    int mask_idx = to_flat_index(i, mask_info.shape, mask_info.stride, mask_info.offset);

    // NOLINTNEXTLINE(*-array-index)
    PREFIX_SUM[threadIdx.x] = (i < N && mask_info.data[mask_idx] == 0) ? 1 : 0;

    // In increasing stride, accumulate
    for (int s = 1; s < static_cast<int>(blockDim.x); s *= 2) {
        __syncthreads();
        int value = 0;
        if (threadIdx.x + s < SECTION_SIZE) {
            value = PREFIX_SUM[threadIdx.x];    // NOLINT(*-array-index)
        }
        __syncthreads();
        if (threadIdx.x + s < SECTION_SIZE) {
            PREFIX_SUM[threadIdx.x + s] += value;    // NOLINT(*-array-index)
        }
    }
    __syncthreads();

    // PREFIX_SUM indicates how many indices to move left, as long its a masked value
    if (i < N) {
        int input_idx = to_flat_index(i, input_info.shape, input_info.stride, input_info.offset);
        auto mask = mask_info.data[mask_idx];
        int value_idx = i - PREFIX_SUM[threadIdx.x];    // NOLINT(*-array-index)
        value_idx = to_flat_index(value_idx, values_info.shape, values_info.stride, values_info.offset);
        if (mask > 0) {
            input_info.data[input_idx] = values_info.data[value_idx];
        }
    }
}

__global__ void kernel_mask_medium_phase1(
    const DataInfo<const uint8_t> mask_info,
    DeviceSpan<int> saved_prefix_sum,
    DeviceSpan<int> scan_block_sum,
    int N
) {
    // Each thread loads its value into shared memory
    __shared__ int PREFIX_SUM[SECTION_SIZE];    // NOLINT(*-avoid-c-arrays)
    int i = GLOBAL_FLAT_THREAD_IDX;
    int mask_idx = to_flat_index(i, mask_info.shape, mask_info.stride, mask_info.offset);

    // NOLINTNEXTLINE(*-array-index)
    PREFIX_SUM[threadIdx.x] = (i < N) && (mask_info.data[mask_idx] == 0) ? 1 : 0;

    // In increasing stride, accumulate
    for (int s = 1; s < static_cast<int>(blockDim.x); s *= 2) {
        __syncthreads();
        int value = 0;
        if (threadIdx.x + s < SECTION_SIZE) {
            value = PREFIX_SUM[threadIdx.x];    // NOLINT(*-array-index)
        }
        __syncthreads();
        if (threadIdx.x + s < SECTION_SIZE) {
            PREFIX_SUM[threadIdx.x + s] += value;    // NOLINT(*-array-index)
        }
    }
    __syncthreads();

    // Save PREFIX_SUM for phase 3
    if (i < N) {
        saved_prefix_sum[i] = PREFIX_SUM[threadIdx.x];    // NOLINT(*-array-index)
    }
    // Store the right-most sum
    if (threadIdx.x == 0) {
        scan_block_sum[static_cast<int>(blockIdx.x)] = PREFIX_SUM[SECTION_SIZE - 1];
    }
}

__global__ void kernel_mask_medium_phase2(DeviceSpan<int> scan_block_sum, int N) {
    // Each thread loads its value into shared memory
    __shared__ int PREFIX_SUM[SECTION_SIZE];    // NOLINT(*-avoid-c-arrays)
    int i = GLOBAL_FLAT_THREAD_IDX;

    PREFIX_SUM[threadIdx.x] = (i < N) ? scan_block_sum[i] : 0;    // NOLINT(*-array-index)

    // In increasing stride, accumulate
    for (int s = 1; s < static_cast<int>(blockDim.x); s *= 2) {
        __syncthreads();
        int value = 0;
        if (threadIdx.x + s < SECTION_SIZE) {
            value = PREFIX_SUM[threadIdx.x];    // NOLINT(*-array-index)
        }
        __syncthreads();
        if (threadIdx.x + s < SECTION_SIZE) {
            PREFIX_SUM[threadIdx.x + s] += value;    // NOLINT(*-array-index)
        }
    }
    __syncthreads();
    if (i < N) {
        scan_block_sum[i] = PREFIX_SUM[threadIdx.x];    // NOLINT(*-array-index)
    }
}

__global__ void kernel_mask_large_phase2(DeviceSpan<int> scan_block_sum1, DeviceSpan<int> scan_block_sum2, int N) {
    // Each thread loads its value into shared memory
    __shared__ int PREFIX_SUM[SECTION_SIZE];    // NOLINT(*-avoid-c-arrays)
    int i = GLOBAL_FLAT_THREAD_IDX;
    PREFIX_SUM[threadIdx.x] = (i < N) ? scan_block_sum1[i] : 0;    // NOLINT(*-array-index)

    // In increasing stride, accumulate
    for (int s = 1; s < static_cast<int>(blockDim.x); s *= 2) {
        __syncthreads();
        int value = 0;
        if (threadIdx.x + s < SECTION_SIZE) {
            value = PREFIX_SUM[threadIdx.x];    // NOLINT(*-array-index)
        }
        __syncthreads();
        if (threadIdx.x + s < SECTION_SIZE) {
            PREFIX_SUM[threadIdx.x + s] += value;    // NOLINT(*-array-index)
        }
    }
    __syncthreads();
    // Save scan block sum
    if (i < N) {
        scan_block_sum1[i] = PREFIX_SUM[threadIdx.x];    // NOLINT(*-array-index)
    }
    // Store right-most into second hierarchy
    if (threadIdx.x == 0) {
        scan_block_sum2[static_cast<int>(blockIdx.x)] = PREFIX_SUM[SECTION_SIZE - 1];
    }
}

// 2nd hierarchy phase 3 of scan block sums to first hierarchy level
__global__ void kernel_mask_large_phase3(DeviceSpan<int> scan_block_sum1, DeviceSpan<int> scan_block_sum2, int N) {
    int i = GLOBAL_FLAT_THREAD_IDX;
    if (i < N) {
        int block_sum = blockIdx.x > 0 ? scan_block_sum2[static_cast<int>(blockIdx.x) - 1] : 0;
        scan_block_sum1[i] += block_sum;
    }
}

template <typename T>
__global__ void kernel_mask_medium_phase3(
    const DataInfo<const T> input_info,
    const DataInfo<const uint8_t> mask_info,
    DeviceSpan<int> saved_prefix_sum,
    DeviceSpan<int> scan_block_sum,
    DataInfo<T> res_info,
    int N
) {
    // Load prefix sum back into shared memory
    __shared__ int PREFIX_SUM[SECTION_SIZE];    // NOLINT(*-avoid-c-arrays)
    int i = GLOBAL_FLAT_THREAD_IDX;
    PREFIX_SUM[threadIdx.x] = (i < N) ? saved_prefix_sum[i] : 0;    // NOLINT(*-array-index)
    __syncthreads();

    // Add left scan block sum to each item in prefix sum
    int block_sum = blockIdx.x > 0 ? scan_block_sum[static_cast<int>(blockIdx.x) - 1] : 0;
    PREFIX_SUM[threadIdx.x] += block_sum;    // NOLINT(*-array-index)
    __syncthreads();

    // PREFIX_SUM indicates how many indices to move left, as long its a masked value
    if (i < N) {
        int input_idx = to_flat_index(i, input_info.shape, input_info.stride, input_info.offset);
        int mask_idx = to_flat_index(i, mask_info.shape, mask_info.stride, mask_info.offset);
        T val = input_info.data[input_idx];
        auto mask = mask_info.data[mask_idx];
        int res_idx = i - PREFIX_SUM[threadIdx.x];    // NOLINT(*-array-index)
        res_idx = to_flat_index(res_idx, res_info.shape, res_info.stride, res_info.offset);
        if (mask > 0) {
            res_info.data[res_idx] = val;
        }
    }
}

template <typename T>
__global__ void kernel_input_put_mask_medium_phase3(
    DataInfo<T> input_info,
    const DataInfo<const T> values_info,
    const DataInfo<const uint8_t> mask_info,
    DeviceSpan<int> saved_prefix_sum,
    DeviceSpan<int> scan_block_sum,
    int N
) {
    // Load prefix sum back into shared memory
    __shared__ int PREFIX_SUM[SECTION_SIZE];    // NOLINT(*-avoid-c-arrays)
    int i = GLOBAL_FLAT_THREAD_IDX;
    PREFIX_SUM[threadIdx.x] = (i < N) ? saved_prefix_sum[i] : 0;    // NOLINT(*-array-index)
    __syncthreads();

    // Add left scan block sum to each item in prefix sum
    int block_sum = blockIdx.x > 0 ? scan_block_sum[static_cast<int>(blockIdx.x) - 1] : 0;
    PREFIX_SUM[threadIdx.x] += block_sum;    // NOLINT(*-array-index)
    __syncthreads();

    // PREFIX_SUM indicates how many indices to move left, as long its a masked value
    if (i < N) {
        int input_idx = to_flat_index(i, input_info.shape, input_info.stride, input_info.offset);
        int mask_idx = to_flat_index(i, mask_info.shape, mask_info.stride, mask_info.offset);
        auto mask = mask_info.data[mask_idx];
        int value_idx = i - PREFIX_SUM[threadIdx.x];    // NOLINT(*-array-index)
        value_idx = to_flat_index(value_idx, values_info.shape, values_info.stride, values_info.offset);
        if (mask > 0) {
            input_info.data[input_idx] = values_info.data[value_idx];
        }
    }
}

// Set indices of input using values
template <typename T>
__global__ void index_put_indices_kernel(
    DataInfo<T> input_info,
    const DataInfo<const T> values_info,
    const DataInfo<const to_ctype_t<kDefaultInt>> indices_info,
    int N
) {
    int i = GLOBAL_FLAT_THREAD_IDX;
    if (i < N) {
        int indices_idx = to_flat_index(i, indices_info.shape, indices_info.stride, indices_info.offset);
        int values_idx = to_flat_index(i, values_info.shape, values_info.stride, values_info.offset);
        auto idx = indices_info.data[indices_idx];
        int input_idx = to_flat_index(idx, input_info.shape, input_info.stride, input_info.offset);
        input_info.data[input_idx] = values_info.data[values_idx];
    }
}

}    // namespace tinytensor::cuda::kernel::index

#endif    // TINYTENSOR_BACKEND_CUDA_KERNEL_INDEX_H_
