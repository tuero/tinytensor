// matmul.cuh
// Matrix multiplication kernel

#ifndef TINYTENSOR_BACKEND_CUDA_KERNEL_MATMUL_H_
#define TINYTENSOR_BACKEND_CUDA_KERNEL_MATMUL_H_

#include <tt/scalar.h>

#include "tensor/backend/cuda/data_types.cuh"

#include <cassert>

namespace tinytensor::cuda::kernel::matmul {

// Kernel properties
namespace properties {
constexpr int TILE_WIDTH = 64;     // Width of tile each grid block will compute
constexpr int TILE_HEIGHT = 64;    // Height of tile each grid block will compute
constexpr int TILE_STRIDE = 8;     // Stride along K dimension the tile progresses when computing partial results
constexpr int TN = 8;              // Width of tile each thread will compute
constexpr int TM = 8;              // Height of tile each thread will compute
}    // namespace properties

using namespace properties;

template <typename T>
__global__ void
    matmul_kernel(const DeviceSpan<const T> A, const DeviceSpan<const T> B, DeviceSpan<T> C, int N, int K, int M) {
    const auto threads_per_block_tile = ((TILE_HEIGHT * TILE_WIDTH) / (TN * TM));
    assert(blockDim.x == threads_per_block_tile);

    // Thread row/coumn which maps into local block tile
    // (i.e. these are not global thread/cols), since we offset C pointer
    // Threads no longer span the entire TILE_WIDTH of the B block,
    // Each thread is responsible for TM entries, so the "width" is divided by TM
    const auto thread_row = (threadIdx.x / (TILE_WIDTH / TM));
    const auto thread_col = (threadIdx.x % (TILE_WIDTH / TM));

    // Shared buffer for current tile block of A and B
    __shared__ T A_block[TILE_HEIGHT * TILE_STRIDE];    // NOLINT(*-c-arrays)
    __shared__ T B_block[TILE_STRIDE * TILE_WIDTH];     // NOLINT(*-c-arrays)

    // starting row and column of C we will write into
    const auto c_row = blockIdx.y * TILE_HEIGHT;
    const auto c_col = blockIdx.x * TILE_WIDTH;

    // Set A, B, and C to point to the block tile we are curently tiled over
    // A starts at C's row, column 0 (which will be summed over)
    // B starts at C's column at row 0 (which will be summed over)
    // C starts at its row and column previously calculated
    auto a_row = c_row;
    decltype(a_row) a_col = 0;
    auto b_col = c_col;
    decltype(b_col) b_row = 0;

    // Each thread needs to load multiply entries from A and B
    // We want adjacent threads to load adjacently along the same row in A
    // Loads are done in A [strideA x TILE_STRIDE] tile
    // Loads are done in B [TILE_STRIDE x TILE_WIDTH] tile
    const int strideA = threads_per_block_tile / TILE_STRIDE;
    const int strideB = threads_per_block_tile / TILE_WIDTH;

    // Each thread is responsible for a row/col inside the inner tile of A and B for loading
    const auto innerRowA = threadIdx.x / TILE_STRIDE;
    const auto innerColA = threadIdx.x % TILE_STRIDE;
    const auto innerRowB = threadIdx.x / TILE_WIDTH;
    const auto innerColB = threadIdx.x % TILE_WIDTH;

    // Each thread will compute a TN * TM block of sums
    T results[TN * TM] = {0};    // NOLINT(*-c-arrays)

    // We load and reuse entries from the shared memory block of A and B
    // So we store in a register
    T cached_A[TN] = {0};    // NOLINT(*-c-arrays)
    T cached_B[TM] = {0};    // NOLINT(*-c-arrays)

    // Outer loop walks length of K, but in TILE_STRIDE strides
    for (int block_idx = 0; block_idx < K; block_idx += TILE_STRIDE) {
        // Each thread needs to load multiple elements from A/B
        // We load a block of size [stride A x TILE_STRIDE] for A
        for (int load_offset = 0; load_offset < TILE_HEIGHT; load_offset += strideA) {
            const auto local_row = innerRowA + load_offset;
            const auto row = static_cast<int>(a_row + innerRowA + load_offset);
            const auto col = static_cast<int>(a_col + innerColA);
            const bool in_range = row < N && col < K;
            const auto idx_A = static_cast<int>((blockIdx.z * N * K) + row * K + col);
            // NOLINTNEXTLINE(*-array-index)
            A_block[local_row * TILE_STRIDE + innerColA] = in_range ? A[idx_A] : 0;
        }
        // We load a block of size [stride A x TILE_STRIDE] for A
        for (int load_offset = 0; load_offset < TILE_STRIDE; load_offset += strideB) {
            const auto local_row = innerRowB + load_offset;
            const auto row = static_cast<int>(b_row + innerRowB + load_offset);
            const auto col = static_cast<int>(b_col + innerColB);
            const bool in_range = row < K && col < M;
            const auto idx_B = static_cast<int>((blockIdx.z * K * M) + row * M + col);
            // NOLINTNEXTLINE(*-array-index)
            B_block[local_row * TILE_WIDTH + innerColB] = in_range ? B[idx_B] : 0;
        }

        // Wait for all threads to load data into the cache
        __syncthreads();

        // Increment A by walking horizontally along K
        // Increment B by walking vertically along K
        a_col += TILE_STRIDE;
        b_row += TILE_STRIDE;

        // Each thread computes a 2d block of C using the cached results in tmpA/tmpB
        for (int dot_idx = 0; dot_idx < TILE_STRIDE; ++dot_idx) {
            // Load TN + TM results into registers first
            for (int i = 0; i < TN; ++i) {
                const auto row_idx = thread_row * TN + i;
                cached_A[i] = A_block[row_idx * TILE_STRIDE + dot_idx];    // NOLINT(*-array-index)
            }
            for (int i = 0; i < TM; ++i) {
                // NOLINTNEXTLINE(*-array-index)
                cached_B[i] = B_block[dot_idx * TILE_WIDTH + thread_col * TM + i];
            }

            // Compute TN * TM results using the cached results
            for (int i = 0; i < TN; ++i) {
                for (int j = 0; j < TM; ++j) {
                    results[i * TM + j] += cached_A[i] * cached_B[j];    // NOLINT(*-array-index)
                }
            }
        }

        // Ensure threads finished with current cache before we set result then increment to next
        // cache tile
        __syncthreads();
    }
    // Write the TN * TM results for this thread back to C
    for (int i = 0; i < TN; ++i) {
        for (int j = 0; j < TM; ++j) {
            const auto row = static_cast<int>(c_row + (thread_row * TN + i));
            const auto col = static_cast<int>(c_col + (thread_col * TM + j));
            if (row < N && col < M) {
                const auto idx_C = static_cast<int>((blockIdx.z * N * M) + row * M + col);
                C[idx_C] = results[i * TM + j];    // NOLINT(*-array-index)
            }
        }
    }
}

}    // namespace tinytensor::cuda::kernel::matmul

#endif    // TINYTENSOR_BACKEND_CUDA_KERNEL_MATMUL_H_
