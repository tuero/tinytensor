// conv.cuh
// 2D convolution kernels

#ifndef TINYTENSOR_BACKEND_CUDA_KERNEL_CONV_H_
#define TINYTENSOR_BACKEND_CUDA_KERNEL_CONV_H_

#include "tensor/backend/common/reduce.h"
#include "tensor/backend/cuda/config.cuh"
#include "tensor/backend/cuda/data_types.cuh"
#include <tt/util.h>

namespace tinytensor::cuda::kernel::conv {

// Unroll X[b, C_in, H_in, W_in] into a X[C_in * K * K, H_out, W_out] matrix
template <typename T>
__global__ void unroll_kernel(
    const DeviceSpan<const T> X,
    DeviceSpan<T> X_unroll,
    int b,
    int C_in,
    int H_in,
    int W_in,
    int K,
    int S,
    int P
) {
    int H_out = (H_in + 2 * P - K) / S + 1;
    int W_out = (W_in + 2 * P - K) / S + 1;
    int X_unroll_width = H_out * W_out;
    int flat_size_in = C_in * H_in * W_in;

    int t = static_cast<int>(blockIdx.x * THREADS_PER_BLOCK + threadIdx.x);
    if (t < C_in * X_unroll_width) {
        int c = (t / X_unroll_width);
        int X_unroll_r_start = c * K * K;
        int X_unroll_c = (t % X_unroll_width);
        for (int p = 0; p < K; ++p) {
            for (int q = 0; q < K; ++q) {
                int h = X_unroll_c / W_out;
                int w = X_unroll_c % W_out;
                int h_in = S * h + p - P;
                int w_in = S * w + q - P;
                int X_unroll_r_offset = p * K + q;
                int X_unroll_idx = (X_unroll_r_start + X_unroll_r_offset) * X_unroll_width + X_unroll_c;
                int x_idx = b * flat_size_in + c * (H_in * W_in) + (h_in * W_in) + w_in;
                T X_val = (h_in < 0 || w_in < 0 || h_in >= H_in || w_in >= W_in) ? 0 : X[x_idx];
                X_unroll[X_unroll_idx] = X_val;
            }
        }
    }
}

template <typename T>
__global__ void unroll_transpose_kernel(
    const DeviceSpan<const T> X,
    DeviceSpan<T> X_unroll,
    int b,
    int C_in,
    int H_in,
    int W_in,
    int K,
    int S,
    int P
) {
    int H_out = (H_in + 2 * P - K) / S + 1;
    int W_out = (W_in + 2 * P - K) / S + 1;
    int X_unroll_width = H_out * W_out;
    int flat_size_in = C_in * H_in * W_in;

    int t = static_cast<int>(blockIdx.x * THREADS_PER_BLOCK + threadIdx.x);
    if (t < C_in * X_unroll_width) {
        int c = (t / X_unroll_width);
        int X_unroll_r_start = c * K * K;
        int X_unroll_c = (t % X_unroll_width);
        for (int p = 0; p < K; ++p) {
            for (int q = 0; q < K; ++q) {
                int h = X_unroll_c / W_out;
                int w = X_unroll_c % W_out;
                int h_in = S * h + p - P;
                int w_in = S * w + q - P;
                int X_unroll_r_offset = p * K + q;
                int X_unroll_idx = X_unroll_c * (K * K * C_in) + (X_unroll_r_start + X_unroll_r_offset);
                int x_idx = b * flat_size_in + c * (H_in * W_in) + (h_in * W_in) + w_in;
                T X_val = (h_in < 0 || w_in < 0 || h_in >= H_in || w_in >= W_in) ? 0 : X[x_idx];
                X_unroll[X_unroll_idx] = X_val;
            }
        }
    }
}

// @NOTE: Slow, a better implementation is to make each tread an output and local sum
template <typename T>
__global__ void col2im_kernel(
    DeviceSpan<T> X,
    const DeviceSpan<const T> X_unroll,
    int b,
    int C_in,
    int H_in,
    int W_in,
    int K,
    int S,
    int P
) {
    int H_out = (H_in + 2 * P - K) / S + 1;
    int W_out = (W_in + 2 * P - K) / S + 1;
    int X_unroll_width = H_out * W_out;
    int flat_size_in = C_in * H_in * W_in;

    int t = static_cast<int>(blockIdx.x * THREADS_PER_BLOCK + threadIdx.x);
    if (t < C_in * X_unroll_width) {
        int c = (t / X_unroll_width);
        int X_unroll_r_start = c * K * K;
        int X_unroll_c = (t % X_unroll_width);
        for (int p = 0; p < K; ++p) {
            for (int q = 0; q < K; ++q) {
                int h = X_unroll_c / W_out;
                int w = X_unroll_c % W_out;
                int h_in = S * h + p - P;
                int w_in = S * w + q - P;
                int X_unroll_r_offset = p * K + q;
                int X_unroll_idx = (X_unroll_r_start + X_unroll_r_offset) * X_unroll_width + X_unroll_c;
                int x_idx = b * flat_size_in + c * (H_in * W_in) + (h_in * W_in) + w_in;
                if (!(h_in < 0 || w_in < 0 || h_in >= H_in || w_in >= W_in)) {
                    X[x_idx] += X_unroll[X_unroll_idx];
                }
                __syncthreads();
            }
        }
    }
}

constexpr int TW = WARP_SIZE;

template <typename T, typename R, typename OP>
__global__ void
    pool_kernel(const DeviceSpan<const T> X, DeviceSpan<R> res, OP op, int C, int H_in, int W_in, int K, int S, int P) {
    const int H_out = (H_in + 2 * P - K) / S + 1;
    const int W_out = (W_in + 2 * P - K) / S + 1;
    int W_grid = ceil_div(W_out, TW);
    int b = static_cast<int>(blockIdx.x);
    int c = static_cast<int>(blockIdx.y);
    int h = (static_cast<int>(blockIdx.z) / W_grid) * TW + static_cast<int>(threadIdx.y);    // h,w in output
    int w = (static_cast<int>(blockIdx.z) % W_grid) * TW + static_cast<int>(threadIdx.x);
    const int flat_size_in = C * H_in * W_in;
    const int flat_size_out = C * H_out * W_out;
    const int idx_res = b * flat_size_out + c * (H_out * W_out) + h * (W_out) + w;

    if (h < H_out && w < W_out) {
        auto out_val = OP::padding_value;
        for (int p = 0; p < K; ++p) {
            for (int q = 0; q < K; ++q) {
                int h_in = S * h + p - P;
                int w_in = S * w + q - P;
                int idx_in = b * flat_size_in + c * (H_in * W_in) + (h_in * W_in) + w_in;
                R X_val = (h_in < 0 || w_in < 0 || h_in >= H_in || w_in >= W_in) ? OP::padding_value
                                                                                 : static_cast<R>(X[idx_in]);
                out_val = op()(X_val, out_val);
            }
        }
        res[idx_res] = out_val;
    }
}

template <typename R>
using MinOp = typename common::reduce::OpFactory<R, common::reduce::ReduceOpT::min>::KernelOp;
template <typename R>
using MaxOp = typename common::reduce::OpFactory<R, common::reduce::ReduceOpT::max>::KernelOp;
template <typename R>
using MeanOp = typename common::reduce::OpFactory<R, common::reduce::ReduceOpT::mean>::KernelOp;

template <typename T, typename R>
__global__ void pool_backward_min_max_kernel(
    const DeviceSpan<const R> grad_output,
    DeviceSpan<R> grad_input,
    const DeviceSpan<const T> X,
    const DeviceSpan<const R> res,
    int C,
    int H_in,
    int W_in,
    int K,
    int S,
    int P
) {
    const int H_out = (H_in + 2 * P - K) / S + 1;
    const int W_out = (W_in + 2 * P - K) / S + 1;
    int W_grid = ceil_div(W_out, TW);
    int b = static_cast<int>(blockIdx.x);
    int c = static_cast<int>(blockIdx.y);
    int h = (static_cast<int>(blockIdx.z) / W_grid) * TW + static_cast<int>(threadIdx.y);    // h,w in output
    int w = (static_cast<int>(blockIdx.z) % W_grid) * TW + static_cast<int>(threadIdx.x);
    const int flat_size_in = C * H_in * W_in;
    const int flat_size_out = C * H_out * W_out;
    const int idx_res = b * flat_size_out + c * (H_out * W_out) + h * (W_out) + w;

    if (h < H_out && w < W_out) {
        // Find how many elements match min
        R N = 0;
        R min_value = res[idx_res];
        for (int p = 0; p < K; ++p) {
            for (int q = 0; q < K; ++q) {
                int h_in = S * h + p - P;
                int w_in = S * w + q - P;
                int idx_in = b * flat_size_in + c * (H_in * W_in) + (h_in * W_in) + w_in;
                bool in_bounds = !(h_in < 0 || w_in < 0 || h_in >= H_in || w_in >= W_in);
                N += (in_bounds && static_cast<R>(X[idx_in]) == min_value) ? static_cast<R>(1) : static_cast<R>(0);
            }
        }
        // Redistribute gradient back to matching elements
        R grad_in = grad_output[idx_res] / N;
        for (int p = 0; p < K; ++p) {
            for (int q = 0; q < K; ++q) {
                int h_in = S * h + p - P;
                int w_in = S * w + q - P;
                int idx_in = b * flat_size_in + c * (H_in * W_in) + (h_in * W_in) + w_in;
                bool in_bounds = !(h_in < 0 || w_in < 0 || h_in >= H_in || w_in >= W_in);
                if (in_bounds && static_cast<R>(X[idx_in]) == min_value) {
                    grad_input[idx_in] += grad_in;
                }
                __syncthreads();
            }
        }
    }
}

template <typename T, typename R>
__global__ void pool_backward_mean_kernel(
    const DeviceSpan<const R> grad_output,
    DeviceSpan<R> grad_input,
    [[maybe_unused]] const DeviceSpan<const T> X,
    [[maybe_unused]] const DeviceSpan<const R> res,
    int C,
    int H_in,
    int W_in,
    int K,
    int S,
    int P
) {
    const int H_out = (H_in + 2 * P - K) / S + 1;
    const int W_out = (W_in + 2 * P - K) / S + 1;
    int W_grid = ceil_div(W_out, TW);
    int b = static_cast<int>(blockIdx.x);
    int c = static_cast<int>(blockIdx.y);
    int h = (static_cast<int>(blockIdx.z) / W_grid) * TW + static_cast<int>(threadIdx.y);    // h,w in output
    int w = (static_cast<int>(blockIdx.z) % W_grid) * TW + static_cast<int>(threadIdx.x);
    const int flat_size_in = C * H_in * W_in;
    const int flat_size_out = C * H_out * W_out;
    const int idx_res = b * flat_size_out + c * (H_out * W_out) + h * (W_out) + w;

    if (h < H_out && w < W_out) {
        // Redistribute gradient back to elements
        R grad_in = grad_output[idx_res] / static_cast<R>(K * K);
        for (int p = 0; p < K; ++p) {
            for (int q = 0; q < K; ++q) {
                int h_in = S * h + p - P;
                int w_in = S * w + q - P;
                int idx_in = b * flat_size_in + c * (H_in * W_in) + (h_in * W_in) + w_in;
                bool in_bounds = !(h_in < 0 || w_in < 0 || h_in >= H_in || w_in >= W_in);
                if (in_bounds) {
                    grad_input[idx_in] += grad_in;
                }
                __syncthreads();
            }
        }
    }
}

template <typename>
constexpr bool dependent_false_v = false;

}    // namespace tinytensor::cuda::kernel::conv

#endif    // TINYTENSOR_BACKEND_CUDA_KERNEL_CONV_H_
