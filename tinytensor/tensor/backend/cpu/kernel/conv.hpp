// conv.hpp
// 2D convolution kernels

#ifndef TINYTENSOR_BACKEND_CPU_KERNEL_CONV_H_
#define TINYTENSOR_BACKEND_CPU_KERNEL_CONV_H_

#include "tensor/backend/common/reduce.h"
#include "tensor/backend/common/span.h"

namespace tinytensor::cpu::kernel::conv {

template <typename T>
void unroll_kernel(
    const HostSpan<const T> X,
    HostSpan<T> X_unroll,
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
    for (int c = 0; c < C_in; ++c) {
        int X_unroll_r_start = c * K * K;
        for (int p = 0; p < K; ++p) {
            for (int q = 0; q < K; ++q) {
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        int h_in = S * h + p - P;
                        int w_in = S * w + q - P;
                        int X_unroll_r_offset = p * K + q;
                        int X_unroll_idx = (X_unroll_r_start + X_unroll_r_offset) * X_unroll_width + (h * W_out + w);
                        int x_idx = b * flat_size_in + c * (H_in * W_in) + (h_in * W_in) + w_in;
                        T X_val = (h_in < 0 || w_in < 0 || h_in >= H_in || w_in >= W_in) ? 0 : X[x_idx];
                        X_unroll[X_unroll_idx] = X_val;
                    }
                }
            }
        }
    }
}

template <typename T>
void unroll_transpose_kernel(
    const HostSpan<const T> X,
    HostSpan<T> X_unroll,
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
    int flat_size_in = C_in * H_in * W_in;
    for (int c = 0; c < C_in; ++c) {
        int X_unroll_c_start = c * K * K;
        for (int p = 0; p < K; ++p) {
            for (int q = 0; q < K; ++q) {
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        int h_in = S * h + p - P;
                        int w_in = S * w + q - P;
                        int X_unroll_c_offset = p * K + q;
                        int X_unroll_idx = (h * W_out + w) * (K * K * C_in) + (X_unroll_c_start + X_unroll_c_offset);
                        int x_idx = b * flat_size_in + c * (H_in * W_in) + (h_in * W_in) + w_in;
                        T X_val = (h_in < 0 || w_in < 0 || h_in >= H_in || w_in >= W_in) ? 0 : X[x_idx];
                        X_unroll[X_unroll_idx] = X_val;
                    }
                }
            }
        }
    }
}

template <typename T>
void col2im_kernel(
    HostSpan<T> X,
    const HostSpan<const T> X_unroll,
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
    for (int c = 0; c < C_in; ++c) {
        int X_unroll_r_start = c * K * K;
        for (int p = 0; p < K; ++p) {
            for (int q = 0; q < K; ++q) {
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        int h_in = S * h + p - P;
                        int w_in = S * w + q - P;
                        int X_unroll_r_offset = p * K + q;
                        int X_unroll_idx = (X_unroll_r_start + X_unroll_r_offset) * X_unroll_width + (h * W_out + w);
                        int x_idx = b * flat_size_in + c * (H_in * W_in) + (h_in * W_in) + w_in;
                        if (!(h_in < 0 || w_in < 0 || h_in >= H_in || w_in >= W_in)) {
                            X[x_idx] += X_unroll[X_unroll_idx];
                        }
                    }
                }
            }
        }
    }
}

template <typename T, typename R, typename OP>
void pool_kernel(
    const HostSpan<const T> X,
    HostSpan<R> res,
    OP op,
    int B,
    int C,
    int H_in,
    int W_in,
    int K,
    int S,
    int P
) {
    const int H_out = (H_in + 2 * P - K) / S + 1;
    const int W_out = (W_in + 2 * P - K) / S + 1;

    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            const int flat_size_in = C * H_in * W_in;
            const int flat_size_out = C * H_out * W_out;
            for (int h = 0; h < H_out; ++h) {
                for (int w = 0; w < W_out; ++w) {
                    const int idx_res = b * flat_size_out + c * (H_out * W_out) + h * (W_out) + w;
                    auto out_val = OP::padding_value;
                    for (int p = 0; p < K; ++p) {
                        for (int q = 0; q < K; ++q) {
                            int h_in = S * h + p - P;
                            int w_in = S * w + q - P;
                            int idx_in = b * flat_size_in + c * (H_in * W_in) + (h_in * W_in) + w_in;
                            R X_val = (h_in < 0 || w_in < 0 || h_in >= H_in || w_in >= W_in)
                                          ? OP::padding_value
                                          : static_cast<R>(X[idx_in]);
                            out_val = op()(X_val, out_val);
                        }
                    }
                    res[idx_res] = out_val;
                }
            }
        }
    }
}

template <typename R>
using MinOp = typename common::reduce::OpFactory<R, common::reduce::ReduceOpT::min>::KernelOp;
template <typename R>
using MaxOp = typename common::reduce::OpFactory<R, common::reduce::ReduceOpT::max>::KernelOp;
template <typename R>
using MeanOp = typename common::reduce::OpFactory<R, common::reduce::ReduceOpT::mean>::KernelOp;

template <typename T, typename R, typename OP>
    requires(std::is_same_v<OP, MinOp<R>> || std::is_same_v<OP, MaxOp<R>>)
void pool_backward_kernel(
    const HostSpan<const R> grad_output,
    HostSpan<R> grad_input,
    const HostSpan<const T> X,
    const HostSpan<const R> res,
    int B,
    int C,
    int H_in,
    int W_in,
    int K,
    int S,
    int P
) {
    const int H_out = (H_in + 2 * P - K) / S + 1;
    const int W_out = (W_in + 2 * P - K) / S + 1;

    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            const int flat_size_in = C * H_in * W_in;
            const int flat_size_out = C * H_out * W_out;
            for (int h = 0; h < H_out; ++h) {
                for (int w = 0; w < W_out; ++w) {
                    const int idx_res = b * flat_size_out + c * (H_out * W_out) + h * (W_out) + w;
                    // Find how many elements match min
                    R N = 0;
                    R min_value = res[idx_res];
                    for (int p = 0; p < K; ++p) {
                        for (int q = 0; q < K; ++q) {
                            int h_in = S * h + p - P;
                            int w_in = S * w + q - P;
                            int idx_in = b * flat_size_in + c * (H_in * W_in) + (h_in * W_in) + w_in;
                            bool in_bounds = !(h_in < 0 || w_in < 0 || h_in >= H_in || w_in >= W_in);
                            N += (in_bounds && static_cast<R>(X[idx_in]) == min_value) ? static_cast<R>(1)
                                                                                       : static_cast<R>(0);
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
                        }
                    }
                }
            }
        }
    }
}

template <typename T, typename R, typename OP>
    requires std::is_same_v<OP, MeanOp<R>>
void pool_backward_kernel(
    const HostSpan<const R> grad_output,
    HostSpan<R> grad_input,
    [[maybe_unused]] const HostSpan<const T> X,
    [[maybe_unused]] const HostSpan<const R> res,
    int B,
    int C,
    int H_in,
    int W_in,
    int K,
    int S,
    int P
) {
    const int H_out = (H_in + 2 * P - K) / S + 1;
    const int W_out = (W_in + 2 * P - K) / S + 1;

    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            const int flat_size_in = C * H_in * W_in;
            const int flat_size_out = C * H_out * W_out;
            for (int h = 0; h < H_out; ++h) {
                for (int w = 0; w < W_out; ++w) {
                    const int idx_res = b * flat_size_out + c * (H_out * W_out) + h * (W_out) + w;
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
                        }
                    }
                }
            }
        }
    }
}

}    // namespace tinytensor::cpu::kernel::conv

#endif    // TINYTENSOR_BACKEND_CPU_KERNEL_CONV_H_
