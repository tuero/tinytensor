// test_conv.cpp
// Test Conv methods

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <tt/device.h>
#include <tt/exception.h>
#include <tt/scalar.h>
#include <tt/shape.h>
#include <tt/tensor.h>

#ifdef TT_TORCH
#include <torch/torch.h>
#undef CHECK
#endif

#include "doctest.h"
#include "test_util.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

using namespace tinytensor;

namespace {

template <typename T>
std::vector<T> conv_forward(
    std::vector<T> &X,
    std::vector<T> &W,
    int B,
    int C_in,
    int C_out,
    int H_in,
    int W_in,
    int K,
    int S,
    int P
) {
    int H_out = (H_in + 2 * P - K) / S + 1;
    int W_out = (W_in + 2 * P - K) / S + 1;
    int flat_size_in = C_in * H_in * W_in;
    int flat_size_out = C_out * H_out * W_out;
    std::vector<T> Y(static_cast<std::size_t>(B * C_out * H_out * W_out), 0);
    for (int b = 0; b < B; ++b) {
        for (int m = 0; m < C_out; ++m) {
            for (int h = 0; h < H_out; ++h) {
                for (int w = 0; w < W_out; ++w) {
                    //
                    for (int c = 0; c < C_in; ++c) {
                        for (int p = 0; p < K; ++p) {
                            for (int q = 0; q < K; ++q) {
                                int h_in = S * h + p - P;
                                int w_in = S * w + q - P;
                                int idx_out = b * flat_size_out + m * (H_out * W_out) + (h * W_out) + w;
                                int idx_in = b * flat_size_in + c * (H_in * W_in) + (h_in * W_in) + w_in;
                                int idx_w = (m * C_in * K * K) + (c * K * K) + (p * K) + q;
                                T X_val = (h_in < 0 || w_in < 0 || h_in >= H_in || w_in >= W_in)
                                              ? 0
                                              : X[static_cast<std::size_t>(idx_in)];
                                Y[static_cast<std::size_t>(idx_out)] +=
                                    static_cast<T>(X_val * W[static_cast<std::size_t>(idx_w)]);
                            }
                        }
                    }
                }
            }
        }
    }
    return Y;
}

template <typename T>
std::vector<T> init_vec(int size) {
    std::vector<T> res;
    res.reserve(static_cast<std::size_t>(size));
    for (int i = 0; i < size; ++i) {
        res.push_back(static_cast<T>(i % 10));
    }
    return res;
}

template <typename T>
std::vector<T> rand_vec(int size, std::mt19937 &gen) {
    std::vector<T> res;
    res.reserve(static_cast<std::size_t>(size));
    std::uniform_int_distribution<int> dis(-5.0, 5.0);
    for (int i = 0; i < size; ++i) {
        res.push_back(static_cast<T>(dis(gen)));
    }
    return res;
}

}    // namespace

// NOLINTNEXTLINE
TEST_CASE("Conv2D Forward") {
    auto test1 = []<typename T>(Device device) {
        const int B = 4;
        const int C_in = 3;
        const int C_out = 128;
        const int H_in = 64;
        const int W_in = 64;

        const int K = 3;
        const int S = 1;
        const int P = 0;
        const int H_out = (H_in + 2 * P - K) / S + 1;
        const int W_out = (W_in + 2 * P - K) / S + 1;
        std::vector<T> _X = init_vec<T>(B * C_in * H_in * W_in);
        std::vector<T> _W = init_vec<T>(C_out * C_in * K * K);

        Tensor X(_X, {B, C_in, H_in, W_in}, device);
        Tensor W(_W, {C_out, C_in, K, K}, device);
        Tensor expected(conv_forward(_X, _W, B, C_in, C_out, H_in, W_in, K, S, P), {B, C_out, H_out, W_out}, device);
        CHECK(allclose(tinytensor::conv2d(X, W, {}, S, P), expected));
    };
    runner_all_except_bool(test1);

    auto test2 = []<typename T>(Device device) {
        const int B = 4;
        const int C_in = 3;
        const int C_out = 128;
        const int H_in = 64;
        const int W_in = 64;

        const int K = 3;
        const int S = 2;
        const int P = 1;
        const int H_out = (H_in + 2 * P - K) / S + 1;
        const int W_out = (W_in + 2 * P - K) / S + 1;
        std::vector<T> _X = init_vec<T>(B * C_in * H_in * W_in);
        std::vector<T> _W = init_vec<T>(C_out * C_in * K * K);

        Tensor X(_X, {B, C_in, H_in, W_in}, device);
        Tensor W(_W, {C_out, C_in, K, K}, device);
        Tensor expected(conv_forward(_X, _W, B, C_in, C_out, H_in, W_in, K, S, P), {B, C_out, H_out, W_out}, device);
        CHECK(allclose(tinytensor::conv2d(X, W, {}, S, P), expected));
    };
    runner_all_except_bool(test2);
}

#ifdef TT_TORCH
// NOLINTNEXTLINE
TEST_CASE("Conv2D Forward Torch") {
    auto test1 = []<typename T>(Device device) {
        const int B = 4;
        const int C_in = 3;
        const int C_out = 128;
        const int H_in = 68;
        const int W_in = 28;

        const int K = 3;
        const int S = 2;
        const int P = 1;
        const int H_out = (H_in + 2 * P - K) / S + 1;
        const int W_out = (W_in + 2 * P - K) / S + 1;
        std::mt19937 gen(0);
        std::vector<T> _X = rand_vec<T>(B * C_in * H_in * W_in, gen);
        std::vector<T> _W = rand_vec<T>(C_out * C_in * K * K, gen);
        std::vector<T> _BIAS = rand_vec<T>(C_out, gen);

        Tensor X(_X, {B, C_in, H_in, W_in}, device);
        Tensor W(_W, {C_out, C_in, K, K}, device);
        Tensor BIAS(_BIAS, {C_out}, device);

        const auto options_float = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor X_torch = torch::from_blob(_X.data(), {B, C_in, H_in, W_in}, options_float);
        torch::Tensor W_torch = torch::from_blob(_W.data(), {C_out, C_in, K, K}, options_float);
        torch::Tensor B_torch = torch::from_blob(_BIAS.data(), {C_out}, options_float);
        torch::Tensor expected_torch = torch::conv2d(X_torch, W_torch, B_torch, S, P);
        std::vector<T> expected_torch_data(
            expected_torch.data_ptr<T>(),
            expected_torch.data_ptr<T>() + expected_torch.numel()
        );

        Tensor expected(expected_torch_data, {B, C_out, H_out, W_out}, device);
        CHECK(allclose(tinytensor::conv2d(X, W, BIAS, S, P), expected, CloseOptions().atol(1e-6)));
    };
    runner_single_type<float>(test1);
}

// NOLINTNEXTLINE
TEST_CASE("Max Pool2D Forward Torch") {
    auto test1 = []<typename T>(Device device) {
        const int B = 4;
        const int C = 128;
        const int H_in = 68;
        const int W_in = 28;

        const int K = 3;
        const int S = 2;
        const int P = 1;
        const int H_out = (H_in + 2 * P - K) / S + 1;
        const int W_out = (W_in + 2 * P - K) / S + 1;
        std::mt19937 gen(0);
        std::vector<T> _X = rand_vec<T>(B * C * H_in * W_in, gen);

        Tensor X(_X, {B, C, H_in, W_in}, device);

        const auto options_float = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor X_torch = torch::from_blob(_X.data(), {B, C, H_in, W_in}, options_float);
        torch::Tensor expected_torch = torch::max_pool2d(X_torch, K, S, P);
        std::vector<T> expected_torch_data(
            expected_torch.data_ptr<T>(),
            expected_torch.data_ptr<T>() + expected_torch.numel()
        );

        Tensor expected(expected_torch_data, {B, C, H_out, W_out}, device);
        CHECK(allclose(tinytensor::max_pool2d(X, K, S, P), expected));
    };
    runner_single_type<float>(test1);
}

// NOLINTNEXTLINE
TEST_CASE("Min Pool2D Forward Torch") {
    auto test1 = []<typename T>(Device device) {
        const int B = 4;
        const int C = 128;
        const int H_in = 68;
        const int W_in = 28;

        const int K = 3;
        const int S = 2;
        const int P = 1;
        const int H_out = (H_in + 2 * P - K) / S + 1;
        const int W_out = (W_in + 2 * P - K) / S + 1;
        std::mt19937 gen(0);
        std::vector<T> _X = rand_vec<T>(B * C * H_in * W_in, gen);

        Tensor X(_X, {B, C, H_in, W_in}, device);

        const auto options_float = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor X_torch = torch::from_blob(_X.data(), {B, C, H_in, W_in}, options_float);
        torch::Tensor expected_torch = -torch::max_pool2d(-X_torch, K, S, P);
        std::vector<T> expected_torch_data(
            expected_torch.data_ptr<T>(),
            expected_torch.data_ptr<T>() + expected_torch.numel()
        );

        Tensor expected(expected_torch_data, {B, C, H_out, W_out}, device);
        CHECK(allclose(tinytensor::min_pool2d(X, K, S, P), expected));
    };
    runner_single_type<float>(test1);
}

// NOLINTNEXTLINE
TEST_CASE("Avg Pool2D Forward Torch") {
    auto test1 = []<typename T>(Device device) {
        const int B = 4;
        const int C = 128;
        const int H_in = 68;
        const int W_in = 28;

        const int K = 3;
        const int S = 2;
        const int P = 1;
        const int H_out = (H_in + 2 * P - K) / S + 1;
        const int W_out = (W_in + 2 * P - K) / S + 1;
        std::mt19937 gen(0);
        std::vector<T> _X = rand_vec<T>(B * C * H_in * W_in, gen);

        Tensor X(_X, {B, C, H_in, W_in}, device);

        const auto options_float = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor X_torch = torch::from_blob(_X.data(), {B, C, H_in, W_in}, options_float);
        torch::Tensor expected_torch = torch::avg_pool2d(X_torch, K, S, P);
        std::vector<T> expected_torch_data(
            expected_torch.data_ptr<T>(),
            expected_torch.data_ptr<T>() + expected_torch.numel()
        );

        Tensor expected(expected_torch_data, {B, C, H_out, W_out}, device);
        CHECK(allclose(tinytensor::avg_pool2d(X, K, S, P), expected, CloseOptions().atol(1e-6)));
    };
    runner_single_type<float>(test1);
}

#endif
