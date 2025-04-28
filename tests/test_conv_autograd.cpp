// test_conv_autograd.cpp
// Test the conv2d autograd ops
// We use double precision due to underlying computation being lossy

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <tt/device.h>
#include <tt/exception.h>
#include <tt/scalar.h>
#include <tt/tensor.h>

#include <algorithm>
#include <cstddef>
#include <random>
#include <vector>

#ifdef TT_TORCH
#include <torch/torch.h>
#undef CHECK
#endif

#include "doctest.h"
#include "test_util.h"

using namespace tinytensor;

namespace {
template <typename T>
std::vector<T> rand_vec(int size, std::mt19937 &gen) {
    std::vector<T> res;
    res.reserve(static_cast<std::size_t>(size));
    std::uniform_real_distribution<double> dis(-1.0, 1.0);
    for (int i = 0; i < size; ++i) {
        res.push_back(static_cast<T>(dis(gen)));
    }
    return res;
}

}    // namespace

#ifdef TT_TORCH

// NOLINTNEXTLINE
TEST_CASE("Conv2D autograd") {
    auto test1 = []<typename T>(Device device) {
        at::globalContext().setUserEnabledCuDNN(false);
        const int B = 4;
        const int C_in = 3;
        const int C_out = 64;
        const int H_in = 38;
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
        std::vector<T> _grad = rand_vec<T>(B * C_out * H_out * W_out, gen);

        Tensor X(_X, {B, C_in, H_in, W_in}, device, true);
        Tensor W(_W, {C_out, C_in, K, K}, device, true);
        Tensor BIAS(_BIAS, {C_out}, device, true);
        Tensor grad(_grad, {B, C_out, H_out, W_out}, device);
        Tensor out = tinytensor::conv2d(X, W, BIAS, S, P);
        out.backward(grad);

        const auto options_float = torch::TensorOptions().dtype(torch::kFloat64);
        torch::Tensor X_torch = torch::from_blob(_X.data(), {B, C_in, H_in, W_in}, options_float.requires_grad(true));
        torch::Tensor W_torch = torch::from_blob(_W.data(), {C_out, C_in, K, K}, options_float.requires_grad(true));
        torch::Tensor B_torch = torch::from_blob(_BIAS.data(), {C_out}, options_float.requires_grad(true));
        torch::Tensor grad_torch = torch::from_blob(_grad.data(), {B, C_out, H_out, W_out}, options_float);
        torch::Tensor out_torch = torch::conv2d(X_torch, W_torch, B_torch, S, P);
        out_torch.backward(grad_torch);

        torch::Tensor expected_grad_x = X_torch.grad();
        torch::Tensor expected_grad_w = W_torch.grad();
        torch::Tensor expected_grad_b = B_torch.grad();
        std::vector<T> expected_grad_x_data(
            expected_grad_x.data_ptr<T>(),
            expected_grad_x.data_ptr<T>() + expected_grad_x.numel()
        );
        std::vector<T> expected_grad_w_data(
            expected_grad_w.data_ptr<T>(),
            expected_grad_w.data_ptr<T>() + expected_grad_w.numel()
        );
        std::vector<T> expected_grad_b_data(
            expected_grad_b.data_ptr<T>(),
            expected_grad_b.data_ptr<T>() + expected_grad_b.numel()
        );
        Tensor expected_x(expected_grad_x_data, X.shape(), device);
        Tensor expected_w(expected_grad_w_data, W.shape(), device);
        Tensor expected_b(expected_grad_b_data, BIAS.shape(), device);

        // mean difference instead of absolute element-wise difference, as cuda implementations can have a few elements
        // off depending on how its implemented
        auto diff = abs(X.grad().value() - expected_x);
        auto close_options = CloseOptions().atol(1e-4).equal_nan();
        CHECK(all(diff.mean() < 0.001));
        CHECK(allclose(W.grad().value(), expected_w, close_options));
        CHECK(allclose(BIAS.grad().value(), expected_b, close_options));
    };
    runner_single_type<double>(test1);
}

// NOLINTNEXTLINE
TEST_CASE("Max pool autograd") {
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
        std::vector<T> _grad = rand_vec<T>(B * C * H_out * W_out, gen);

        Tensor X(_X, {B, C, H_in, W_in}, device, true);
        Tensor grad(_grad, {B, C, H_out, W_out}, device);
        Tensor out = tinytensor::max_pool2d(X, K, S, P);
        out.backward(grad);

        const auto options_float = torch::TensorOptions().dtype(torch::kFloat64);
        torch::Tensor X_torch = torch::from_blob(_X.data(), {B, C, H_in, W_in}, options_float.requires_grad(true));
        torch::Tensor grad_torch = torch::from_blob(_grad.data(), {B, C, H_out, W_out}, options_float);
        torch::Tensor out_torch = torch::max_pool2d(X_torch, K, S, P);
        out_torch.backward(grad_torch);

        torch::Tensor expected_grad_x = X_torch.grad();
        std::vector<T> expected_grad_x_data(
            expected_grad_x.data_ptr<T>(),
            expected_grad_x.data_ptr<T>() + expected_grad_x.numel()
        );
        Tensor expected_x(expected_grad_x_data, X.shape(), device);

        auto close_options = CloseOptions().atol(1e-4).equal_nan();
        CHECK(allclose(X.grad().value(), expected_x, close_options));
    };
    runner_single_type<double>(test1);
}

// NOLINTNEXTLINE
TEST_CASE("Min pool autograd") {
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
        std::vector<T> _grad = rand_vec<T>(B * C * H_out * W_out, gen);

        Tensor X(_X, {B, C, H_in, W_in}, device, true);
        Tensor grad(_grad, {B, C, H_out, W_out}, device);
        Tensor out = tinytensor::min_pool2d(X, K, S, P);
        out.backward(grad);

        const auto options_float = torch::TensorOptions().dtype(torch::kFloat64);
        torch::Tensor X_torch = torch::from_blob(_X.data(), {B, C, H_in, W_in}, options_float.requires_grad(true));
        torch::Tensor grad_torch = torch::from_blob(_grad.data(), {B, C, H_out, W_out}, options_float);
        torch::Tensor out_torch = -torch::max_pool2d(-X_torch, K, S, P);
        out_torch.backward(grad_torch);

        torch::Tensor expected_grad_x = X_torch.grad();
        std::vector<T> expected_grad_x_data(
            expected_grad_x.data_ptr<T>(),
            expected_grad_x.data_ptr<T>() + expected_grad_x.numel()
        );
        Tensor expected_x(expected_grad_x_data, X.shape(), device);

        auto close_options = CloseOptions().atol(1e-4).equal_nan();
        CHECK(allclose(X.grad().value(), expected_x, close_options));
    };
    runner_single_type<double>(test1);
}

// NOLINTNEXTLINE
TEST_CASE("Avg pool autograd") {
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
        std::vector<T> _grad = rand_vec<T>(B * C * H_out * W_out, gen);

        Tensor X(_X, {B, C, H_in, W_in}, device, true);
        Tensor grad(_grad, {B, C, H_out, W_out}, device);
        Tensor out = tinytensor::avg_pool2d(X, K, S, P);
        out.backward(grad);

        const auto options_float = torch::TensorOptions().dtype(torch::kFloat64);
        torch::Tensor X_torch = torch::from_blob(_X.data(), {B, C, H_in, W_in}, options_float.requires_grad(true));
        torch::Tensor grad_torch = torch::from_blob(_grad.data(), {B, C, H_out, W_out}, options_float);
        torch::Tensor out_torch = torch::avg_pool2d(X_torch, K, S, P);
        out_torch.backward(grad_torch);

        torch::Tensor expected_grad_x = X_torch.grad();
        std::vector<T> expected_grad_x_data(
            expected_grad_x.data_ptr<T>(),
            expected_grad_x.data_ptr<T>() + expected_grad_x.numel()
        );
        Tensor expected_x(expected_grad_x_data, X.shape(), device);

        auto close_options = CloseOptions().atol(1e-4).equal_nan();
        CHECK(allclose(X.grad().value(), expected_x, close_options));
    };
    runner_single_type<double>(test1);
}

#endif
