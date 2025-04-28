// test_matmul_autograd.cpp
// Test the matmul autograd ops
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
    std::uniform_real_distribution<float> dis(-5.0, 5.0);
    for (int i = 0; i < size; ++i) {
        res.push_back(static_cast<T>(dis(gen)));
    }
    return res;
}

}    // namespace

#ifdef TT_TORCH

// NOLINTNEXTLINE
TEST_CASE("Misc Dot product autograd") {
    auto test = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> d_lhs = rand_vec<T>(2048, gen);
        std::vector<T> d_rhs = rand_vec<T>(2048, gen);
        std::vector<T> grad_out = rand_vec<T>(1, gen);

        Tensor lhs(d_lhs, {2048}, device, true);
        Tensor rhs(d_rhs, {2048}, device, true);
        Tensor output_grad(grad_out, {1}, device);

        Tensor x2 = matmul(lhs, rhs);
        x2.backward(output_grad);

        const auto options_f = torch::TensorOptions().dtype(torch::kFloat64);
        torch::Tensor t_lhs = torch::from_blob(d_lhs.data(), {2048}, options_f.requires_grad(true));
        torch::Tensor t_rhs = torch::from_blob(d_rhs.data(), {2048}, options_f.requires_grad(true));
        torch::Tensor t_grad = torch::from_blob(grad_out.data(), {1}, options_f);

        torch::Tensor t2 = torch::matmul(t_lhs, t_rhs);
        t2.backward(t_grad);
        torch::Tensor expected_grad_lhs = t_lhs.grad();
        torch::Tensor expected_grad_rhs = t_rhs.grad();
        std::vector<T> expected_grad_lhs_data(
            expected_grad_lhs.data_ptr<T>(),
            expected_grad_lhs.data_ptr<T>() + expected_grad_lhs.numel()
        );
        std::vector<T> expected_grad_rhs_data(
            expected_grad_rhs.data_ptr<T>(),
            expected_grad_rhs.data_ptr<T>() + expected_grad_rhs.numel()
        );
        Tensor expected_lhs(expected_grad_lhs_data, {2048}, device);
        Tensor expected_rhs(expected_grad_rhs_data, {2048}, device);

        auto close_options = CloseOptions().atol(1e-4).equal_nan();
        CHECK(allclose(lhs.grad().value(), expected_lhs, close_options));
        CHECK(allclose(rhs.grad().value(), expected_rhs, close_options));
    };
    runner_single_type<double>(test);
}

// NOLINTNEXTLINE
TEST_CASE("Misc Vector Matrix product autograd") {
    auto test = []<typename T>(Device device) {
        std::mt19937 gen(0);
        constexpr int N = 256 + 123;
        constexpr int M = 512 + 123;
        std::vector<T> d_lhs = rand_vec<T>(N, gen);
        std::vector<T> d_rhs = rand_vec<T>(N * M, gen);
        std::vector<T> grad_out = rand_vec<T>(M, gen);

        Tensor lhs(d_lhs, {N}, device, true);
        Tensor rhs(d_rhs, {N, M}, device, true);
        Tensor output_grad(grad_out, {M}, device);

        Tensor x2 = matmul(lhs, rhs);
        x2.backward(output_grad);

        const auto options_f = torch::TensorOptions().dtype(torch::kFloat64);
        torch::Tensor t_lhs = torch::from_blob(d_lhs.data(), {N}, options_f.requires_grad(true));
        torch::Tensor t_rhs = torch::from_blob(d_rhs.data(), {N, M}, options_f.requires_grad(true));
        torch::Tensor t_grad = torch::from_blob(grad_out.data(), {M}, options_f);

        torch::Tensor t2 = torch::matmul(t_lhs, t_rhs);
        t2.backward(t_grad);
        torch::Tensor expected_grad_lhs = t_lhs.grad();
        torch::Tensor expected_grad_rhs = t_rhs.grad();
        std::vector<T> expected_grad_lhs_data(
            expected_grad_lhs.data_ptr<T>(),
            expected_grad_lhs.data_ptr<T>() + expected_grad_lhs.numel()
        );
        std::vector<T> expected_grad_rhs_data(
            expected_grad_rhs.data_ptr<T>(),
            expected_grad_rhs.data_ptr<T>() + expected_grad_rhs.numel()
        );
        Tensor expected_lhs(expected_grad_lhs_data, {N}, device);
        Tensor expected_rhs(expected_grad_rhs_data, {N, M}, device);

        auto close_options = CloseOptions().atol(1e-6).equal_nan();
        CHECK(allclose(lhs.grad().value(), expected_lhs, close_options));
        CHECK(allclose(rhs.grad().value(), expected_rhs, close_options));
    };
    runner_single_type<double>(test);
}

// NOLINTNEXTLINE
TEST_CASE("Misc Matrix Vector product autograd") {
    auto test = []<typename T>(Device device) {
        std::mt19937 gen(0);
        constexpr int N = 256 + 123;
        constexpr int M = 512 + 123;
        std::vector<T> d_lhs = rand_vec<T>(N * M, gen);
        std::vector<T> d_rhs = rand_vec<T>(M, gen);
        std::vector<T> grad_out = rand_vec<T>(N, gen);

        Tensor lhs(d_lhs, {N, M}, device, true);
        Tensor rhs(d_rhs, {M}, device, true);
        Tensor output_grad(grad_out, {N}, device);

        Tensor x2 = matmul(lhs, rhs);
        x2.backward(output_grad);

        const auto options_f = torch::TensorOptions().dtype(torch::kFloat64);
        torch::Tensor t_lhs = torch::from_blob(d_lhs.data(), {N, M}, options_f.requires_grad(true));
        torch::Tensor t_rhs = torch::from_blob(d_rhs.data(), {M}, options_f.requires_grad(true));
        torch::Tensor t_grad = torch::from_blob(grad_out.data(), {N}, options_f);

        torch::Tensor t2 = torch::matmul(t_lhs, t_rhs);
        t2.backward(t_grad);
        torch::Tensor expected_grad_lhs = t_lhs.grad();
        torch::Tensor expected_grad_rhs = t_rhs.grad();
        std::vector<T> expected_grad_lhs_data(
            expected_grad_lhs.data_ptr<T>(),
            expected_grad_lhs.data_ptr<T>() + expected_grad_lhs.numel()
        );
        std::vector<T> expected_grad_rhs_data(
            expected_grad_rhs.data_ptr<T>(),
            expected_grad_rhs.data_ptr<T>() + expected_grad_rhs.numel()
        );
        Tensor expected_lhs(expected_grad_lhs_data, {N, M}, device);
        Tensor expected_rhs(expected_grad_rhs_data, {M}, device);

        auto close_options = CloseOptions().atol(1e-6).equal_nan();
        CHECK(allclose(lhs.grad().value(), expected_lhs, close_options));
        CHECK(allclose(rhs.grad().value(), expected_rhs, close_options));
    };
    runner_single_type<double>(test);
}

// NOLINTNEXTLINE
TEST_CASE("Misc Matrix Matrix product autograd") {
    auto test = []<typename T>(Device device) {
        std::mt19937 gen(0);
        constexpr int N = 256 + 123;
        constexpr int M = 1024 + 123;
        constexpr int K = 512 + 123;
        std::vector<T> d_lhs = rand_vec<T>(N * M, gen);
        std::vector<T> d_rhs = rand_vec<T>(M * K, gen);
        std::vector<T> grad_out = rand_vec<T>(N * K, gen);

        Tensor lhs(d_lhs, {N, M}, device, true);
        Tensor rhs(d_rhs, {M, K}, device, true);
        Tensor output_grad(grad_out, {N, K}, device);

        Tensor x2 = matmul(lhs, rhs);
        x2.backward(output_grad);

        const auto options_f = torch::TensorOptions().dtype(torch::kFloat64);
        torch::Tensor t_lhs = torch::from_blob(d_lhs.data(), {N, M}, options_f.requires_grad(true));
        torch::Tensor t_rhs = torch::from_blob(d_rhs.data(), {M, K}, options_f.requires_grad(true));
        torch::Tensor t_grad = torch::from_blob(grad_out.data(), {N, K}, options_f);

        torch::Tensor t2 = torch::matmul(t_lhs, t_rhs);
        t2.backward(t_grad);
        torch::Tensor expected_grad_lhs = t_lhs.grad();
        torch::Tensor expected_grad_rhs = t_rhs.grad();
        std::vector<T> expected_grad_lhs_data(
            expected_grad_lhs.data_ptr<T>(),
            expected_grad_lhs.data_ptr<T>() + expected_grad_lhs.numel()
        );
        std::vector<T> expected_grad_rhs_data(
            expected_grad_rhs.data_ptr<T>(),
            expected_grad_rhs.data_ptr<T>() + expected_grad_rhs.numel()
        );
        Tensor expected_lhs(expected_grad_lhs_data, {N, M}, device);
        Tensor expected_rhs(expected_grad_rhs_data, {M, K}, device);

        auto close_options = CloseOptions().atol(1e-6).equal_nan();
        CHECK(allclose(lhs.grad().value(), expected_lhs, close_options));
        CHECK(allclose(rhs.grad().value(), expected_rhs, close_options));
    };
    runner_single_type<double>(test);
}

// NOLINTNEXTLINE
TEST_CASE("Misc Batched Matrix Matrix product autograd") {
    auto test = []<typename T>(Device device) {
        std::mt19937 gen(0);
        constexpr int B = 4;
        constexpr int N = 256 + 123;
        constexpr int M = 1024 + 123;
        constexpr int K = 512 + 123;
        std::vector<T> d_lhs = rand_vec<T>(B * N * M, gen);
        std::vector<T> d_rhs = rand_vec<T>(B * M * K, gen);
        std::vector<T> grad_out = rand_vec<T>(B * N * K, gen);

        Tensor lhs(d_lhs, {B, N, M}, device, true);
        Tensor rhs(d_rhs, {B, M, K}, device, true);
        Tensor output_grad(grad_out, {B, N, K}, device);

        Tensor x2 = matmul(lhs, rhs);
        x2.backward(output_grad);

        const auto options_f = torch::TensorOptions().dtype(torch::kFloat64);
        torch::Tensor t_lhs = torch::from_blob(d_lhs.data(), {B, N, M}, options_f.requires_grad(true));
        torch::Tensor t_rhs = torch::from_blob(d_rhs.data(), {B, M, K}, options_f.requires_grad(true));
        torch::Tensor t_grad = torch::from_blob(grad_out.data(), {B, N, K}, options_f);

        torch::Tensor t2 = torch::matmul(t_lhs, t_rhs);
        t2.backward(t_grad);
        torch::Tensor expected_grad_lhs = t_lhs.grad();
        torch::Tensor expected_grad_rhs = t_rhs.grad();
        std::vector<T> expected_grad_lhs_data(
            expected_grad_lhs.data_ptr<T>(),
            expected_grad_lhs.data_ptr<T>() + expected_grad_lhs.numel()
        );
        std::vector<T> expected_grad_rhs_data(
            expected_grad_rhs.data_ptr<T>(),
            expected_grad_rhs.data_ptr<T>() + expected_grad_rhs.numel()
        );
        Tensor expected_lhs(expected_grad_lhs_data, {B, N, M}, device);
        Tensor expected_rhs(expected_grad_rhs_data, {B, M, K}, device);

        auto close_options = CloseOptions().atol(1e-6).equal_nan();
        CHECK(allclose(lhs.grad().value(), expected_lhs, close_options));
        CHECK(allclose(rhs.grad().value(), expected_rhs, close_options));
    };
    runner_single_type<double>(test);
}

#endif
