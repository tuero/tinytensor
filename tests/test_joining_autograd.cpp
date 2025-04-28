// test_joining_autograd.cpp
// Test the joining autograd ops

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
TEST_CASE("Joining cat autograd") {
    auto test_dim0 = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> d1 = rand_vec<T>(3 * 4, gen);
        std::vector<T> d2 = rand_vec<T>(1 * 4, gen);
        std::vector<T> d3 = rand_vec<T>(3 * 4, gen);
        std::vector<T> grad_out = rand_vec<T>(7 * 4, gen);

        Tensor x1(d1, {3, 4}, device, true);
        Tensor x2(d2, {1, 4}, device, true);
        Tensor x3(d3, {3, 4}, device, true);
        Tensor output_grad(grad_out, {7, 4}, device);

        Tensor x4 = cat({x1, x2, x3}, 0);
        x4.backward(output_grad);

        const auto options_f = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor t1 = torch::from_blob(d1.data(), {3, 4}, options_f.requires_grad(true));
        torch::Tensor t2 = torch::from_blob(d2.data(), {1, 4}, options_f.requires_grad(true));
        torch::Tensor t3 = torch::from_blob(d3.data(), {3, 4}, options_f.requires_grad(true));
        torch::Tensor t_grad = torch::from_blob(grad_out.data(), {7, 4}, options_f);

        torch::Tensor t4 = torch::cat({t1, t2, t3}, 0);
        t4.backward(t_grad);
        torch::Tensor expected_grad_1 = t1.grad();
        torch::Tensor expected_grad_2 = t2.grad();
        torch::Tensor expected_grad_3 = t3.grad();
        std::vector<T> expected_grad_1_data(
            expected_grad_1.data_ptr<T>(),
            expected_grad_1.data_ptr<T>() + expected_grad_1.numel()
        );
        std::vector<T> expected_grad_2_data(
            expected_grad_2.data_ptr<T>(),
            expected_grad_2.data_ptr<T>() + expected_grad_2.numel()
        );
        std::vector<T> expected_grad_3_data(
            expected_grad_3.data_ptr<T>(),
            expected_grad_3.data_ptr<T>() + expected_grad_3.numel()
        );
        Tensor expected_1(expected_grad_1_data, {3, 4}, device);
        Tensor expected_2(expected_grad_2_data, {1, 4}, device);
        Tensor expected_3(expected_grad_3_data, {3, 4}, device);

        auto close_options = CloseOptions().atol(1e-4).equal_nan();
        CHECK(allclose(x1.grad().value(), expected_1, close_options));
        CHECK(allclose(x2.grad().value(), expected_2, close_options));
        CHECK(allclose(x3.grad().value(), expected_3, close_options));
    };
    runner_single_type<float>(test_dim0);

    auto test_dim1 = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> d1 = rand_vec<T>(3 * 4, gen);
        std::vector<T> d2 = rand_vec<T>(3 * 1, gen);
        std::vector<T> d3 = rand_vec<T>(3 * 4, gen);
        std::vector<T> grad_out = rand_vec<T>(3 * 9, gen);

        Tensor x1(d1, {3, 4}, device, true);
        Tensor x2(d2, {3, 1}, device, true);
        Tensor x3(d3, {3, 4}, device, true);
        Tensor output_grad(grad_out, {3, 9}, device);

        Tensor x4 = cat({x1, x2, x3}, 1);
        x4.backward(output_grad);

        const auto options_f = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor t1 = torch::from_blob(d1.data(), {3, 4}, options_f.requires_grad(true));
        torch::Tensor t2 = torch::from_blob(d2.data(), {3, 1}, options_f.requires_grad(true));
        torch::Tensor t3 = torch::from_blob(d3.data(), {3, 4}, options_f.requires_grad(true));
        torch::Tensor t_grad = torch::from_blob(grad_out.data(), {3, 9}, options_f);

        torch::Tensor t4 = torch::cat({t1, t2, t3}, 1);
        t4.backward(t_grad);
        torch::Tensor expected_grad_1 = t1.grad();
        torch::Tensor expected_grad_2 = t2.grad();
        torch::Tensor expected_grad_3 = t3.grad();
        std::vector<T> expected_grad_1_data(
            expected_grad_1.data_ptr<T>(),
            expected_grad_1.data_ptr<T>() + expected_grad_1.numel()
        );
        std::vector<T> expected_grad_2_data(
            expected_grad_2.data_ptr<T>(),
            expected_grad_2.data_ptr<T>() + expected_grad_2.numel()
        );
        std::vector<T> expected_grad_3_data(
            expected_grad_3.data_ptr<T>(),
            expected_grad_3.data_ptr<T>() + expected_grad_3.numel()
        );
        Tensor expected_1(expected_grad_1_data, {3, 4}, device);
        Tensor expected_2(expected_grad_2_data, {3, 1}, device);
        Tensor expected_3(expected_grad_3_data, {3, 4}, device);

        auto close_options = CloseOptions().atol(1e-4).equal_nan();
        CHECK(allclose(x1.grad().value(), expected_1, close_options));
        CHECK(allclose(x2.grad().value(), expected_2, close_options));
        CHECK(allclose(x3.grad().value(), expected_3, close_options));
    };
    runner_single_type<float>(test_dim1);
}

// NOLINTNEXTLINE
TEST_CASE("Joining stack autograd") {
    auto test_dim0 = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> d1 = rand_vec<T>(3 * 4, gen);
        std::vector<T> d2 = rand_vec<T>(3 * 4, gen);
        std::vector<T> d3 = rand_vec<T>(3 * 4, gen);
        std::vector<T> grad_out = rand_vec<T>(3 * 3 * 4, gen);

        Tensor x1(d1, {3, 4}, device, true);
        Tensor x2(d2, {3, 4}, device, true);
        Tensor x3(d3, {3, 4}, device, true);
        Tensor output_grad(grad_out, {3, 3, 4}, device);

        Tensor x4 = stack({x1, x2, x3}, 0);
        x4.backward(output_grad);

        const auto options_f = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor t1 = torch::from_blob(d1.data(), {3, 4}, options_f.requires_grad(true));
        torch::Tensor t2 = torch::from_blob(d2.data(), {3, 4}, options_f.requires_grad(true));
        torch::Tensor t3 = torch::from_blob(d3.data(), {3, 4}, options_f.requires_grad(true));
        torch::Tensor t_grad = torch::from_blob(grad_out.data(), {3, 3, 4}, options_f);

        torch::Tensor t4 = torch::stack({t1, t2, t3}, 0);
        t4.backward(t_grad);
        torch::Tensor expected_grad_1 = t1.grad();
        torch::Tensor expected_grad_2 = t2.grad();
        torch::Tensor expected_grad_3 = t3.grad();
        std::vector<T> expected_grad_1_data(
            expected_grad_1.data_ptr<T>(),
            expected_grad_1.data_ptr<T>() + expected_grad_1.numel()
        );
        std::vector<T> expected_grad_2_data(
            expected_grad_2.data_ptr<T>(),
            expected_grad_2.data_ptr<T>() + expected_grad_2.numel()
        );
        std::vector<T> expected_grad_3_data(
            expected_grad_3.data_ptr<T>(),
            expected_grad_3.data_ptr<T>() + expected_grad_3.numel()
        );
        Tensor expected_1(expected_grad_1_data, {3, 4}, device);
        Tensor expected_2(expected_grad_2_data, {3, 4}, device);
        Tensor expected_3(expected_grad_3_data, {3, 4}, device);

        auto close_options = CloseOptions().atol(1e-4).equal_nan();
        CHECK(allclose(x1.grad().value(), expected_1, close_options));
        CHECK(allclose(x2.grad().value(), expected_2, close_options));
        CHECK(allclose(x3.grad().value(), expected_3, close_options));
    };
    runner_single_type<float>(test_dim0);

    auto test_dim1 = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> d1 = rand_vec<T>(3 * 4, gen);
        std::vector<T> d2 = rand_vec<T>(3 * 4, gen);
        std::vector<T> d3 = rand_vec<T>(3 * 4, gen);
        std::vector<T> grad_out = rand_vec<T>(3 * 3 * 4, gen);

        Tensor x1(d1, {3, 4}, device, true);
        Tensor x2(d2, {3, 4}, device, true);
        Tensor x3(d3, {3, 4}, device, true);
        Tensor output_grad(grad_out, {3, 3, 4}, device);

        Tensor x4 = stack({x1, x2, x3}, 1);
        x4.backward(output_grad);

        const auto options_f = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor t1 = torch::from_blob(d1.data(), {3, 4}, options_f.requires_grad(true));
        torch::Tensor t2 = torch::from_blob(d2.data(), {3, 4}, options_f.requires_grad(true));
        torch::Tensor t3 = torch::from_blob(d3.data(), {3, 4}, options_f.requires_grad(true));
        torch::Tensor t_grad = torch::from_blob(grad_out.data(), {3, 3, 4}, options_f);

        torch::Tensor t4 = torch::stack({t1, t2, t3}, 1);
        t4.backward(t_grad);
        torch::Tensor expected_grad_1 = t1.grad();
        torch::Tensor expected_grad_2 = t2.grad();
        torch::Tensor expected_grad_3 = t3.grad();
        std::vector<T> expected_grad_1_data(
            expected_grad_1.data_ptr<T>(),
            expected_grad_1.data_ptr<T>() + expected_grad_1.numel()
        );
        std::vector<T> expected_grad_2_data(
            expected_grad_2.data_ptr<T>(),
            expected_grad_2.data_ptr<T>() + expected_grad_2.numel()
        );
        std::vector<T> expected_grad_3_data(
            expected_grad_3.data_ptr<T>(),
            expected_grad_3.data_ptr<T>() + expected_grad_3.numel()
        );
        Tensor expected_1(expected_grad_1_data, {3, 4}, device);
        Tensor expected_2(expected_grad_2_data, {3, 4}, device);
        Tensor expected_3(expected_grad_3_data, {3, 4}, device);

        auto close_options = CloseOptions().atol(1e-4).equal_nan();
        CHECK(allclose(x1.grad().value(), expected_1, close_options));
        CHECK(allclose(x2.grad().value(), expected_2, close_options));
        CHECK(allclose(x3.grad().value(), expected_3, close_options));
    };
    runner_single_type<float>(test_dim1);

    auto test_dim2 = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> d1 = rand_vec<T>(3 * 4, gen);
        std::vector<T> d2 = rand_vec<T>(3 * 4, gen);
        std::vector<T> d3 = rand_vec<T>(3 * 4, gen);
        std::vector<T> grad_out = rand_vec<T>(3 * 4 * 3, gen);

        Tensor x1(d1, {3, 4}, device, true);
        Tensor x2(d2, {3, 4}, device, true);
        Tensor x3(d3, {3, 4}, device, true);
        Tensor output_grad(grad_out, {3, 4, 3}, device);

        Tensor x4 = stack({x1, x2, x3}, 2);
        x4.backward(output_grad);

        const auto options_f = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor t1 = torch::from_blob(d1.data(), {3, 4}, options_f.requires_grad(true));
        torch::Tensor t2 = torch::from_blob(d2.data(), {3, 4}, options_f.requires_grad(true));
        torch::Tensor t3 = torch::from_blob(d3.data(), {3, 4}, options_f.requires_grad(true));
        torch::Tensor t_grad = torch::from_blob(grad_out.data(), {3, 4, 3}, options_f);

        torch::Tensor t4 = torch::stack({t1, t2, t3}, 2);
        t4.backward(t_grad);
        torch::Tensor expected_grad_1 = t1.grad();
        torch::Tensor expected_grad_2 = t2.grad();
        torch::Tensor expected_grad_3 = t3.grad();
        std::vector<T> expected_grad_1_data(
            expected_grad_1.data_ptr<T>(),
            expected_grad_1.data_ptr<T>() + expected_grad_1.numel()
        );
        std::vector<T> expected_grad_2_data(
            expected_grad_2.data_ptr<T>(),
            expected_grad_2.data_ptr<T>() + expected_grad_2.numel()
        );
        std::vector<T> expected_grad_3_data(
            expected_grad_3.data_ptr<T>(),
            expected_grad_3.data_ptr<T>() + expected_grad_3.numel()
        );
        Tensor expected_1(expected_grad_1_data, {3, 4}, device);
        Tensor expected_2(expected_grad_2_data, {3, 4}, device);
        Tensor expected_3(expected_grad_3_data, {3, 4}, device);

        auto close_options = CloseOptions().atol(1e-4).equal_nan();
        CHECK(allclose(x1.grad().value(), expected_1, close_options));
        CHECK(allclose(x2.grad().value(), expected_2, close_options));
        CHECK(allclose(x3.grad().value(), expected_3, close_options));
    };
    runner_single_type<float>(test_dim2);
}

#endif
