// test_shape_autograd.cpp
// Test the shape autograd ops

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <tt/device.h>
#include <tt/exception.h>
#include <tt/index.h>
#include <tt/scalar.h>
#include <tt/tensor.h>

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
TEST_CASE("Shape squeeze autograd") {
    auto test_dim0 = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> d = rand_vec<T>(4 * 4, gen);
        std::vector<T> g = rand_vec<T>(4 * 4, gen);

        Tensor x1(d, {1, 4, 4}, device, true);
        Tensor input_grad_x(g, {4, 4}, device, true);
        Tensor x2 = x1.squeeze(0);
        x2.backward(input_grad_x);

        const auto options = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor t1 = torch::from_blob(d.data(), {1, 4, 4}, options.requires_grad(true));
        torch::Tensor input_grad_t = torch::from_blob(g.data(), {4, 4}, options);
        torch::Tensor t2 = t1.squeeze(0);
        t2.backward(input_grad_t);

        torch::Tensor expected_grad = t1.grad();
        std::vector<T> expected_grad_data(
            expected_grad.data_ptr<T>(),
            expected_grad.data_ptr<T>() + expected_grad.numel()
        );

        Tensor expected(expected_grad_data, {1, 4, 4}, device);
        auto close_options = CloseOptions().atol(1e-6).equal_nan();
        CHECK(allclose(x1.grad().value(), expected, close_options));
    };
    runner_single_type<float>(test_dim0);

    auto test_dim1 = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> d = rand_vec<T>(4 * 4, gen);
        std::vector<T> g = rand_vec<T>(4 * 4, gen);

        Tensor x1(d, {4, 1, 4}, device, true);
        Tensor input_grad_x(g, {4, 4}, device, true);
        Tensor x2 = x1.squeeze(1);
        x2.backward(input_grad_x);

        const auto options = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor t1 = torch::from_blob(d.data(), {4, 1, 4}, options.requires_grad(true));
        torch::Tensor input_grad_t = torch::from_blob(g.data(), {4, 4}, options);
        torch::Tensor t2 = t1.squeeze(1);
        t2.backward(input_grad_t);

        torch::Tensor expected_grad = t1.grad();
        std::vector<T> expected_grad_data(
            expected_grad.data_ptr<T>(),
            expected_grad.data_ptr<T>() + expected_grad.numel()
        );

        Tensor expected(expected_grad_data, {4, 1, 4}, device);
        auto close_options = CloseOptions().atol(1e-6).equal_nan();
        CHECK(allclose(x1.grad().value(), expected, close_options));
    };
    runner_single_type<float>(test_dim1);

    auto test_dim2 = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> d = rand_vec<T>(4 * 4, gen);
        std::vector<T> g = rand_vec<T>(4 * 4, gen);

        Tensor x1(d, {4, 4, 1}, device, true);
        Tensor input_grad_x(g, {4, 4}, device, true);
        Tensor x2 = x1.squeeze(2);
        x2.backward(input_grad_x);

        const auto options = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor t1 = torch::from_blob(d.data(), {4, 4, 1}, options.requires_grad(true));
        torch::Tensor input_grad_t = torch::from_blob(g.data(), {4, 4}, options);
        torch::Tensor t2 = t1.squeeze(2);
        t2.backward(input_grad_t);

        torch::Tensor expected_grad = t1.grad();
        std::vector<T> expected_grad_data(
            expected_grad.data_ptr<T>(),
            expected_grad.data_ptr<T>() + expected_grad.numel()
        );

        Tensor expected(expected_grad_data, {4, 4, 1}, device);
        auto close_options = CloseOptions().atol(1e-6).equal_nan();
        CHECK(allclose(x1.grad().value(), expected, close_options));
    };
    runner_single_type<float>(test_dim2);
}

// NOLINTNEXTLINE
TEST_CASE("Shape unsqueeze autograd") {
    auto test_dim0 = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> d = rand_vec<T>(4 * 4, gen);
        std::vector<T> g = rand_vec<T>(4 * 4, gen);

        Tensor x1(d, {4, 4}, device, true);
        Tensor input_grad_x(g, {1, 4, 4}, device, true);
        Tensor x2 = x1.unsqueeze(0);
        x2.backward(input_grad_x);

        const auto options = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor t1 = torch::from_blob(d.data(), {4, 4}, options.requires_grad(true));
        torch::Tensor input_grad_t = torch::from_blob(g.data(), {1, 4, 4}, options);
        torch::Tensor t2 = t1.unsqueeze(0);
        t2.backward(input_grad_t);

        torch::Tensor expected_grad = t1.grad();
        std::vector<T> expected_grad_data(
            expected_grad.data_ptr<T>(),
            expected_grad.data_ptr<T>() + expected_grad.numel()
        );

        Tensor expected(expected_grad_data, {4, 4}, device);
        auto close_options = CloseOptions().atol(1e-6).equal_nan();
        CHECK(allclose(x1.grad().value(), expected, close_options));
    };
    runner_single_type<float>(test_dim0);

    auto test_dim1 = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> d = rand_vec<T>(4 * 4, gen);
        std::vector<T> g = rand_vec<T>(4 * 4, gen);

        Tensor x1(d, {4, 4}, device, true);
        Tensor input_grad_x(g, {4, 1, 4}, device, true);
        Tensor x2 = x1.unsqueeze(1);
        x2.backward(input_grad_x);

        const auto options = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor t1 = torch::from_blob(d.data(), {4, 4}, options.requires_grad(true));
        torch::Tensor input_grad_t = torch::from_blob(g.data(), {4, 1, 4}, options);
        torch::Tensor t2 = t1.unsqueeze(1);
        t2.backward(input_grad_t);

        torch::Tensor expected_grad = t1.grad();
        std::vector<T> expected_grad_data(
            expected_grad.data_ptr<T>(),
            expected_grad.data_ptr<T>() + expected_grad.numel()
        );

        Tensor expected(expected_grad_data, {4, 4}, device);
        auto close_options = CloseOptions().atol(1e-6).equal_nan();
        CHECK(allclose(x1.grad().value(), expected, close_options));
    };
    runner_single_type<float>(test_dim1);

    auto test_dim2 = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> d = rand_vec<T>(4 * 4, gen);
        std::vector<T> g = rand_vec<T>(4 * 4, gen);

        Tensor x1(d, {4, 4}, device, true);
        Tensor input_grad_x(g, {4, 4, 1}, device, true);
        Tensor x2 = x1.unsqueeze(2);
        x2.backward(input_grad_x);

        const auto options = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor t1 = torch::from_blob(d.data(), {4, 4}, options.requires_grad(true));
        torch::Tensor input_grad_t = torch::from_blob(g.data(), {4, 4, 1}, options);
        torch::Tensor t2 = t1.unsqueeze(2);
        t2.backward(input_grad_t);

        torch::Tensor expected_grad = t1.grad();
        std::vector<T> expected_grad_data(
            expected_grad.data_ptr<T>(),
            expected_grad.data_ptr<T>() + expected_grad.numel()
        );

        Tensor expected(expected_grad_data, {4, 4}, device);
        auto close_options = CloseOptions().atol(1e-6).equal_nan();
        CHECK(allclose(x1.grad().value(), expected, close_options));
    };
    runner_single_type<float>(test_dim2);
}

// NOLINTNEXTLINE
TEST_CASE("Shape broadcast autograd") {
    auto test_dim0 = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> d = rand_vec<T>(4 * 3, gen);
        std::vector<T> g = rand_vec<T>(2 * 4 * 3, gen);

        Tensor x1(d, {1, 4, 3}, device, true);
        Tensor input_grad_x(g, {2, 4, 3}, device, true);
        Tensor x2 = x1.expand({2, 4, 3});
        x2.backward(input_grad_x);

        const auto options = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor t1 = torch::from_blob(d.data(), {1, 4, 3}, options.requires_grad(true));
        torch::Tensor input_grad_t = torch::from_blob(g.data(), {2, 4, 3}, options);
        torch::Tensor t2 = t1.expand({2, 4, 3});
        t2.backward(input_grad_t);

        torch::Tensor expected_grad = t1.grad();
        std::vector<T> expected_grad_data(
            expected_grad.data_ptr<T>(),
            expected_grad.data_ptr<T>() + expected_grad.numel()
        );

        Tensor expected(expected_grad_data, {1, 4, 3}, device);
        auto close_options = CloseOptions().atol(1e-6).equal_nan();
        CHECK(allclose(x1.grad().value(), expected, close_options));
    };
    runner_single_type<float>(test_dim0);

    auto test_dim1 = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> d = rand_vec<T>(4 * 3, gen);
        std::vector<T> g = rand_vec<T>(2 * 4 * 3, gen);

        Tensor x1(d, {4, 1, 3}, device, true);
        Tensor input_grad_x(g, {4, 2, 3}, device, true);
        Tensor x2 = x1.expand({4, 2, 3});
        x2.backward(input_grad_x);

        const auto options = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor t1 = torch::from_blob(d.data(), {4, 1, 3}, options.requires_grad(true));
        torch::Tensor input_grad_t = torch::from_blob(g.data(), {4, 2, 3}, options);
        torch::Tensor t2 = t1.expand({4, 2, 3});
        t2.backward(input_grad_t);

        torch::Tensor expected_grad = t1.grad();
        std::vector<T> expected_grad_data(
            expected_grad.data_ptr<T>(),
            expected_grad.data_ptr<T>() + expected_grad.numel()
        );

        Tensor expected(expected_grad_data, {4, 1, 3}, device);
        auto close_options = CloseOptions().atol(1e-6).equal_nan();
        CHECK(allclose(x1.grad().value(), expected, close_options));
    };
    runner_single_type<float>(test_dim1);

    auto test_extended = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> d = rand_vec<T>(3 * 1 * 2, gen);
        std::vector<T> g = rand_vec<T>(4 * 3 * 2 * 2, gen);

        Tensor x1(d, {3, 1, 2}, device, true);
        Tensor input_grad_x(g, {4, 3, 2, 2}, device, true);
        Tensor x2 = x1.expand({4, 3, 2, 2});
        x2.backward(input_grad_x);

        const auto options = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor t1 = torch::from_blob(d.data(), {3, 1, 2}, options.requires_grad(true));
        torch::Tensor input_grad_t = torch::from_blob(g.data(), {4, 3, 2, 2}, options);
        torch::Tensor t2 = t1.expand({4, 3, 2, 2});
        t2.backward(input_grad_t);

        torch::Tensor expected_grad = t1.grad();
        std::vector<T> expected_grad_data(
            expected_grad.data_ptr<T>(),
            expected_grad.data_ptr<T>() + expected_grad.numel()
        );

        Tensor expected(expected_grad_data, {3, 1, 2}, device);
        auto close_options = CloseOptions().atol(1e-6).equal_nan();
        CHECK(allclose(x1.grad().value(), expected, close_options));
    };
    runner_single_type<float>(test_extended);
}

// NOLINTNEXTLINE
TEST_CASE("Shape reshape autograd") {
    auto test = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> d = rand_vec<T>(4 * 4, gen);
        std::vector<T> g = rand_vec<T>(4 * 4, gen);

        Tensor x1(d, {4, 4}, device, true);
        Tensor input_grad_x(g, {8, 2}, device, true);
        Tensor x2 = x1.reshape({8, 2});
        x2.backward(input_grad_x);

        const auto options = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor t1 = torch::from_blob(d.data(), {4, 4}, options.requires_grad(true));
        torch::Tensor input_grad_t = torch::from_blob(g.data(), {8, 2}, options);
        torch::Tensor t2 = t1.reshape({8, 2});
        t2.backward(input_grad_t);

        torch::Tensor expected_grad = t1.grad();
        std::vector<T> expected_grad_data(
            expected_grad.data_ptr<T>(),
            expected_grad.data_ptr<T>() + expected_grad.numel()
        );

        Tensor expected(expected_grad_data, {4, 4}, device);
        auto close_options = CloseOptions().atol(1e-6).equal_nan();
        CHECK(allclose(x1.grad().value(), expected, close_options));
    };
    runner_single_type<float>(test);
}

// NOLINTNEXTLINE
TEST_CASE("Shape flatten autograd") {
    auto test = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> d = rand_vec<T>(4 * 4, gen);
        std::vector<T> g = rand_vec<T>(4 * 4, gen);

        Tensor x1(d, {4, 4}, device, true);
        Tensor input_grad_x(g, {4 * 4}, device, true);
        Tensor x2 = x1.flatten();
        x2.backward(input_grad_x);

        const auto options = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor t1 = torch::from_blob(d.data(), {4, 4}, options.requires_grad(true));
        torch::Tensor input_grad_t = torch::from_blob(g.data(), {4 * 4}, options);
        torch::Tensor t2 = t1.flatten();
        t2.backward(input_grad_t);

        torch::Tensor expected_grad = t1.grad();
        std::vector<T> expected_grad_data(
            expected_grad.data_ptr<T>(),
            expected_grad.data_ptr<T>() + expected_grad.numel()
        );

        Tensor expected(expected_grad_data, {4, 4}, device);
        auto close_options = CloseOptions().atol(1e-6).equal_nan();
        CHECK(allclose(x1.grad().value(), expected, close_options));
    };
    runner_single_type<float>(test);
}

// NOLINTNEXTLINE
TEST_CASE("Shape permute autograd") {
    auto test = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> d = rand_vec<T>(3 * 1 * 2, gen);
        std::vector<T> g = rand_vec<T>(3 * 1 * 2, gen);

        Tensor x1(d, {3, 1, 2}, device, true);
        Tensor input_grad_x(g, {1, 2, 3}, device, true);
        Tensor x2 = x1.permute({1, 2, 0});
        x2.backward(input_grad_x);

        const auto options = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor t1 = torch::from_blob(d.data(), {3, 1, 2}, options.requires_grad(true));
        torch::Tensor input_grad_t = torch::from_blob(g.data(), {1, 2, 3}, options);
        torch::Tensor t2 = t1.permute({1, 2, 0});
        t2.backward(input_grad_t);

        torch::Tensor expected_grad = t1.grad();
        std::vector<T> expected_grad_data(
            expected_grad.data_ptr<T>(),
            expected_grad.data_ptr<T>() + expected_grad.numel()
        );

        Tensor expected(expected_grad_data, {3, 1, 2}, device);
        auto close_options = CloseOptions().atol(1e-6).equal_nan();
        CHECK(allclose(x1.grad().value(), expected, close_options));
    };
    runner_single_type<float>(test);
}

// NOLINTNEXTLINE
TEST_CASE("Shape multi autograd") {
    auto test = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> d = rand_vec<T>(2 * 3 * 4, gen);
        std::vector<T> g = rand_vec<T>(12 * 4, gen);

        Tensor x1(d, {2, 3, 4}, device, true);
        Tensor input_grad_x(g, {12, 4}, device, true);
        Tensor x2 = x1.permute({1, 2, 0});
        x2 = x2[{indexing::Slice(), 1}];
        Tensor x3 = x2.unsqueeze(1);
        Tensor x4 = x3.expand({4, 3, 2, 2});
        Tensor x5 = x4.reshape({12, 4});
        x5.backward(input_grad_x);

        const auto options = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor t1 = torch::from_blob(d.data(), {2, 3, 4}, options.requires_grad(true));
        torch::Tensor input_grad_t = torch::from_blob(g.data(), {12, 4}, options);
        torch::Tensor t2 = t1.permute({1, 2, 0});
        t2 = t2.index({torch::indexing::Slice(), 1});
        torch::Tensor t3 = t2.unsqueeze(1);
        torch::Tensor t4 = t3.expand({4, 3, 2, 2});
        torch::Tensor t5 = t4.reshape({12, 4});
        t5.backward(input_grad_t);

        torch::Tensor expected_grad = t1.grad();
        std::vector<T> expected_grad_data(
            expected_grad.data_ptr<T>(),
            expected_grad.data_ptr<T>() + expected_grad.numel()
        );

        Tensor expected(expected_grad_data, {2, 3, 4}, device);
        auto close_options = CloseOptions().atol(1e-6).equal_nan();
        CHECK(allclose(x1.grad().value(), expected, close_options));
    };
    runner_single_type<float>(test);
}

#endif
