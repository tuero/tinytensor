// test_reduce_autograd.cpp
// Test the reduction autograd ops

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <tt/device.h>
#include <tt/exception.h>
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
#define REDUCE_AUTOGRAD_TEST_CASE(NAME, FUNC, TORCH_FUNC)                                       \
    TEST_CASE("Unary " NAME " autograd") {                                                      \
        auto test_all = []<typename T>(Device device) {                                         \
            std::mt19937 gen(0);                                                                \
            std::vector<T> d = rand_vec<T>(4 * 4, gen);                                         \
            std::vector<T> g = rand_vec<T>(1, gen);                                             \
            Tensor x1(d, {4, 4}, device, true);                                                 \
            Tensor input_grad_x(g, {1}, device, true);                                          \
            Tensor x2 = FUNC(x1);                                                               \
            x2.backward(input_grad_x);                                                          \
            const auto options = torch::TensorOptions().dtype(torch::kFloat);                   \
            torch::Tensor t1 = torch::from_blob(d.data(), {4, 4}, options.requires_grad(true)); \
            torch::Tensor input_grad_t = torch::from_blob(g.data(), {1}, options);              \
            torch::Tensor t2 = TORCH_FUNC(t1);                                                  \
            t2.backward(input_grad_t);                                                          \
            torch::Tensor expected_grad = t1.grad();                                            \
            std::vector<T> expected_grad_data(                                                  \
                expected_grad.data_ptr<T>(),                                                    \
                expected_grad.data_ptr<T>() + expected_grad.numel()                             \
            );                                                                                  \
            Tensor expected(expected_grad_data, {4, 4}, device);                                \
            auto close_options = CloseOptions().atol(1e-6).equal_nan();                         \
            CHECK(allclose(x1.grad().value(), expected, close_options));                        \
        };                                                                                      \
        runner_single_type<float>(test_all);                                                    \
        auto test_dim_0 = []<typename T>(Device device) {                                       \
            std::mt19937 gen(0);                                                                \
            std::vector<T> d = rand_vec<T>(4 * 4, gen);                                         \
            std::vector<T> g = rand_vec<T>(4, gen);                                             \
            Tensor x1(d, {4, 4}, device, true);                                                 \
            Tensor input_grad_x(g, {4}, device, true);                                          \
            Tensor x2 = FUNC(x1, 0, false);                                                     \
            x2.backward(input_grad_x);                                                          \
            const auto options = torch::TensorOptions().dtype(torch::kFloat);                   \
            torch::Tensor t1 = torch::from_blob(d.data(), {4, 4}, options.requires_grad(true)); \
            torch::Tensor input_grad_t = torch::from_blob(g.data(), {4}, options);              \
            torch::Tensor t2 = TORCH_FUNC(t1, 0, false);                                        \
            t2.backward(input_grad_t);                                                          \
            torch::Tensor expected_grad = t1.grad();                                            \
            std::vector<T> expected_grad_data(                                                  \
                expected_grad.data_ptr<T>(),                                                    \
                expected_grad.data_ptr<T>() + expected_grad.numel()                             \
            );                                                                                  \
            Tensor expected(expected_grad_data, {4, 4}, device);                                \
            auto close_options = CloseOptions().atol(1e-6).equal_nan();                         \
            CHECK(allclose(x1.grad().value(), expected, close_options));                        \
        };                                                                                      \
        runner_single_type<float>(test_dim_0);                                                  \
        auto test_dim_0_keep = []<typename T>(Device device) {                                  \
            std::mt19937 gen(0);                                                                \
            std::vector<T> d = rand_vec<T>(4 * 4, gen);                                         \
            std::vector<T> g = rand_vec<T>(4, gen);                                             \
            Tensor x1(d, {4, 4}, device, true);                                                 \
            Tensor input_grad_x(g, {1, 4}, device, true);                                       \
            Tensor x2 = FUNC(x1, 0, true);                                                      \
            x2.backward(input_grad_x);                                                          \
            const auto options = torch::TensorOptions().dtype(torch::kFloat);                   \
            torch::Tensor t1 = torch::from_blob(d.data(), {4, 4}, options.requires_grad(true)); \
            torch::Tensor input_grad_t = torch::from_blob(g.data(), {1, 4}, options);           \
            torch::Tensor t2 = TORCH_FUNC(t1, 0, true);                                         \
            t2.backward(input_grad_t);                                                          \
            torch::Tensor expected_grad = t1.grad();                                            \
            std::vector<T> expected_grad_data(                                                  \
                expected_grad.data_ptr<T>(),                                                    \
                expected_grad.data_ptr<T>() + expected_grad.numel()                             \
            );                                                                                  \
            Tensor expected(expected_grad_data, {4, 4}, device);                                \
            auto close_options = CloseOptions().atol(1e-6).equal_nan();                         \
            CHECK(allclose(x1.grad().value(), expected, close_options));                        \
        };                                                                                      \
        runner_single_type<float>(test_dim_0_keep);                                             \
        auto test_dim_1 = []<typename T>(Device device) {                                       \
            std::mt19937 gen(0);                                                                \
            std::vector<T> d = rand_vec<T>(4 * 4, gen);                                         \
            std::vector<T> g = rand_vec<T>(4, gen);                                             \
            Tensor x1(d, {4, 4}, device, true);                                                 \
            Tensor input_grad_x(g, {4}, device, true);                                          \
            Tensor x2 = FUNC(x1, 1, false);                                                     \
            x2.backward(input_grad_x);                                                          \
            const auto options = torch::TensorOptions().dtype(torch::kFloat);                   \
            torch::Tensor t1 = torch::from_blob(d.data(), {4, 4}, options.requires_grad(true)); \
            torch::Tensor input_grad_t = torch::from_blob(g.data(), {4}, options);              \
            torch::Tensor t2 = TORCH_FUNC(t1, 1, false);                                        \
            t2.backward(input_grad_t);                                                          \
            torch::Tensor expected_grad = t1.grad();                                            \
            std::vector<T> expected_grad_data(                                                  \
                expected_grad.data_ptr<T>(),                                                    \
                expected_grad.data_ptr<T>() + expected_grad.numel()                             \
            );                                                                                  \
            Tensor expected(expected_grad_data, {4, 4}, device);                                \
            auto close_options = CloseOptions().atol(1e-6).equal_nan();                         \
            CHECK(allclose(x1.grad().value(), expected, close_options));                        \
        };                                                                                      \
        runner_single_type<float>(test_dim_1);                                                  \
        auto test_dim_1_keep = []<typename T>(Device device) {                                  \
            std::mt19937 gen(0);                                                                \
            std::vector<T> d = rand_vec<T>(4 * 4, gen);                                         \
            std::vector<T> g = rand_vec<T>(4, gen);                                             \
            Tensor x1(d, {4, 4}, device, true);                                                 \
            Tensor input_grad_x(g, {4, 1}, device, true);                                       \
            Tensor x2 = FUNC(x1, 1, true);                                                      \
            x2.backward(input_grad_x);                                                          \
            const auto options = torch::TensorOptions().dtype(torch::kFloat);                   \
            torch::Tensor t1 = torch::from_blob(d.data(), {4, 4}, options.requires_grad(true)); \
            torch::Tensor input_grad_t = torch::from_blob(g.data(), {4, 1}, options);           \
            torch::Tensor t2 = TORCH_FUNC(t1, 1, true);                                         \
            t2.backward(input_grad_t);                                                          \
            torch::Tensor expected_grad = t1.grad();                                            \
            std::vector<T> expected_grad_data(                                                  \
                expected_grad.data_ptr<T>(),                                                    \
                expected_grad.data_ptr<T>() + expected_grad.numel()                             \
            );                                                                                  \
            Tensor expected(expected_grad_data, {4, 4}, device);                                \
            auto close_options = CloseOptions().atol(1e-6).equal_nan();                         \
            CHECK(allclose(x1.grad().value(), expected, close_options));                        \
        };                                                                                      \
        runner_single_type<float>(test_dim_1_keep);                                             \
    }

// NOLINTBEGIN
REDUCE_AUTOGRAD_TEST_CASE("min", min, torch::amin);
REDUCE_AUTOGRAD_TEST_CASE("max", max, torch::amax);
REDUCE_AUTOGRAD_TEST_CASE("sum", sum, torch::sum);
REDUCE_AUTOGRAD_TEST_CASE("mean", mean, torch::mean);

// NOLINTEND

#endif
