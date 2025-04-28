// test_binary_autograd.cpp
// Test the binary autograd ops

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
#define BINARY_AUTOGRAD_TEST_CASE(NAME, FUNC, TORCH_FUNC)                                                \
    TEST_CASE("Binary " NAME " autograd") {                                                              \
        auto test = []<typename T>(Device device) {                                                      \
            std::mt19937 gen(0);                                                                         \
            std::vector<T> d_lhs = rand_vec<T>(4 * 4, gen);                                              \
            std::vector<T> d_rhs = rand_vec<T>(4 * 4, gen);                                              \
            std::vector<T> grad_out = rand_vec<T>(4 * 4, gen);                                           \
            Tensor x_lhs(d_lhs, {4, 4}, device, true);                                                   \
            Tensor x_rhs(d_rhs, {4, 4}, device, true);                                                   \
            Tensor output_grad(grad_out, {4, 4}, device, true);                                          \
            Tensor x2 = FUNC(x_lhs, x_rhs);                                                              \
            x2.backward(output_grad);                                                                    \
            const auto options_f = torch::TensorOptions().dtype(torch::kFloat);                          \
            torch::Tensor t_lhs = torch::from_blob(d_lhs.data(), {4, 4}, options_f.requires_grad(true)); \
            torch::Tensor t_rhs = torch::from_blob(d_rhs.data(), {4, 4}, options_f.requires_grad(true)); \
            torch::Tensor t_grad = torch::from_blob(grad_out.data(), {4, 4}, options_f);                 \
            torch::Tensor t2 = TORCH_FUNC(t_lhs, t_rhs);                                                 \
            t2.backward(t_grad);                                                                         \
            torch::Tensor expected_grad_lhs = t_lhs.grad();                                              \
            std::vector<T> expected_grad_lhs_data(                                                       \
                expected_grad_lhs.data_ptr<T>(),                                                         \
                expected_grad_lhs.data_ptr<T>() + expected_grad_lhs.numel()                              \
            );                                                                                           \
            torch::Tensor expected_grad_rhs = t_rhs.grad();                                              \
            std::vector<T> expected_grad_rhs_data(                                                       \
                expected_grad_rhs.data_ptr<T>(),                                                         \
                expected_grad_rhs.data_ptr<T>() + expected_grad_rhs.numel()                              \
            );                                                                                           \
            Tensor expected_lhs(expected_grad_lhs_data, {4, 4}, device);                                 \
            Tensor expected_rhs(expected_grad_rhs_data, {4, 4}, device);                                 \
            auto close_options = CloseOptions().atol(1e-4).equal_nan();                                  \
            CHECK(allclose(x_lhs.grad().value(), expected_lhs, close_options));                          \
            CHECK(allclose(x_rhs.grad().value(), expected_rhs, close_options));                          \
        };                                                                                               \
        runner_single_type<float>(test);                                                                 \
    }

// NOLINTBEGIN
BINARY_AUTOGRAD_TEST_CASE("add", add, torch::add);
BINARY_AUTOGRAD_TEST_CASE("sub", sub, torch::sub);
BINARY_AUTOGRAD_TEST_CASE("mul", mul, torch::mul);
BINARY_AUTOGRAD_TEST_CASE("div", div, torch::div);
BINARY_AUTOGRAD_TEST_CASE("pow", pow, torch::pow);
// NOLINTEND

#endif
