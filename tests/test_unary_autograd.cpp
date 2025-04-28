// test_unary_autograd.cpp
// Test the unary autograd ops

// Clang is noisy on hard-coded test data
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wimplicit-float-conversion"
#endif

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

// NOLINTNEXTLINE
TEST_CASE("Unary tgamma autograd") {
    // Torch doesn't define this, so manually test
    auto test = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> d = {
            0.488135,
            0.928446,
            2.15189,
            3.44266,
            1.02763,
            3.57946,
            0.448832,
            3.47252,
            -0.763452,
            1.23564,
            1.45894,
            -1.15618,
            -0.624128,
            -2.02465,
            3.91773,
            -4.43287
        };
        std::vector<T> g = {
            4.63663,
            -2.27344,
            -1.16558,
            -0.223349,
            2.91725,
            3.12169,
            0.288949,
            -0.200228,
            0.680445,
            -1.07215,
            4.25597,
            3.36079,
            -4.28964,
            -1.62604,
            -4.12871,
            1.48172,
        };
        std::vector<T> d_expected = {
            -17.0256,
            1.66933,
            -0.646413,
            -0.755734,
            -1.53027,
            12.7999,
            -1.27747,
            -0.706353,
            10.8307,
            0.23869,
            -0.00983109,
            133.212,
            -18.6042,
            1336.54,
            -27.5607,
            -0.229944
        };
        Tensor x1(d, {4, 4}, device, true);
        Tensor input_grad_x(g, {4, 4}, device, true);
        Tensor expected(d_expected, {4, 4}, device);
        Tensor x2 = tgamma(x1);
        x2.backward(input_grad_x);

        auto close_options = CloseOptions().atol(1e-4).equal_nan();
        CHECK(allclose(x1.grad().value(), expected, close_options));
    };
    runner_single_type<float>(test);
}

// NOLINTNEXTLINE
TEST_CASE("Unary clone autograd") {
    auto test = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> d = rand_vec<T>(4 * 4, gen);
        std::vector<T> g = rand_vec<T>(4 * 4, gen);

        Tensor x1(d, {4, 4}, device, true);
        Tensor expected(g, {4, 4}, device);

        Tensor x2 = x1.clone();
        Tensor input_grad_x(g, {4, 4}, device, true);
        x2.backward(input_grad_x);

        auto close_options = CloseOptions().atol(1e-4).equal_nan();
        CHECK(allclose(x1.grad().value(), expected, close_options));
    };
    runner_single_type<float>(test);
}

#ifdef TT_TORCH

// NOLINTBEGIN
#define SOFTMAX_0(x)           softmax(x, 0)
#define SOFTMAX_1(x)           softmax(x, 1)
#define LOG_SOFTMAX_0(x)       log_softmax(x, 0)
#define LOG_SOFTMAX_1(x)       log_softmax(x, 1)
#define SOFTMAX_TORCH_0(x)     torch::softmax(x, 0)
#define SOFTMAX_TORCH_1(x)     torch::softmax(x, 1)
#define LOG_SOFTMAX_TORCH_0(x) torch::nn::functional::log_softmax(x, 0)
#define LOG_SOFTMAX_TORCH_1(x) torch::nn::functional::log_softmax(x, 1)
// NOLINTEND

// NOLINTNEXTLINE
#define UNARY_AUTOGRAD_TEST_CASE(NAME, FUNC, TORCH_FUNC)                                        \
    TEST_CASE("Unary " NAME " autograd") {                                                      \
        auto test = []<typename T>(Device device) {                                             \
            std::mt19937 gen(0);                                                                \
            std::vector<T> d = rand_vec<T>(4 * 4, gen);                                         \
            std::vector<T> g = rand_vec<T>(4 * 4, gen);                                         \
            Tensor x1(d, {4, 4}, device, true);                                                 \
            Tensor input_grad_x(g, {4, 4}, device, true);                                       \
            Tensor x2 = FUNC(x1);                                                               \
            x2.backward(input_grad_x);                                                          \
            const auto options = torch::TensorOptions().dtype(torch::kFloat);                   \
            torch::Tensor t1 = torch::from_blob(d.data(), {4, 4}, options.requires_grad(true)); \
            torch::Tensor input_grad_t = torch::from_blob(g.data(), {4, 4}, options);           \
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
        runner_single_type<float>(test);                                                        \
    }

// NOLINTBEGIN
UNARY_AUTOGRAD_TEST_CASE("abs", abs, torch::abs);
UNARY_AUTOGRAD_TEST_CASE("sign", sign, torch::sign);
UNARY_AUTOGRAD_TEST_CASE("negate", negate, torch::negative);
UNARY_AUTOGRAD_TEST_CASE("log", log, torch::log);
UNARY_AUTOGRAD_TEST_CASE("log2", log2, torch::log2);
UNARY_AUTOGRAD_TEST_CASE("log10", log10, torch::log10);
UNARY_AUTOGRAD_TEST_CASE("log1p", log1p, torch::log1p);
UNARY_AUTOGRAD_TEST_CASE("exp", exp, torch::exp);
UNARY_AUTOGRAD_TEST_CASE("exp2", exp2, torch::exp2);
UNARY_AUTOGRAD_TEST_CASE("expm1", expm1, torch::expm1);
UNARY_AUTOGRAD_TEST_CASE("sqrt", sqrt, torch::sqrt);
UNARY_AUTOGRAD_TEST_CASE("sin", sin, torch::sin);
UNARY_AUTOGRAD_TEST_CASE("cos", cos, torch::cos);
UNARY_AUTOGRAD_TEST_CASE("tan", tan, torch::tan);
UNARY_AUTOGRAD_TEST_CASE("asin", asin, torch::asin);
UNARY_AUTOGRAD_TEST_CASE("acos", acos, torch::acos);
UNARY_AUTOGRAD_TEST_CASE("atan", atan, torch::atan);
UNARY_AUTOGRAD_TEST_CASE("sinh", sinh, torch::sinh);
UNARY_AUTOGRAD_TEST_CASE("cosh", cosh, torch::cosh);
UNARY_AUTOGRAD_TEST_CASE("tanh", tanh, torch::tanh);
UNARY_AUTOGRAD_TEST_CASE("asinh", asinh, torch::asinh);
UNARY_AUTOGRAD_TEST_CASE("acosh", acosh, torch::acosh);
UNARY_AUTOGRAD_TEST_CASE("atanh", atanh, torch::atanh);
UNARY_AUTOGRAD_TEST_CASE("erf", erf, torch::erf);
UNARY_AUTOGRAD_TEST_CASE("erfc", erfc, torch::erfc);
UNARY_AUTOGRAD_TEST_CASE("lgamma", lgamma, torch::lgamma);
UNARY_AUTOGRAD_TEST_CASE("sigmoid", sigmoid, torch::sigmoid);
UNARY_AUTOGRAD_TEST_CASE("log_sigmoid", log_sigmoid, torch::log_sigmoid);
UNARY_AUTOGRAD_TEST_CASE("hardsigmoid", hardsigmoid, torch::hardsigmoid);
UNARY_AUTOGRAD_TEST_CASE("softplus", softplus, torch::softplus);
UNARY_AUTOGRAD_TEST_CASE("relu", relu, torch::relu);
UNARY_AUTOGRAD_TEST_CASE("relu6", relu6, torch::relu6);
UNARY_AUTOGRAD_TEST_CASE("leaky_relu", leaky_relu, torch::leaky_relu);
UNARY_AUTOGRAD_TEST_CASE("elu", elu, torch::elu);
UNARY_AUTOGRAD_TEST_CASE("selu", selu, torch::selu);
UNARY_AUTOGRAD_TEST_CASE("silu", silu, torch::silu);
UNARY_AUTOGRAD_TEST_CASE("hardtanh", hardtanh, torch::hardtanh);
UNARY_AUTOGRAD_TEST_CASE("softsign", softsign, torch::nn::functional::softsign);
UNARY_AUTOGRAD_TEST_CASE("softmax0", SOFTMAX_0, SOFTMAX_TORCH_0);
UNARY_AUTOGRAD_TEST_CASE("softmax1", SOFTMAX_1, SOFTMAX_TORCH_1);
UNARY_AUTOGRAD_TEST_CASE("log_softmax0", LOG_SOFTMAX_0, LOG_SOFTMAX_TORCH_0);
UNARY_AUTOGRAD_TEST_CASE("log_softmax1", LOG_SOFTMAX_1, LOG_SOFTMAX_TORCH_1);
// NOLINTEND

#endif
