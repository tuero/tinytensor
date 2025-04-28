// test_to_autograd.cpp
// Test the dtype and device to autograd ops

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <tt/device.h>
#include <tt/exception.h>
#include <tt/index.h>
#include <tt/scalar.h>
#include <tt/tensor.h>

#include "doctest.h"
#include "test_util.h"

#include <cstddef>
#include <random>
#include <vector>

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
TEST_CASE("To dtype autograd") {
    auto test = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> d = rand_vec<T>(4 * 4, gen);
        std::vector<double> grad_out = rand_vec<double>(4 * 4, gen);
        std::vector<T> grad_in;
        for (const auto &v : grad_out) {
            grad_in.push_back(static_cast<T>(v));
        }

        Tensor x1(d, {4, 4}, device, true);
        Tensor input_grad(grad_in, {4, 4}, device, true);
        Tensor output_grad(grad_out, {4, 4}, device, true);
        Tensor x2 = x1.to(kF64);
        x2.backward(output_grad);

        auto close_options = CloseOptions().atol(1e-4).equal_nan();
        CHECK(allclose(x1.grad().value(), input_grad, close_options));
    };
    runner_single_type<float>(test);
}

// NOLINTNEXTLINE
TEST_CASE("To device CPU CPU autograd") {
    using T = float;
    Device CPU2{.backend = Backend::cpu, .id = 1};
    std::mt19937 gen(0);
    std::vector<T> d = rand_vec<T>(4 * 4, gen);
    std::vector<T> g = rand_vec<T>(4 * 4, gen);

    Tensor x1(d, {4, 4}, kCPU, true);
    Tensor grad_cpu(g, {4, 4}, kCPU, true);
    Tensor grad_cpu2(g, {4, 4}, CPU2, true);

    Tensor x2 = x1.to(CPU2);
    Tensor x3 = x2.to(kCPU);
    x3.backward(grad_cpu);

    auto close_options = CloseOptions().atol(1e-6).equal_nan();
    CHECK(allclose(x1.grad().value(), grad_cpu, close_options));
    CHECK(allclose(x2.grad().value(), grad_cpu2, close_options));
}

#ifdef TT_CUDA

// NOLINTNEXTLINE
TEST_CASE("To device CPU CUDA autograd") {
    using T = float;
    std::mt19937 gen(0);
    std::vector<T> d = rand_vec<T>(4 * 4, gen);
    std::vector<T> g = rand_vec<T>(4 * 4, gen);

    Tensor x1(d, {4, 4}, kCPU, true);
    Tensor grad_cpu(g, {4, 4}, kCPU, true);
    Tensor grad_cuda(g, {4, 4}, kCUDA, true);

    Tensor x2 = x1.to(kCUDA);
    Tensor x3 = x2.to(kCPU);
    x3.backward(grad_cpu);

    auto close_options = CloseOptions().atol(1e-6).equal_nan();
    CHECK(allclose(x1.grad().value(), grad_cpu, close_options));
    CHECK(allclose(x2.grad().value(), grad_cuda, close_options));
}

#endif
