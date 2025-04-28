// test_retain_graph.cpp
// Test the retain graph autograd functionality

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <tt/device.h>
#include <tt/exception.h>
#include <tt/grad_mode.h>
#include <tt/scalar.h>
#include <tt/tensor.h>

#include "doctest.h"
#include "test_util.h"

#include <optional>

using namespace tinytensor;

// NOLINTNEXTLINE
TEST_CASE("Retain graph autograd") {
    // Double backward fails if graph not retained
    auto test_false = []<typename T>(Device device) {
        Tensor g = uniform_real(0, 1, {4, 4}, TensorOptions().device(device));
        Tensor result = [&]() {
            Tensor x1 = uniform_real(0, 1, {4, 4}, TensorOptions().device(device).requires_grad(true));
            Tensor x2 = uniform_real(0, 1, {4, 4}, TensorOptions().device(device).requires_grad(true));
            return x1 + x2;
        }();
        result.backward(g);
        CHECK_THROWS_AS(result.backward(g), TTException);
    };
    runner_single_type<float>(test_false);

    auto test_true = []<typename T>(Device device) {
        Tensor g = uniform_real(0, 1, {4, 4}, TensorOptions().device(device));
        Tensor result = [&]() {
            Tensor x1 = uniform_real(0, 1, {4, 4}, TensorOptions().device(device).requires_grad(true));
            Tensor x2 = uniform_real(0, 1, {4, 4}, TensorOptions().device(device).requires_grad(true));
            return x1 + x2;
        }();
        result.backward(g, true);
        CHECK_NOTHROW(result.backward(g));
    };
    runner_single_type<float>(test_true);
}
