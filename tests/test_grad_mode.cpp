// test_grad_mode.cpp
// Test the grad mode autograd functionality

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <tt/device.h>
#include <tt/exception.h>
#include <tt/grad_mode.h>
#include <tt/scalar.h>
#include <tt/tensor.h>

#include "doctest.h"
#include "test_util.h"

#include <cstddef>
#include <optional>
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
TEST_CASE("Grad mode autograd") {
    auto test_grad_false = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> d = rand_vec<T>(4 * 4, gen);
        std::vector<T> g = rand_vec<T>(4 * 4, gen);

        Tensor x1(d, {4, 4}, device, true);
        std::optional<Tensor> result;
        {
            autograd::NoGradGuard guard;
            result = x1.clone();
        }
        CHECK_FALSE(result.value().requires_grad());
    };
    runner_single_type<float>(test_grad_false);

    auto test_grad_true = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> d = rand_vec<T>(4 * 4, gen);
        std::vector<T> g = rand_vec<T>(4 * 4, gen);

        Tensor x1(d, {4, 4}, device, true);
        std::optional<Tensor> result;
        {
            result = x1.clone();
        }
        CHECK(result.value().requires_grad());
    };
    runner_single_type<float>(test_grad_true);
}
