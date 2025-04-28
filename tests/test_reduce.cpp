// test_reduce.cpp
// Test the reduce ops

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <tt/device.h>
#include <tt/exception.h>
#include <tt/scalar.h>
#include <tt/shape.h>
#include <tt/tensor.h>

#include "doctest.h"
#include "test_util.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <random>
#include <tuple>
#include <type_traits>
#include <vector>

using namespace tinytensor;

namespace {
template <typename T>
auto init_rnd_vec(std::size_t n) -> std::vector<T> {
    using UniformDistT = std::
        conditional_t<std::is_floating_point_v<T>, std::uniform_real_distribution<T>, std::uniform_int_distribution<T>>;
    std::vector<T> values;
    UniformDistT dist(std::numeric_limits<T>::lowest() / 1000, std::numeric_limits<T>::max() / 1000);
    std::mt19937 gen(0);
    for (std::size_t i = 0; i < n; ++i) {
        values.push_back(dist(gen));
    }
    return values;
}
}    // namespace

// NOLINTNEXTLINE
TEST_CASE("Reduce min") {
    auto test_min_rand = []<typename T>(Device device) {
        std::vector<T> input_values = init_rnd_vec<T>(4 * 2048);
        T min_value = *std::ranges::min_element(input_values);
        std::vector<T> expected_values = {min_value};
        Tensor input(input_values, {4, 2048}, device);
        Tensor expected(expected_values, {1}, device);
        const auto res = min(input);
        CHECK_EQ(res.item<T>(), min_value);
        CHECK(allclose(res, expected));
        CHECK(allclose(input.min(), expected));
    };
    runner_all_except_bool(test_min_rand);

    auto test_min_idx0 = []<typename T>(Device device) {
        Tensor input = arange({2, 3, 4}, to_scalar<T>::type, device).reshape({2, 3, 4});
        std::vector<T> expected_values = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        Tensor expected(expected_values, {3, 4}, device);
        Tensor expected_keepdim(expected_values, {1, 3, 4}, device);
        CHECK(allclose(min(input, 0), expected));
        CHECK(allclose(input.min(0), expected));
        CHECK(allclose(min(input, -3), expected));
        CHECK(allclose(input.min(-3), expected));
        CHECK(allclose(min(input, 0, true), expected_keepdim));
        CHECK(allclose(input.min(0, true), expected_keepdim));
    };
    runner_all_except_bool(test_min_idx0);

    auto test_min_idx1 = []<typename T>(Device device) {
        Tensor input = arange({2, 3, 4}, to_scalar<T>::type, device).reshape({2, 3, 4});
        std::vector<T> expected_values = {0, 1, 2, 3, 12, 13, 14, 15};
        Tensor expected(expected_values, {2, 4}, device);
        Tensor expected_keepdim(expected_values, {2, 1, 4}, device);
        CHECK(allclose(min(input, 1), expected));
        CHECK(allclose(input.min(1), expected));
        CHECK(allclose(min(input, -2), expected));
        CHECK(allclose(input.min(-2), expected));
        CHECK(allclose(min(input, 1, true), expected_keepdim));
        CHECK(allclose(input.min(1, true), expected_keepdim));
    };
    runner_all_except_bool(test_min_idx1);

    auto test_min_idx2 = []<typename T>(Device device) {
        Tensor input = arange({2, 3, 4}, to_scalar<T>::type, device).reshape({2, 3, 4});
        std::vector<T> expected_values = {0, 4, 8, 12, 16, 20};
        Tensor expected(expected_values, {2, 3}, device);
        Tensor expected_keepdim(expected_values, {2, 3, 1}, device);
        CHECK(allclose(min(input, 2), expected));
        CHECK(allclose(input.min(2), expected));
        CHECK(allclose(min(input, -1), expected));
        CHECK(allclose(input.min(-1), expected));
        CHECK(allclose(min(input, 2, true), expected_keepdim));
        CHECK(allclose(input.min(2, true), expected_keepdim));
    };
    runner_all_except_bool(test_min_idx2);

    auto test_min_bounds = []<typename T>(Device device) {
        Tensor input = arange({2, 3, 4}, to_scalar<T>::type, device).reshape({2, 3, 4});
        CHECK_THROWS_AS(std::ignore = min(input, 3), TTException);
        CHECK_THROWS_AS(std::ignore = input.min(3), TTException);
        CHECK_THROWS_AS(std::ignore = min(input, -4), TTException);
        CHECK_THROWS_AS(std::ignore = input.min(-4), TTException);
    };
    runner_all_except_bool(test_min_bounds);
}

// NOLINTNEXTLINE
TEST_CASE("Reduce argmin") {
    auto test_argmin = []<typename T>(Device device) {
        std::vector<T> values = {21, 11, 14, 23, 10, 6, 12, 13, 7, 22, 2, 5, 17, 1, 18, 19, 16, 20, 15, 3, 9, 0, 8, 4};
        Tensor input(values, {2, 3, 4}, device);
        std::vector<int> expected_values = {21};
        Tensor expected(expected_values, {1}, device);
        CHECK(allclose(argmin(input), expected));
        CHECK(allclose(input.argmin(), expected));
    };
    runner_all_except_bool(test_argmin);

    auto test_argmin_idx0 = []<typename T>(Device device) {
        std::vector<T> values = {21, 11, 14, 23, 10, 6, 12, 13, 7, 22, 2, 5, 17, 1, 18, 19, 16, 20, 15, 3, 9, 0, 8, 4};
        Tensor input(values, {2, 3, 4}, device);
        std::vector<int> expected_values = {1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1};
        Tensor expected(expected_values, {3, 4}, device);
        Tensor expected_keepdim(expected_values, {1, 3, 4}, device);
        CHECK(allclose(argmin(input, 0), expected));
        CHECK(allclose(input.argmin(0), expected));
        CHECK(allclose(argmin(input, -3), expected));
        CHECK(allclose(input.argmin(-3), expected));
        CHECK(allclose(argmin(input, 0, true), expected_keepdim));
        CHECK(allclose(input.argmin(0, true), expected_keepdim));
    };
    runner_all_except_bool(test_argmin_idx0);

    auto test_argmin_idx1 = []<typename T>(Device device) {
        std::vector<T> values = {21, 11, 14, 23, 10, 6, 12, 13, 7, 22, 2, 5, 17, 1, 18, 19, 16, 20, 15, 3, 9, 0, 8, 4};
        Tensor input(values, {2, 3, 4}, device);
        std::vector<int> expected_values = {2, 1, 2, 2, 2, 2, 2, 1};
        Tensor expected(expected_values, {2, 4}, device);
        Tensor expected_keepdim(expected_values, {2, 1, 4}, device);
        CHECK(allclose(argmin(input, 1), expected));
        CHECK(allclose(input.argmin(1), expected));
        CHECK(allclose(argmin(input, -2), expected));
        CHECK(allclose(input.argmin(-2), expected));
        CHECK(allclose(argmin(input, 1, true), expected_keepdim));
        CHECK(allclose(input.argmin(1, true), expected_keepdim));
    };
    runner_all_except_bool(test_argmin_idx1);

    auto test_argmin_idx2 = []<typename T>(Device device) {
        std::vector<T> values = {21, 11, 14, 23, 10, 6, 12, 13, 7, 22, 2, 5, 17, 1, 18, 19, 16, 20, 15, 3, 9, 0, 8, 4};
        Tensor input(values, {2, 3, 4}, device);
        std::vector<int> expected_values = {1, 1, 2, 1, 3, 1};
        Tensor expected(expected_values, {2, 3}, device);
        Tensor expected_keepdim(expected_values, {2, 3, 1}, device);
        CHECK(allclose(argmin(input, 2), expected));
        CHECK(allclose(input.argmin(2), expected));
        CHECK(allclose(argmin(input, -1), expected));
        CHECK(allclose(input.argmin(-1), expected));
        CHECK(allclose(argmin(input, 2, true), expected_keepdim));
        CHECK(allclose(input.argmin(2, true), expected_keepdim));
    };
    runner_all_except_bool(test_argmin_idx2);

    auto test_argmin_bounds = []<typename T>(Device device) {
        Tensor input = arange({2, 3, 4}, to_scalar<T>::type, device).reshape({2, 3, 4});
        CHECK_THROWS_AS(std::ignore = argmin(input, 3), TTException);
        CHECK_THROWS_AS(std::ignore = input.argmin(3), TTException);
        CHECK_THROWS_AS(std::ignore = argmin(input, -4), TTException);
        CHECK_THROWS_AS(std::ignore = input.argmin(-4), TTException);
    };
    runner_all_except_bool(test_argmin_bounds);
}

// NOLINTNEXTLINE
TEST_CASE("Reduce max") {
    auto test_max_rand = []<typename T>(Device device) {
        std::vector<T> input_values = init_rnd_vec<T>(4 * 2048);
        T max_value = *std::ranges::max_element(input_values);
        std::vector<T> expected_values = {max_value};
        Tensor input(input_values, {4, 2048}, device);
        Tensor expected(expected_values, {1}, device);
        const auto res = max(input);
        CHECK_EQ(res.item<T>(), max_value);
        CHECK(allclose(res, expected));
        CHECK(allclose(input.max(), expected));
    };
    runner_all_except_bool(test_max_rand);

    auto test_max_idx0 = []<typename T>(Device device) {
        Tensor input = arange({2, 3, 4}, to_scalar<T>::type, device).reshape({2, 3, 4});
        std::vector<T> expected_values = {12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
        Tensor expected(expected_values, {3, 4}, device);
        Tensor expected_keepdim(expected_values, {1, 3, 4}, device);
        CHECK(allclose(max(input, 0), expected));
        CHECK(allclose(input.max(0), expected));
        CHECK(allclose(max(input, -3), expected));
        CHECK(allclose(input.max(-3), expected));
        CHECK(allclose(max(input, 0, true), expected_keepdim));
        CHECK(allclose(input.max(0, true), expected_keepdim));
    };
    runner_all_except_bool(test_max_idx0);

    auto test_max_idx1 = []<typename T>(Device device) {
        Tensor input = arange({2, 3, 4}, to_scalar<T>::type, device).reshape({2, 3, 4});
        std::vector<T> expected_values = {8, 9, 10, 11, 20, 21, 22, 23};
        Tensor expected(expected_values, {2, 4}, device);
        Tensor expected_keepdim(expected_values, {2, 1, 4}, device);
        CHECK(allclose(max(input, 1), expected));
        CHECK(allclose(input.max(1), expected));
        CHECK(allclose(max(input, -2), expected));
        CHECK(allclose(input.max(-2), expected));
        CHECK(allclose(max(input, 1, true), expected_keepdim));
        CHECK(allclose(input.max(1, true), expected_keepdim));
    };
    runner_all_except_bool(test_max_idx1);

    auto test_max_idx2 = []<typename T>(Device device) {
        Tensor input = arange({2, 3, 4}, to_scalar<T>::type, device).reshape({2, 3, 4});
        std::vector<T> expected_values = {3, 7, 11, 15, 19, 23};
        Tensor expected(expected_values, {2, 3}, device);
        Tensor expected_keepdim(expected_values, {2, 3, 1}, device);
        CHECK(allclose(max(input, 2), expected));
        CHECK(allclose(input.max(2), expected));
        CHECK(allclose(max(input, -1), expected));
        CHECK(allclose(input.max(-1), expected));
        CHECK(allclose(max(input, 2, true), expected_keepdim));
        CHECK(allclose(input.max(2, true), expected_keepdim));
    };
    runner_all_except_bool(test_max_idx2);

    auto test_max_bounds = []<typename T>(Device device) {
        Tensor input = arange({2, 3, 4}, to_scalar<T>::type, device).reshape({2, 3, 4});
        CHECK_THROWS_AS(std::ignore = max(input, 3), TTException);
        CHECK_THROWS_AS(std::ignore = input.max(3), TTException);
        CHECK_THROWS_AS(std::ignore = max(input, -4), TTException);
        CHECK_THROWS_AS(std::ignore = input.max(-4), TTException);
    };
    runner_all_except_bool(test_max_bounds);
}

// NOLINTNEXTLINE
TEST_CASE("Reduce argmax") {
    auto test_argmax = []<typename T>(Device device) {
        std::vector<T> values = {21, 11, 14, 23, 10, 6, 12, 13, 7, 22, 2, 5, 17, 1, 18, 19, 16, 20, 15, 3, 9, 0, 8, 4};
        Tensor input(values, {2, 3, 4}, device);
        std::vector<int> expected_values = {3};
        Tensor expected(expected_values, {1}, device);
        CHECK(allclose(argmax(input), expected));
        CHECK(allclose(input.argmax(), expected));
    };
    runner_all_except_bool(test_argmax);

    auto test_argmax_idx0 = []<typename T>(Device device) {
        std::vector<T> values = {21, 11, 14, 23, 10, 6, 12, 13, 7, 22, 2, 5, 17, 1, 18, 19, 16, 20, 15, 3, 9, 0, 8, 4};
        Tensor input(values, {2, 3, 4}, device);
        std::vector<int> expected_values = {0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0};
        Tensor expected(expected_values, {3, 4}, device);
        Tensor expected_keepdim(expected_values, {1, 3, 4}, device);
        CHECK(allclose(argmax(input, 0), expected));
        CHECK(allclose(input.argmax(0), expected));
        CHECK(allclose(argmax(input, -3), expected));
        CHECK(allclose(input.argmax(-3), expected));
        CHECK(allclose(argmax(input, 0, true), expected_keepdim));
        CHECK(allclose(input.argmax(0, true), expected_keepdim));
    };
    runner_all_except_bool(test_argmax_idx0);

    auto test_argmax_idx1 = []<typename T>(Device device) {
        std::vector<T> values = {21, 11, 14, 23, 10, 6, 12, 13, 7, 22, 2, 5, 17, 1, 18, 19, 16, 20, 15, 3, 9, 0, 8, 4};
        Tensor input(values, {2, 3, 4}, device);
        std::vector<int> expected_values = {0, 2, 0, 0, 0, 1, 0, 0};
        Tensor expected(expected_values, {2, 4}, device);
        Tensor expected_keepdim(expected_values, {2, 1, 4}, device);
        CHECK(allclose(argmax(input, 1), expected));
        CHECK(allclose(input.argmax(1), expected));
        CHECK(allclose(argmax(input, -2), expected));
        CHECK(allclose(input.argmax(-2), expected));
        CHECK(allclose(argmax(input, 1, true), expected_keepdim));
        CHECK(allclose(input.argmax(1, true), expected_keepdim));
    };
    runner_all_except_bool(test_argmax_idx1);

    auto test_argmax_idx2 = []<typename T>(Device device) {
        std::vector<T> values = {21, 11, 14, 23, 10, 6, 12, 13, 7, 22, 2, 5, 17, 1, 18, 19, 16, 20, 15, 3, 9, 0, 8, 4};
        Tensor input(values, {2, 3, 4}, device);
        std::vector<int> expected_values = {3, 3, 1, 3, 1, 0};
        Tensor expected(expected_values, {2, 3}, device);
        Tensor expected_keepdim(expected_values, {2, 3, 1}, device);
        CHECK(allclose(argmax(input, 2), expected));
        CHECK(allclose(input.argmax(2), expected));
        CHECK(allclose(argmax(input, -1), expected));
        CHECK(allclose(input.argmax(-1), expected));
        CHECK(allclose(argmax(input, 2, true), expected_keepdim));
        CHECK(allclose(input.argmax(2, true), expected_keepdim));
    };
    runner_all_except_bool(test_argmax_idx2);

    auto test_argmax_bounds = []<typename T>(Device device) {
        Tensor input = arange({2, 3, 4}, to_scalar<T>::type, device).reshape({2, 3, 4});
        CHECK_THROWS_AS(std::ignore = argmax(input, 3), TTException);
        CHECK_THROWS_AS(std::ignore = input.argmax(3), TTException);
        CHECK_THROWS_AS(std::ignore = argmax(input, -4), TTException);
        CHECK_THROWS_AS(std::ignore = input.argmax(-4), TTException);
    };
    runner_all_except_bool(test_argmax_bounds);
}

// NOLINTNEXTLINE
TEST_CASE("Reduce sum") {
    auto test_sum_rand = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kI64>>;
        std::vector<T> input_values = init_rnd_vec<T>(4 * 2048);
        R sum_value = std::reduce(input_values.begin(), input_values.end());
        std::vector<R> expected_values = {sum_value};
        Tensor input(input_values, {4, 2048}, device);
        Tensor expected(expected_values, {1}, device);
        auto res = sum(input);
        // Sum can have floating point accumulation errors ...
        CHECK(allclose(res, expected));
        CHECK(allclose(input.sum(), expected));
    };
    runner_all_except_bool(test_sum_rand);

    auto test_sum_idx0 = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kI64>>;
        Tensor input = arange({2, 3, 4}, to_scalar<T>::type, device).reshape({2, 3, 4});
        std::vector<R> expected_values = {12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34};
        Tensor expected(expected_values, {3, 4}, device);
        Tensor expected_keepdim(expected_values, {1, 3, 4}, device);
        CHECK(allclose(sum(input, 0), expected));
        CHECK(allclose(input.sum(0), expected));
        CHECK(allclose(sum(input, -3), expected));
        CHECK(allclose(input.sum(-3), expected));
        CHECK(allclose(sum(input, 0, true), expected_keepdim));
        CHECK(allclose(input.sum(0, true), expected_keepdim));
    };
    runner_all_except_bool(test_sum_idx0);

    auto test_sum_idx1 = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kI64>>;
        Tensor input = arange({2, 3, 4}, to_scalar<T>::type, device).reshape({2, 3, 4});
        std::vector<R> expected_values = {12, 15, 18, 21, 48, 51, 54, 57};
        Tensor expected(expected_values, {2, 4}, device);
        Tensor expected_keepdim(expected_values, {2, 1, 4}, device);
        CHECK(allclose(sum(input, 1), expected));
        CHECK(allclose(input.sum(1), expected));
        CHECK(allclose(sum(input, -2), expected));
        CHECK(allclose(input.sum(-2), expected));
        CHECK(allclose(sum(input, 1, true), expected_keepdim));
        CHECK(allclose(input.sum(1, true), expected_keepdim));
    };
    runner_all_except_bool(test_sum_idx1);

    auto test_sum_idx2 = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kI64>>;
        Tensor input = arange({2, 3, 4}, to_scalar<T>::type, device).reshape({2, 3, 4});
        std::vector<R> expected_values = {6, 22, 38, 54, 70, 86};
        Tensor expected(expected_values, {2, 3}, device);
        Tensor expected_keepdim(expected_values, {2, 3, 1}, device);
        CHECK(allclose(sum(input, 2), expected));
        CHECK(allclose(input.sum(2), expected));
        CHECK(allclose(sum(input, -1), expected));
        CHECK(allclose(input.sum(-1), expected));
        CHECK(allclose(sum(input, 2, true), expected_keepdim));
        CHECK(allclose(input.sum(2, true), expected_keepdim));
    };
    runner_all_except_bool(test_sum_idx2);

    auto test_sum_bounds = []<typename T>(Device device) {
        Tensor input = arange({2, 3, 4}, to_scalar<T>::type, device).reshape({2, 3, 4});
        CHECK_THROWS_AS(std::ignore = sum(input, 3), TTException);
        CHECK_THROWS_AS(std::ignore = input.sum(3), TTException);
        CHECK_THROWS_AS(std::ignore = sum(input, -4), TTException);
        CHECK_THROWS_AS(std::ignore = input.sum(-4), TTException);
    };
    runner_all_except_bool(test_sum_bounds);
}

// NOLINTNEXTLINE
TEST_CASE("Reduce mean") {
    auto test_mean_idx0 = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        Tensor input = arange({2, 3, 4}, to_scalar<T>::type, device).reshape({2, 3, 4});
        std::vector<R> expected_values = {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
        Tensor expected(expected_values, {3, 4}, device);
        Tensor expected_keepdim(expected_values, {1, 3, 4}, device);
        CHECK(allclose(mean(input, 0), expected));
        CHECK(allclose(input.mean(0), expected));
        CHECK(allclose(mean(input, -3), expected));
        CHECK(allclose(input.mean(-3), expected));
        CHECK(allclose(mean(input, 0, true), expected_keepdim));
        CHECK(allclose(input.mean(0, true), expected_keepdim));
    };
    runner_all_except_bool(test_mean_idx0);

    auto test_mean_idx1 = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        Tensor input = arange({2, 3, 4}, to_scalar<T>::type, device).reshape({2, 3, 4});
        std::vector<R> expected_values = {4, 5, 6, 7, 16, 17, 18, 19};
        Tensor expected(expected_values, {2, 4}, device);
        Tensor expected_keepdim(expected_values, {2, 1, 4}, device);
        CHECK(allclose(mean(input, 1), expected));
        CHECK(allclose(input.mean(1), expected));
        CHECK(allclose(mean(input, -2), expected));
        CHECK(allclose(input.mean(-2), expected));
        CHECK(allclose(mean(input, 1, true), expected_keepdim));
        CHECK(allclose(input.mean(1, true), expected_keepdim));
    };
    runner_all_except_bool(test_mean_idx1);

    auto test_mean_idx2 = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        Tensor input = arange({2, 3, 4}, to_scalar<T>::type, device).reshape({2, 3, 4});
        std::vector<R> expected_values = {1.5, 5.5, 9.5, 13.5, 17.5, 21.5};
        Tensor expected(expected_values, {2, 3}, device);
        Tensor expected_keepdim(expected_values, {2, 3, 1}, device);
        CHECK(allclose(mean(input, 2), expected));
        CHECK(allclose(input.mean(2), expected));
        CHECK(allclose(mean(input, -1), expected));
        CHECK(allclose(input.mean(-1), expected));
        CHECK(allclose(mean(input, 2, true), expected_keepdim));
        CHECK(allclose(input.mean(2, true), expected_keepdim));
    };
    runner_all_except_bool(test_mean_idx2);

    auto test_mean_bounds = []<typename T>(Device device) {
        Tensor input = arange({2, 3, 4}, to_scalar<T>::type, device).reshape({2, 3, 4});
        CHECK_THROWS_AS(std::ignore = mean(input, 3), TTException);
        CHECK_THROWS_AS(std::ignore = input.mean(3), TTException);
        CHECK_THROWS_AS(std::ignore = mean(input, -4), TTException);
        CHECK_THROWS_AS(std::ignore = input.mean(-4), TTException);
    };
    runner_all_except_bool(test_mean_bounds);
}

// NOLINTNEXTLINE
TEST_CASE("Reduce all") {
    auto test_all_true = []<typename T>(Device device) {
        std::vector<T> input_values = std::vector<T>(4 * 2048, 1);
        Tensor input(input_values, {4, 2048}, device);
        auto res = all(input);
        CHECK(res == true);
        CHECK(input.all() == true);
    };
    runner_all_except_bool(test_all_true);
    auto test_all_false = []<typename T>(Device device) {
        std::vector<T> input_values = std::vector<T>(4 * 2048, 1);
        input_values[0] = 0;
        Tensor input(input_values, {4, 2048}, device);
        auto res = all(input);
        CHECK(res == false);
        CHECK(input.all() == false);
    };
    runner_all_except_bool(test_all_false);
}
