// test_creation.cpp
// Test the creation functions

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <tt/device.h>
#include <tt/exception.h>
#include <tt/scalar.h>
#include <tt/shape.h>
#include <tt/tensor.h>

#include "doctest.h"
#include "test_util.h"

#include <cmath>
#include <tuple>
#include <type_traits>
#include <vector>

using namespace tinytensor;

// NOLINTNEXTLINE
TEST_CASE("Creation full") {
    auto test_full_bool_scalar = []<typename T>(Device device) {
        bool value = true;
        std::vector<bool> expected_values(4 * 3, value);
        Tensor tensor = full(Scalar(value), {4, 3}, device);
        Tensor expected(expected_values, {4, 3}, device);
        CHECK(allclose(tensor, expected));
    };
    runner_boolean(test_full_bool_scalar);
    auto test_full_scalar = []<typename T>(Device device) {
        T value = 2;
        std::vector<T> expected_values(4 * 3, value);
        Tensor tensor = full(Scalar(value), {4, 3}, device);
        Tensor expected(expected_values, {4, 3}, device);
        CHECK(allclose(tensor, expected));
    };
    runner_all_except_bool(test_full_scalar);

    auto test_full_bool_value = []<typename T>(Device device) {
        std::vector<bool> expected_values(4 * 3, true);
        Tensor tensor = full(true, {4, 3}, to_scalar<T>::type, device);
        Tensor expected(expected_values, {4, 3}, device);
        CHECK(allclose(tensor, expected));
    };
    runner_boolean(test_full_bool_value);

    auto test_full_value1 = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, to_ctype_t<kDefaultFloat>, to_ctype_t<kDefaultInt>>;
        R value = 2;
        std::vector<T> expected_values(4 * 3, static_cast<T>(value));
        Tensor tensor = full(value, {4, 3}, to_scalar<T>::type, device);
        Tensor expected(expected_values, {4, 3}, device);
        CHECK(allclose(tensor, expected));
    };
    runner_all_except_bool(test_full_value1);

    auto test_full_value2 = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, to_ctype_t<kDefaultFloat>, to_ctype_t<kDefaultInt>>;
        R value = 2;
        std::vector<T> expected_values(4 * 3, static_cast<T>(value));
        Tensor tensor = full(value, {4, 3}, TensorOptions().dtype(to_scalar<T>::type).device(device));
        Tensor expected(expected_values, {4, 3}, device);
        CHECK(allclose(tensor, expected));
    };
    runner_all_except_bool(test_full_value2);
}

// NOLINTNEXTLINE
TEST_CASE("Creation zeros") {
    auto test_zeros_bool = []<typename T>(Device device) {
        std::vector<T> expected_values(4 * 3, false);
        Tensor tensor = zeros({4, 3}, kBool, device);
        Tensor expected(expected_values, {4, 3}, device);
        CHECK(allclose(tensor, expected));
    };
    runner_boolean(test_zeros_bool);

    auto test_zeros1 = []<typename T>(Device device) {
        std::vector<T> expected_values(4 * 3, 0);
        Tensor tensor = zeros({4, 3}, to_scalar<T>::type, device);
        Tensor expected(expected_values, {4, 3}, device);
        CHECK(allclose(tensor, expected));
    };
    runner_all_except_bool(test_zeros1);

    auto test_zeros2 = []<typename T>(Device device) {
        std::vector<T> expected_values(4 * 3, 0);
        Tensor tensor = zeros({4, 3}, TensorOptions().dtype(to_scalar<T>::type).device(device));
        Tensor expected(expected_values, {4, 3}, device);
        CHECK(allclose(tensor, expected));
    };
    runner_all_except_bool(test_zeros2);

    auto test_zeros3 = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, to_ctype_t<kDefaultFloat>, to_ctype_t<kDefaultInt>>;
        std::vector<R> expected_values(4 * 3, 0);
        Tensor tensor_temp = ones({4, 3}, TensorOptions().dtype(to_scalar<R>::type).device(device));
        Tensor tensor = zeros_like(tensor_temp);
        Tensor expected(expected_values, {4, 3}, device);
        CHECK(allclose(tensor, expected));
    };
    runner_all_except_bool(test_zeros3);
}

// NOLINTNEXTLINE
TEST_CASE("Creation ones") {
    auto test_ones_bool = []<typename T>(Device device) {
        std::vector<T> expected_values(4 * 3, true);
        Tensor tensor = ones({4, 3}, kBool, device);
        Tensor expected(expected_values, {4, 3}, device);
        CHECK(allclose(tensor, expected));
    };
    runner_boolean(test_ones_bool);

    auto test_ones1 = []<typename T>(Device device) {
        std::vector<T> expected_values(4 * 3, 1);
        Tensor tensor = ones({4, 3}, to_scalar<T>::type, device);
        Tensor expected(expected_values, {4, 3}, device);
        CHECK(allclose(tensor, expected));
    };
    runner_all_except_bool(test_ones1);

    auto test_ones2 = []<typename T>(Device device) {
        std::vector<T> expected_values(4 * 3, 1);
        Tensor tensor = ones({4, 3}, TensorOptions().dtype(to_scalar<T>::type).device(device));
        Tensor expected(expected_values, {4, 3}, device);
        CHECK(allclose(tensor, expected));
    };
    runner_all_except_bool(test_ones2);

    auto test_ones3 = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, to_ctype_t<kDefaultFloat>, to_ctype_t<kDefaultInt>>;
        std::vector<R> expected_values(4 * 3, 1);
        Tensor tensor_temp = zeros({4, 3}, TensorOptions().dtype(to_scalar<R>::type).device(device));
        Tensor tensor = ones_like(tensor_temp);
        Tensor expected(expected_values, {4, 3}, device);
        CHECK(allclose(tensor, expected));
    };
    runner_all_except_bool(test_ones3);
}

// NOLINTNEXTLINE
TEST_CASE("Creation arange") {
    auto test_arange_bool = []<typename T>(Device device) {
        CHECK_THROWS_AS(std::ignore = arange({2, 2}, kBool, device), TTException);
    };
    runner_all_except_bool(test_arange_bool);

    auto test_arange1 = []<typename T>(Device device) {
        std::vector<T> expected_values;
        for (int i = 0; i < 4 * 3; ++i) {
            expected_values.push_back(static_cast<T>(i));
        }
        Tensor tensor = arange({4, 3}, to_scalar<T>::type, device);
        Tensor expected(expected_values, {4, 3}, device);
        CHECK(allclose(tensor, expected));
    };
    runner_all_except_bool(test_arange1);

    auto test_arange2 = []<typename T>(Device device) {
        std::vector<T> expected_values;
        for (int i = 0; i < 4 * 3; ++i) {
            expected_values.push_back(static_cast<T>(i));
        }
        Tensor tensor = arange({4, 3}, TensorOptions().dtype(to_scalar<T>::type).device(device));
        Tensor expected(expected_values, {4, 3}, device);
        CHECK(allclose(tensor, expected));
    };
    runner_all_except_bool(test_arange2);
}

// NOLINTNEXTLINE
TEST_CASE("Creation eye") {
    auto test_eye_bool = []<typename T>(Device device) {
        CHECK_THROWS_AS(std::ignore = eye(5, 3, kBool, device), TTException);
    };
    runner_all_except_bool(test_eye_bool);

    auto test_eye1 = []<typename T>(Device device) {
        std::vector<T> expected_values = {1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0};
        Tensor tensor = eye(5, 3, to_scalar<T>::type, device);
        Tensor expected(expected_values, {5, 3}, device);
        CHECK(allclose(tensor, expected));
    };
    runner_all_except_bool(test_eye1);

    auto test_eye2 = []<typename T>(Device device) {
        std::vector<T> expected_values = {1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0};
        Tensor tensor = eye(3, 5, to_scalar<T>::type, device);
        Tensor expected(expected_values, {3, 5}, device);
        CHECK(allclose(tensor, expected));
    };
    runner_all_except_bool(test_eye2);
}

// NOLINTNEXTLINE
TEST_CASE("Creation one_hot") {
    auto test_one_hot1 = []<typename T>(Device device) {
        std::vector<T> expected_values = {1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0};
        Tensor indices({0, 1, 2, 0, 1}, device);
        Tensor tensor = one_hot(indices, -1);
        Tensor expected(expected_values, {5, 3}, device);
        CHECK(allclose(tensor, expected));
    };
    runner_single_type<int>(test_one_hot1);

    auto test_one_hot2 = []<typename T>(Device device) {
        std::vector<T> expected_values = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0};
        Tensor indices({0, 1, 2, 0, 1}, device);
        Tensor tensor = one_hot(indices, 4);
        Tensor expected(expected_values, {5, 4}, device);
        CHECK(allclose(tensor, expected));
    };
    runner_single_type<int>(test_one_hot2);
}

// NOLINTNEXTLINE
TEST_CASE("Creation vec") {
    auto test_vec1 = []<typename T>(Device device) {
        std::vector<T> expected_values = {1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0};
        Tensor expected(expected_values, device);
        Tensor tensor(std::move(expected_values), device);
        CHECK(allclose(tensor, expected));
    };
    runner_all(test_vec1);

    auto test_vec2 = []<typename T>(Device device) {
        std::vector<T> expected_values = {1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0};
        Tensor expected(expected_values, {5, 3}, device);
        Tensor tensor(std::move(expected_values), {5, 3}, device);
        CHECK(allclose(tensor, expected));
    };
    runner_all(test_vec2);
}
