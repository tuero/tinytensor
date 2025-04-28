// test_concat.cpp
// Test concatenation related methods

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <tt/device.h>
#include <tt/exception.h>
#include <tt/index.h>
#include <tt/scalar.h>
#include <tt/shape.h>
#include <tt/tensor.h>

#include "doctest.h"
#include "test_util.h"

#include <cmath>
#include <tuple>
#include <vector>

using namespace tinytensor;

// NOLINTNEXTLINE
TEST_CASE("Shape concat") {
    auto test_concat1 = []<typename T>(Device device) {
        std::vector<T> v1;
        std::vector<T> v2;
        std::vector<T> v3;
        for (int i = 0; i < 12; ++i) {
            v1.push_back(static_cast<T>(i));
            v2.push_back(static_cast<T>(i + 12));
            v3.push_back(static_cast<T>(i + 24));
        }

        Tensor a1(v1, {3, 4}, device);
        Tensor a2(v2, {3, 4}, device);
        Tensor a3(v3, {3, 4}, device);
        Tensor expected = arange({9, 4}, TensorOptions().dtype(to_scalar<T>::type).device(device));

        CHECK(allclose(cat({a1, a2, a3}, 0), expected));
        CHECK(allclose(cat({a1, a2, a3}, -2), expected));
    };
    runner_all_except_bool(test_concat1);

    auto test_concat2 = []<typename T>(Device device) {
        std::vector<T> v1;
        std::vector<T> v2;
        std::vector<T> v3;
        for (int i = 0; i < 12; ++i) {
            v1.push_back(static_cast<T>(i));
            v2.push_back(static_cast<T>(i + 12));
            v3.push_back(static_cast<T>(i + 24));
        }
        std::vector<T> res = {0,  1,  2,  3,  12, 13, 14, 15, 24, 25, 26, 27, 4,  5,  6,  7,  16, 17,
                              18, 19, 28, 29, 30, 31, 8,  9,  10, 11, 20, 21, 22, 23, 32, 33, 34, 35};

        Tensor a1(v1, {3, 4}, device);
        Tensor a2(v2, {3, 4}, device);
        Tensor a3(v3, {3, 4}, device);
        Tensor expected(res, {3, 12}, device);

        CHECK(allclose(cat({a1, a2, a3}, 1), expected));
        CHECK(allclose(cat({a1, a2, a3}, -1), expected));
    };
    runner_all_except_bool(test_concat2);

    auto test_concat_unequal1 = []<typename T>(Device device) {
        std::vector<T> v1 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        std::vector<T> v2 = {12, 13, 14, 15};
        std::vector<T> v3 = {24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35};
        std::vector<T> res = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                              14, 15, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35};

        Tensor a1(v1, {3, 4}, device);
        Tensor a2(v2, {1, 4}, device);
        Tensor a3(v3, {3, 4}, device);
        Tensor expected(res, {7, 4}, device);

        CHECK(allclose(cat({a1, a2, a3}, 0), expected));
        CHECK(allclose(cat({a1, a2, a3}, -2), expected));
    };
    runner_all_except_bool(test_concat_unequal1);

    auto test_concat_unequal2 = []<typename T>(Device device) {
        std::vector<T> v1 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        std::vector<T> v2 = {12, 13, 14};
        std::vector<T> v3 = {24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35};
        std::vector<T> res = {0,  1,  2,  3,  12, 24, 25, 26, 27, 4,  5,  6,  7, 13,
                              28, 29, 30, 31, 8,  9,  10, 11, 14, 32, 33, 34, 35};

        Tensor a1(v1, {3, 4}, device);
        Tensor a2(v2, {3, 1}, device);
        Tensor a3(v3, {3, 4}, device);
        Tensor expected(res, {3, 9}, device);

        CHECK(allclose(cat({a1, a2, a3}, 1), expected));
        CHECK(allclose(cat({a1, a2, a3}, -1), expected));
    };
    runner_all_except_bool(test_concat_unequal2);

    auto test_concat_dim_error = []<typename T>(Device device) {
        const auto options = TensorOptions().dtype(to_scalar<T>::type).device(device);
        Tensor a1 = zeros({3, 4}, options);
        Tensor a2 = zeros({3, 1}, options);
        Tensor a3 = zeros({3, 4}, options);
        CHECK_THROWS_AS(std::ignore = cat({a1, a2, a3}, 2), TTException);
        CHECK_THROWS_AS(std::ignore = cat({a1, a2, a3}, -3), TTException);
    };
    runner_all_except_bool(test_concat_dim_error);
}

// NOLINTNEXTLINE
TEST_CASE("Shape stack") {
    auto test_stack1 = []<typename T>(Device device) {
        std::vector<T> v1;
        std::vector<T> v2;
        std::vector<T> v3;
        for (int i = 0; i < 12; ++i) {
            v1.push_back(static_cast<T>(i));
            v2.push_back(static_cast<T>(i + 12));
            v3.push_back(static_cast<T>(i + 24));
        }

        Tensor a1(v1, {3, 4}, device);
        Tensor a2(v2, {3, 4}, device);
        Tensor a3(v3, {3, 4}, device);
        Tensor expected = arange({3, 3, 4}, TensorOptions().dtype(to_scalar<T>::type).device(device));

        CHECK(allclose(stack({a1, a2, a3}, 0), expected));
        CHECK(allclose(stack({a1, a2, a3}, -3), expected));
    };
    runner_all_except_bool(test_stack1);

    auto test_stack2 = []<typename T>(Device device) {
        std::vector<T> v1;
        std::vector<T> v2;
        std::vector<T> v3;
        for (int i = 0; i < 12; ++i) {
            v1.push_back(static_cast<T>(i));
            v2.push_back(static_cast<T>(i + 12));
            v3.push_back(static_cast<T>(i + 24));
        }
        std::vector<T> res = {0,  1,  2,  3,  12, 13, 14, 15, 24, 25, 26, 27, 4,  5,  6,  7,  16, 17,
                              18, 19, 28, 29, 30, 31, 8,  9,  10, 11, 20, 21, 22, 23, 32, 33, 34, 35};

        Tensor a1(v1, {3, 4}, device);
        Tensor a2(v2, {3, 4}, device);
        Tensor a3(v3, {3, 4}, device);
        Tensor expected(res, {3, 3, 4}, device);

        CHECK(allclose(stack({a1, a2, a3}, 1), expected));
        CHECK(allclose(stack({a1, a2, a3}, -2), expected));
    };
    runner_all_except_bool(test_stack2);

    auto test_stack3 = []<typename T>(Device device) {
        std::vector<T> v1;
        std::vector<T> v2;
        std::vector<T> v3;
        std::vector<T> res;
        for (int i = 0; i < 12; ++i) {
            v1.push_back(static_cast<T>(i));
            v2.push_back(static_cast<T>(i + 12));
            v3.push_back(static_cast<T>(i + 24));
            res.push_back(static_cast<T>(i));
            res.push_back(static_cast<T>(i + 12));
            res.push_back(static_cast<T>(i + 24));
        }

        Tensor a1(v1, {3, 4}, device);
        Tensor a2(v2, {3, 4}, device);
        Tensor a3(v3, {3, 4}, device);
        Tensor expected(res, {3, 4, 3}, device);

        CHECK(allclose(stack({a1, a2, a3}, 2), expected));
        CHECK(allclose(stack({a1, a2, a3}, -1), expected));
    };
    runner_all_except_bool(test_stack3);

    auto test_stack_unequal = []<typename T>(Device device) {
        const auto options = TensorOptions().dtype(to_scalar<T>::type).device(device);
        Tensor a1 = zeros({3, 4}, options);
        Tensor a2 = zeros({3, 1}, options);
        Tensor a3 = zeros({3, 4}, options);
        CHECK_THROWS_AS(std::ignore = stack({a1, a2, a3}, 0), TTException);
    };
    runner_all_except_bool(test_stack_unequal);

    auto test_stack_dim_error = []<typename T>(Device device) {
        const auto options = TensorOptions().dtype(to_scalar<T>::type).device(device);
        Tensor a1 = zeros({3, 4}, options);
        Tensor a2 = zeros({3, 4}, options);
        Tensor a3 = zeros({3, 4}, options);
        CHECK_THROWS_AS(std::ignore = stack({a1, a2, a3}, 3), TTException);
        CHECK_THROWS_AS(std::ignore = stack({a1, a2, a3}, -4), TTException);
    };
    runner_all_except_bool(test_stack_dim_error);
}
