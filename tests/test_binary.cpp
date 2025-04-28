// test_binary.cpp
// Test the binary ops

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <tt/device.h>
#include <tt/exception.h>
#include <tt/scalar.h>
#include <tt/shape.h>
#include <tt/tensor.h>

#include "doctest.h"
#include "test_util.h"

#include <cmath>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <vector>

using namespace tinytensor;

// NOLINTNEXTLINE
TEST_CASE("Binary eq") {
    auto test_eq_all = []<typename T>(Device device) {
        Tensor lhs(std::vector<T>{1, 2, 3, 4, 5, 6}, {2, 3}, device);
        Tensor rhs(std::vector<T>{1, 2, 3, 4, 5, 6}, {2, 3}, device);
        Tensor expected(std::vector<bool>(6, true), {2, 3}, device);
        CHECK(allclose((lhs == rhs), expected));
        CHECK(allclose(eq(lhs, rhs), expected));
    };
    runner_all_except_bool(test_eq_all);
    auto test_eq_all_bool = []<typename T>(Device device) {
        Tensor lhs(std::vector<bool>{false, false, true, true, true, false}, {2, 3}, device);
        Tensor rhs(std::vector<bool>{false, false, true, true, true, false}, {2, 3}, device);
        Tensor expected(std::vector<bool>(6, true), {2, 3}, device);
        CHECK(allclose((lhs == rhs), expected));
        CHECK(allclose(eq(lhs, rhs), expected));
    };
    runner_boolean(test_eq_all_bool);

    auto test_eq_all_scalar = []<typename T>(Device device) {
        Tensor lhs(std::vector<T>{10, 10, 10, 10, 10, 10}, {2, 3}, device);
        auto rhs = Scalar(static_cast<T>(10));
        Tensor expected(std::vector<bool>(6, true), {2, 3}, device);
        {
            CHECK(allclose((lhs == rhs), expected));
            CHECK(allclose(eq(lhs, rhs), expected));
        }
        {
            CHECK(allclose((lhs == 10), expected));
            CHECK(allclose(eq(lhs, 10), expected));
        }
        {
            CHECK(allclose((10 == lhs), expected));
            CHECK(allclose(eq(10, lhs), expected));
        }
    };
    runner_all_except_bool(test_eq_all_scalar);

    auto test_eq_none = []<typename T>(Device device) {
        Tensor lhs(std::vector<T>{1, 2, 3, 4, 5, 6}, {2, 3}, device);
        Tensor rhs(std::vector<T>{2, 1, 4, 3, 6, 5}, {2, 3}, device);
        Tensor expected(std::vector<bool>(6, false), {2, 3}, device);
        CHECK(allclose((lhs == rhs), expected));
        CHECK(allclose(eq(lhs, rhs), expected));
    };
    runner_all_except_bool(test_eq_none);
    auto test_eq_none_bool = []<typename T>(Device device) {
        Tensor lhs(std::vector<bool>{false, false, true, true, true, false}, {2, 3}, device);
        Tensor rhs(std::vector<bool>{true, true, false, false, false, true}, {2, 3}, device);
        Tensor expected(std::vector<bool>(6, false), {2, 3}, device);
        CHECK(allclose((lhs == rhs), expected));
        CHECK(allclose(eq(lhs, rhs), expected));
    };
    runner_boolean(test_eq_none_bool);
}

// NOLINTNEXTLINE
TEST_CASE("Binary ne") {
    auto test_ne_all = []<typename T>(Device device) {
        Tensor lhs(std::vector<T>{1, 2, 3, 4, 5, 6}, {2, 3}, device);
        Tensor rhs(std::vector<T>{11, 12, 13, 14, 15, 16}, {2, 3}, device);
        Tensor expected(std::vector<bool>(6, true), {2, 3}, device);
        CHECK(allclose((lhs != rhs), expected));
        CHECK(allclose(ne(lhs, rhs), expected));
    };
    runner_all_except_bool(test_ne_all);
    auto test_ne_all_bool = []<typename T>(Device device) {
        Tensor lhs(std::vector<bool>{false, false, true, true, true, false}, {2, 3}, device);
        Tensor rhs(std::vector<bool>{true, true, false, false, false, true}, {2, 3}, device);
        Tensor expected(std::vector<bool>(6, true), {2, 3}, device);
        CHECK(allclose((lhs != rhs), expected));
        CHECK(allclose(ne(lhs, rhs), expected));
    };
    runner_boolean(test_ne_all_bool);

    auto test_ne_all_scalar = []<typename T>(Device device) {
        Tensor lhs(std::vector<T>{1, 2, 3, 4, 5, 6}, {2, 3}, device);
        auto rhs = Scalar(static_cast<T>(10));
        Tensor expected(std::vector<bool>(6, true), {2, 3}, device);
        {
            CHECK(allclose((lhs != rhs), expected));
            CHECK(allclose(ne(lhs, rhs), expected));
        }
        {
            CHECK(allclose((lhs != 10), expected));
            CHECK(allclose(ne(lhs, 10), expected));
        }
        {
            CHECK(allclose((10 != lhs), expected));
            CHECK(allclose(ne(10, lhs), expected));
        }
    };
    runner_all_except_bool(test_ne_all_scalar);

    auto test_ne_none = []<typename T>(Device device) {
        Tensor lhs(std::vector<T>{1, 2, 3, 4, 5, 6}, {2, 3}, device);
        Tensor rhs(std::vector<T>{1, 2, 3, 4, 5, 6}, {2, 3}, device);
        Tensor expected(std::vector<bool>(6, false), {2, 3}, device);
        CHECK(allclose((lhs != rhs), expected));
        CHECK(allclose(ne(lhs, rhs), expected));
    };
    runner_all_except_bool(test_ne_none);
    auto test_ne_none_bool = []<typename T>(Device device) {
        Tensor lhs(std::vector<bool>{false, false, true, true, true, false}, {2, 3}, device);
        Tensor rhs(std::vector<bool>{false, false, true, true, true, false}, {2, 3}, device);
        Tensor expected(std::vector<bool>(6, false), {2, 3}, device);
        CHECK(allclose((lhs != rhs), expected));
        CHECK(allclose(ne(lhs, rhs), expected));
    };
    runner_boolean(test_ne_none_bool);
}

// NOLINTNEXTLINE
TEST_CASE("Binary lt") {
    auto test_lt_all = []<typename T>(Device device) {
        Tensor lhs(std::vector<T>{1, 2, 3, 4, 5, 6}, {2, 3}, device);
        Tensor rhs(std::vector<T>{11, 12, 13, 14, 15, 16}, {2, 3}, device);
        Tensor expected(std::vector<bool>(6, true), {2, 3}, device);
        CHECK(allclose((lhs < rhs), expected));
        CHECK(allclose(lt(lhs, rhs), expected));
    };
    runner_all_except_bool(test_lt_all);
    auto test_lt_all_bool = []<typename T>(Device device) {
        Tensor lhs = full(false, {2, 3}, device);
        Tensor rhs = full(true, {2, 3}, device);
        Tensor expected = full(true, {2, 3}, device);
        CHECK(allclose((lhs < rhs), expected));
        CHECK(allclose(lt(lhs, rhs), expected));
    };
    runner_boolean(test_lt_all_bool);

    auto test_lt_all_scalar = []<typename T>(Device device) {
        Tensor lhs(std::vector<T>{1, 2, 3, 4, 5, 6}, {2, 3}, device);
        auto rhs = Scalar(static_cast<T>(10));
        Tensor expected(std::vector<bool>(6, true), {2, 3}, device);
        {
            CHECK(allclose((lhs < rhs), expected));
            CHECK(allclose(lt(lhs, rhs), expected));
        }
        {
            CHECK(allclose((lhs < 10), expected));
            CHECK(allclose(lt(lhs, 10), expected));
        }
        {
            CHECK(allclose((0 < lhs), expected));
            CHECK(allclose(lt(0, lhs), expected));
        }
    };
    runner_all_except_bool(test_lt_all_scalar);

    auto test_lt_none = []<typename T>(Device device) {
        Tensor lhs(std::vector<T>{1, 2, 3, 4, 5, 6}, {2, 3}, device);
        Tensor rhs(std::vector<T>{1, 2, 3, 4, 5, 6}, {2, 3}, device);
        Tensor expected(std::vector<bool>(6, false), {2, 3}, device);
        CHECK(allclose((lhs < rhs), expected));
        CHECK(allclose(lt(lhs, rhs), expected));
    };
    runner_all_except_bool(test_lt_none);
    auto test_lt_none_bool = []<typename T>(Device device) {
        Tensor lhs = full(true, {2, 3}, device);
        Tensor rhs = full(false, {2, 3}, device);
        Tensor expected = full(false, {2, 3}, device);
        CHECK(allclose((lhs < rhs), expected));
        CHECK(allclose(lt(lhs, rhs), expected));
    };
    runner_boolean(test_lt_none_bool);
}

// NOLINTNEXTLINE
TEST_CASE("Binary le") {
    auto test_le = []<typename T>(Device device) {
        Tensor lhs(std::vector<T>{1, 2, 3, 14, 15, 6}, {2, 3}, device);
        Tensor rhs(std::vector<T>{11, 2, 3, 4, 5, 6}, {2, 3}, device);
        Tensor expected(std::vector<bool>{true, true, true, false, false, true}, {2, 3}, device);
        CHECK(allclose((lhs <= rhs), expected));
        CHECK(allclose(le(lhs, rhs), expected));
    };
    runner_all_except_bool(test_le);
}

// NOLINTNEXTLINE
TEST_CASE("Binary gt") {
    auto test_gt_all = []<typename T>(Device device) {
        Tensor lhs(std::vector<T>{11, 12, 13, 14, 15, 16}, {2, 3}, device);
        Tensor rhs(std::vector<T>{1, 2, 3, 4, 5, 6}, {2, 3}, device);
        Tensor expected(std::vector<bool>(6, true), {2, 3}, device);
        CHECK(allclose((lhs > rhs), expected));
        CHECK(allclose(gt(lhs, rhs), expected));
    };
    runner_all_except_bool(test_gt_all);
    auto test_gt_all_bool = []<typename T>(Device device) {
        Tensor lhs = full(true, {2, 3}, device);
        Tensor rhs = full(false, {2, 3}, device);
        Tensor expected = full(true, {2, 3}, device);
        CHECK(allclose((lhs > rhs), expected));
        CHECK(allclose(gt(lhs, rhs), expected));
    };
    runner_boolean(test_gt_all_bool);

    auto test_gt_all_scalar = []<typename T>(Device device) {
        Tensor lhs(std::vector<T>{11, 12, 13, 14, 15, 16}, {2, 3}, device);
        auto rhs = Scalar(static_cast<T>(10));
        Tensor expected(std::vector<bool>(6, true), {2, 3}, device);
        {
            CHECK(allclose((lhs > rhs), expected));
            CHECK(allclose(gt(lhs, rhs), expected));
        }
        {
            CHECK(allclose((lhs > 10), expected));
            CHECK(allclose(gt(lhs, 10), expected));
        }
        {
            CHECK(allclose((20 > lhs), expected));
            CHECK(allclose(gt(20, lhs), expected));
        }
    };
    runner_all_except_bool(test_gt_all_scalar);

    auto test_gt_none = []<typename T>(Device device) {
        Tensor lhs(std::vector<T>{1, 2, 3, 4, 5, 6}, {2, 3}, device);
        Tensor rhs(std::vector<T>{1, 2, 3, 4, 5, 6}, {2, 3}, device);
        Tensor expected(std::vector<bool>(6, false), {2, 3}, device);
        CHECK(allclose((lhs > rhs), expected));
        CHECK(allclose(gt(lhs, rhs), expected));
    };
    runner_all_except_bool(test_gt_none);
    auto test_gt_none_bool = []<typename T>(Device device) {
        Tensor lhs = full(false, {2, 3}, device);
        Tensor rhs = full(true, {2, 3}, device);
        Tensor expected = full(false, {2, 3}, device);
        CHECK(allclose((lhs > rhs), expected));
        CHECK(allclose(gt(lhs, rhs), expected));
    };
    runner_boolean(test_gt_none_bool);
}

// NOLINTNEXTLINE
TEST_CASE("Binary ge") {
    auto test_ge = []<typename T>(Device device) {
        Tensor lhs(std::vector<T>{1, 2, 3, 14, 15, 6}, {2, 3}, device);
        Tensor rhs(std::vector<T>{11, 2, 3, 4, 5, 7}, {2, 3}, device);
        Tensor expected(std::vector<bool>{false, true, true, true, true, false}, {2, 3}, device);
        CHECK(allclose((lhs >= rhs), expected));
        CHECK(allclose(ge(lhs, rhs), expected));
    };
    runner_all_except_bool(test_ge);
}

// NOLINTNEXTLINE
TEST_CASE("Binary logical_or") {
    auto test_or_all = []<typename T>(Device device) {
        Tensor lhs(std::vector<T>{0, 0, 0, 1, 1, 1}, {2, 3}, device);
        Tensor rhs(std::vector<T>{0, 1, 0, 1, 0, 1}, {2, 3}, device);
        Tensor expected(std::vector<bool>{false, true, false, true, true, true}, {2, 3}, device);
        CHECK(allclose((lhs || rhs), expected));
        CHECK(allclose(logical_or(lhs, rhs), expected));
    };
    runner_all(test_or_all);

    auto test_or_all_scalar = []<typename T>(Device device) {
        Tensor lhs(std::vector<T>{0, 1, 0, 1, 0, 1}, {2, 3}, device);
        auto rhs = Scalar(static_cast<T>(1));
        Tensor expected(std::vector<bool>(6, true), {2, 3}, device);
        {
            CHECK(allclose((lhs || rhs), expected));
            CHECK(allclose(logical_or(lhs, rhs), expected));
        }
        {
            CHECK(allclose((lhs || 1), expected));
            CHECK(allclose(logical_or(lhs, 1), expected));
        }
        {
            CHECK(allclose((1 || lhs), expected));
            CHECK(allclose(logical_or(1, lhs), expected));
        }
    };
    runner_all(test_or_all_scalar);

    auto test_or_none = []<typename T>(Device device) {
        Tensor lhs(std::vector<T>{0, 0, 0, 0, 0, 0}, {2, 3}, device);
        Tensor rhs(std::vector<T>{0, 0, 0, 0, 0, 0}, {2, 3}, device);
        Tensor expected(std::vector<bool>(6, false), {2, 3}, device);
        CHECK(allclose((lhs || rhs), expected));
        CHECK(allclose(logical_or(lhs, rhs), expected));
    };
    runner_all(test_or_none);
}

// NOLINTNEXTLINE
TEST_CASE("Binary logical_and") {
    auto test_and_all = []<typename T>(Device device) {
        Tensor lhs(std::vector<T>{0, 0, 0, 1, 1, 1}, {2, 3}, device);
        Tensor rhs(std::vector<T>{0, 1, 0, 1, 0, 1}, {2, 3}, device);
        Tensor expected(std::vector<bool>{false, false, false, true, false, true}, {2, 3}, device);
        CHECK(allclose((lhs && rhs), expected));
        CHECK(allclose(logical_and(lhs, rhs), expected));
    };
    runner_all(test_and_all);

    auto test_and_all_scalar = []<typename T>(Device device) {
        Tensor lhs(std::vector<T>{1, 1, 1, 1, 1, 1}, {2, 3}, device);
        auto rhs = Scalar(static_cast<T>(1));
        Tensor expected_true(std::vector<bool>(6, true), {2, 3}, device);
        Tensor expected_false(std::vector<bool>(6, false), {2, 3}, device);
        {
            CHECK(allclose((lhs && rhs), expected_true));
            CHECK(allclose(logical_and(lhs, rhs), expected_true));
        }
        {
            CHECK(allclose((lhs && 1), expected_true));
            CHECK(allclose(logical_and(lhs, 1), expected_true));
        }
        {
            CHECK(allclose((1 && lhs), expected_true));
            CHECK(allclose(logical_and(1, lhs), expected_true));
        }
        {
            CHECK(allclose((0 && lhs), expected_false));
            CHECK(allclose(logical_and(0, lhs), expected_false));
        }
    };
    runner_all(test_and_all_scalar);

    auto test_and_none = []<typename T>(Device device) {
        Tensor lhs(std::vector<T>{1, 1, 1, 0, 0, 0}, {2, 3}, device);
        Tensor rhs(std::vector<T>{0, 0, 0, 1, 1, 1}, {2, 3}, device);
        Tensor expected(std::vector<bool>(6, false), {2, 3}, device);
        CHECK(allclose((lhs && rhs), expected));
        CHECK(allclose(logical_and(lhs, rhs), expected));
    };
    runner_all(test_and_none);
}

// NOLINTNEXTLINE
TEST_CASE("Binary bitwise_or") {
    auto test_bitwise_or_bool = []<typename T>(Device device) {
        std::vector<bool> data_lhs = {true, true, true, false, false, false};
        std::vector<bool> data_rhs = {true, false, true, false, true, false};
        std::vector<bool> data_result = {true, true, true, false, true, false};
        Tensor lhs(data_lhs, {2, 3}, device);
        Tensor rhs(data_rhs, {2, 3}, device);
        Tensor expected(data_result, {2, 3}, device);
        CHECK(allclose((lhs | rhs), expected));
        CHECK(allclose(bitwise_or(lhs, rhs), expected));
    };
    runner_boolean(test_bitwise_or_bool);

    auto test_bitwise_or_integral = []<typename T>(Device device) {
        std::vector<T> data_lhs = {0x0, 0x0, 0x0, 0xFF, 0xFF, 0xFF};
        std::vector<T> data_rhs = {0x0, 0xF0, 0x0F, 0xFF, 0xF0, 0x0F};
        std::vector<T> data_result(data_lhs.size());
        for (std::size_t i = 0; i < data_result.size(); ++i) {
            data_result[i] = data_lhs[i] | data_rhs[i];
        }
        Tensor lhs(data_lhs, {2, 3}, device);
        Tensor rhs(data_rhs, {2, 3}, device);
        Tensor expected(data_result, {2, 3}, device);
        CHECK(allclose((lhs | rhs), expected));
        CHECK(allclose(bitwise_or(lhs, rhs), expected));
    };
    runner_integral(test_bitwise_or_integral);

    auto test_bitwise_or_float = []<typename T>(Device device) {
        std::vector<T> data_lhs = {0x0, 0x0, 0x0, 0xFF, 0xFF, 0xFF};
        std::vector<T> data_rhs = {0x0, 0xF0, 0x0F, 0xFF, 0xF0, 0x0F};
        Tensor lhs(data_lhs, {2, 3}, device);
        Tensor rhs(data_rhs, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = lhs | rhs, TTException);
        CHECK_THROWS_AS(std::ignore = bitwise_or(lhs, rhs), TTException);
    };
    runner_floating_point(test_bitwise_or_float);
}

// NOLINTNEXTLINE
TEST_CASE("Binary bitwise_and") {
    auto test_bitwise_and_bool = []<typename T>(Device device) {
        std::vector<bool> data_lhs = {true, true, true, false, false, false};
        std::vector<bool> data_rhs = {true, false, true, false, true, false};
        std::vector<bool> data_result = {true, false, true, false, false, false};
        Tensor lhs(data_lhs, {2, 3}, device);
        Tensor rhs(data_rhs, {2, 3}, device);
        Tensor expected(data_result, {2, 3}, device);
        CHECK(allclose((lhs & rhs), expected));
        CHECK(allclose(bitwise_and(lhs, rhs), expected));
    };
    runner_boolean(test_bitwise_and_bool);

    auto test_bitwise_and_integral = []<typename T>(Device device) {
        std::vector<T> data_lhs = {0x0, 0x0, 0x0, 0xFF, 0xFF, 0xFF};
        std::vector<T> data_rhs = {0x0, 0xF0, 0x0F, 0xFF, 0xF0, 0x0F};
        std::vector<T> data_result(data_lhs.size());
        for (std::size_t i = 0; i < data_result.size(); ++i) {
            data_result[i] = data_lhs[i] & data_rhs[i];
        }
        Tensor lhs(data_lhs, {2, 3}, device);
        Tensor rhs(data_rhs, {2, 3}, device);
        Tensor expected(data_result, {2, 3}, device);
        CHECK(allclose((lhs & rhs), expected));
        CHECK(allclose(bitwise_and(lhs, rhs), expected));
    };
    runner_integral(test_bitwise_and_integral);

    auto test_bitwise_and_float = []<typename T>(Device device) {
        std::vector<T> data_lhs = {0x0, 0x0, 0x0, 0xFF, 0xFF, 0xFF};
        std::vector<T> data_rhs = {0x0, 0xF0, 0x0F, 0xFF, 0xF0, 0x0F};
        Tensor lhs(data_lhs, {2, 3}, device);
        Tensor rhs(data_rhs, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = lhs & rhs, TTException);
        CHECK_THROWS_AS(std::ignore = bitwise_and(lhs, rhs), TTException);
    };
    runner_floating_point(test_bitwise_and_float);
}

// NOLINTNEXTLINE
TEST_CASE("Binary bitwise_xor") {
    auto test_bitwise_xor_bool = []<typename T>(Device device) {
        std::vector<bool> data_lhs = {true, true, true, false, false, false};
        std::vector<bool> data_rhs = {true, false, true, false, true, false};
        std::vector<bool> data_result = {false, true, false, false, true, false};
        Tensor lhs(data_lhs, {2, 3}, device);
        Tensor rhs(data_rhs, {2, 3}, device);
        Tensor expected(data_result, {2, 3}, device);
        CHECK(allclose((lhs ^ rhs), expected));
        CHECK(allclose(bitwise_xor(lhs, rhs), expected));
    };
    runner_boolean(test_bitwise_xor_bool);

    auto test_bitwise_xor_integral = []<typename T>(Device device) {
        std::vector<T> data_lhs = {0x0, 0x0, 0x0, 0xFF, 0xFF, 0xFF};
        std::vector<T> data_rhs = {0x0, 0xF0, 0x0F, 0xFF, 0xF0, 0x0F};
        std::vector<T> data_result(data_lhs.size());
        for (std::size_t i = 0; i < data_result.size(); ++i) {
            data_result[i] = data_lhs[i] ^ data_rhs[i];
        }
        Tensor lhs(data_lhs, {2, 3}, device);
        Tensor rhs(data_rhs, {2, 3}, device);
        Tensor expected(data_result, {2, 3}, device);
        CHECK(allclose((lhs ^ rhs), expected));
        CHECK(allclose(bitwise_xor(lhs, rhs), expected));
    };
    runner_integral(test_bitwise_xor_integral);

    auto test_bitwise_xor_float = []<typename T>(Device device) {
        std::vector<T> data_lhs = {0x0, 0x0, 0x0, 0xFF, 0xFF, 0xFF};
        std::vector<T> data_rhs = {0x0, 0xF0, 0x0F, 0xFF, 0xF0, 0x0F};
        Tensor lhs(data_lhs, {2, 3}, device);
        Tensor rhs(data_rhs, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = lhs ^ rhs, TTException);
        CHECK_THROWS_AS(std::ignore = bitwise_xor(lhs, rhs), TTException);
    };
    runner_floating_point(test_bitwise_xor_float);
}

// NOLINTNEXTLINE
TEST_CASE("Binary modulo") {
    // mod disabled for bool
    auto test_mod_bool = []<typename T>(Device device) {
        Tensor lhs = full(true, {2, 3}, device);
        Tensor rhs = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = lhs % rhs, TTException);
        CHECK_THROWS_AS(std::ignore = modulo(lhs, rhs), TTException);
    };
    runner_boolean(test_mod_bool);

    auto test_modulo = []<typename T>(Device device) {
        std::vector<T> data_lhs = {11, 22, 30, 44, 55, 66};
        std::vector<T> data_rhs = {6, 5, 4, 3, 4, 5};
        std::vector<T> data_result(data_lhs.size());
        for (std::size_t i = 0; i < data_result.size(); ++i) {
            if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
                data_result[i] = static_cast<T>(std::fmod(data_lhs[i], data_rhs[i]));
            } else {
                data_result[i] = data_lhs[i] % data_rhs[i];
            }
        }
        Tensor lhs(data_lhs, {2, 3}, device);
        Tensor rhs(data_rhs, {2, 3}, device);
        Tensor expected(data_result, {2, 3}, device);
        CHECK(allclose((lhs % rhs), expected));
        CHECK(allclose(modulo(lhs, rhs), expected));
    };
    runner_all_except_bool(test_modulo);
}

// NOLINTNEXTLINE
TEST_CASE("Binary bitwise_left_shift") {
    auto test_bitwise_left_shift_integral = []<typename T>(Device device) {
        constexpr T fixed_shift = 4;
        std::vector<T> data_lhs = {0x0, 0x0, 0x0, 0xFF, 0xFF, 0xFF};
        std::vector<T> data_rhs = {10, 12, 8, 6, 10, 12};
        std::vector<T> data_result1(data_lhs.size());
        std::vector<T> data_result2(data_lhs.size());
        for (std::size_t i = 0; i < data_result1.size(); ++i) {
            data_result1[i] = static_cast<T>(data_lhs[i] << data_rhs[i]);
            data_result2[i] = static_cast<T>(data_lhs[i] << fixed_shift);
        }
        Tensor lhs(data_lhs, {2, 3}, device);
        Tensor rhs(data_rhs, {2, 3}, device);
        Tensor expected1(data_result1, {2, 3}, device);
        Tensor expected2(data_result2, {2, 3}, device);
        CHECK(allclose((lhs << rhs), expected1));
        CHECK(allclose(bitwise_left_shift(lhs, rhs), expected1));
        CHECK(allclose((lhs << fixed_shift), expected2));
        CHECK(allclose(bitwise_left_shift(lhs, fixed_shift), expected2));
    };
    runner_integral(test_bitwise_left_shift_integral);

    auto test_bitwise_left_shift_float = []<typename T>(Device device) {
        constexpr T fixed_shift = 4;
        std::vector<T> data_lhs = {0x0, 0x0, 0x0, 0xFF, 0xFF, 0xFF};
        std::vector<T> data_rhs = {10, 12, 8, 6, 10, 12};
        Tensor lhs(data_lhs, {2, 3}, device);
        Tensor rhs(data_rhs, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = lhs << rhs, TTException);
        CHECK_THROWS_AS(std::ignore = bitwise_left_shift(lhs, rhs), TTException);
        CHECK_THROWS_AS(std::ignore = lhs << fixed_shift, TTException);
        CHECK_THROWS_AS(std::ignore = bitwise_left_shift(lhs, fixed_shift), TTException);
    };
    runner_floating_point(test_bitwise_left_shift_float);
}

// NOLINTNEXTLINE
TEST_CASE("Binary bitwise_right_shift") {
    auto test_bitwise_right_shift_integral = []<typename T>(Device device) {
        constexpr T fixed_shift = 4;
        std::vector<T> data_lhs = {0x0, 0x0, 0x0, 0xFF, 0xFF, 0xFF};
        std::vector<T> data_rhs = {10, 12, 8, 6, 10, 12};
        std::vector<T> data_result1(data_lhs.size());
        std::vector<T> data_result2(data_lhs.size());
        for (std::size_t i = 0; i < data_result1.size(); ++i) {
            data_result1[i] = static_cast<T>(data_lhs[i] >> data_rhs[i]);
            data_result2[i] = static_cast<T>(data_lhs[i] >> fixed_shift);
        }
        Tensor lhs(data_lhs, {2, 3}, device);
        Tensor rhs(data_rhs, {2, 3}, device);
        Tensor expected1(data_result1, {2, 3}, device);
        Tensor expected2(data_result2, {2, 3}, device);
        CHECK(allclose((lhs >> rhs), expected1));
        CHECK(allclose(bitwise_right_shift(lhs, rhs), expected1));
        CHECK(allclose((lhs >> fixed_shift), expected2));
        CHECK(allclose(bitwise_right_shift(lhs, fixed_shift), expected2));
    };
    runner_integral(test_bitwise_right_shift_integral);

    auto test_bitwise_right_shift_float = []<typename T>(Device device) {
        constexpr T fixed_shift = 4;
        std::vector<T> data_lhs = {0x0, 0x0, 0x0, 0xFF, 0xFF, 0xFF};
        std::vector<T> data_rhs = {10, 12, 8, 6, 10, 12};
        Tensor lhs(data_lhs, {2, 3}, device);
        Tensor rhs(data_rhs, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = lhs >> rhs, TTException);
        CHECK_THROWS_AS(std::ignore = bitwise_right_shift(lhs, rhs), TTException);
        CHECK_THROWS_AS(std::ignore = lhs >> fixed_shift, TTException);
        CHECK_THROWS_AS(std::ignore = bitwise_right_shift(lhs, fixed_shift), TTException);
    };
    runner_floating_point(test_bitwise_right_shift_float);
}

// NOLINTNEXTLINE
TEST_CASE("Binary add") {
    // add disabled for bool
    auto test_add_bool = []<typename T>(Device device) {
        Tensor lhs = full(true, {2, 3}, device);
        Tensor rhs = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = lhs + rhs, TTException);
        CHECK_THROWS_AS(std::ignore = add(lhs, rhs), TTException);
    };
    runner_boolean(test_add_bool);

    auto test_add = []<typename T>(Device device) {
        std::vector<T> data_lhs = {1, 2, 3, 4, 5, 6};
        std::vector<T> data_rhs = {1, 2, 3, 4, 5, 6};
        std::vector<T> data_result(data_lhs.size());
        for (std::size_t i = 0; i < data_result.size(); ++i) {
            data_result[i] = data_lhs[i] + data_rhs[i];
        }
        Tensor lhs(data_lhs, {2, 3}, device);
        Tensor rhs(data_rhs, {2, 3}, device);
        Tensor expected(data_result, {2, 3}, device);
        CHECK(allclose((lhs + rhs), expected));
        CHECK(allclose(add(lhs, rhs), expected));
    };
    runner_all_except_bool(test_add);
}

// NOLINTNEXTLINE
TEST_CASE("Binary sub") {
    // sub disabled for bool
    auto test_sub_bool = []<typename T>(Device device) {
        Tensor lhs = full(true, {2, 3}, device);
        Tensor rhs = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = lhs - rhs, TTException);
        CHECK_THROWS_AS(std::ignore = sub(lhs, rhs), TTException);
    };
    runner_boolean(test_sub_bool);

    auto test_sub = []<typename T>(Device device) {
        std::vector<T> data_lhs = {1, 2, 3, 4, 5, 6};
        std::vector<T> data_rhs = {2, 3, 4, 5, 6, 1};
        std::vector<T> data_result(data_lhs.size());
        for (std::size_t i = 0; i < data_result.size(); ++i) {
            data_result[i] = data_lhs[i] - data_rhs[i];
        }
        Tensor lhs(data_lhs, {2, 3}, device);
        Tensor rhs(data_rhs, {2, 3}, device);
        Tensor expected(data_result, {2, 3}, device);
        CHECK(allclose((lhs - rhs), expected));
        CHECK(allclose(sub(lhs, rhs), expected));
    };
    runner_all_except_bool(test_sub);
}

// NOLINTNEXTLINE
TEST_CASE("Binary mul") {
    // mul disabled for bool
    auto test_mul_bool = []<typename T>(Device device) {
        Tensor lhs = full(true, {2, 3}, device);
        Tensor rhs = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = lhs * rhs, TTException);
        CHECK_THROWS_AS(std::ignore = mul(lhs, rhs), TTException);
    };
    runner_boolean(test_mul_bool);

    auto test_mul = []<typename T>(Device device) {
        std::vector<T> data_lhs = {1, 2, 3, 4, 5, 6};
        std::vector<T> data_rhs = {2, 3, 4, 5, 6, 1};
        std::vector<T> data_result(data_lhs.size());
        for (std::size_t i = 0; i < data_result.size(); ++i) {
            data_result[i] = data_lhs[i] * data_rhs[i];
        }
        Tensor lhs(data_lhs, {2, 3}, device);
        Tensor rhs(data_rhs, {2, 3}, device);
        Tensor expected(data_result, {2, 3}, device);
        Tensor result = (lhs * rhs);
        CHECK(allclose((lhs * rhs), expected));
        CHECK(allclose(mul(lhs, rhs), expected));
    };
    runner_all_except_bool(test_mul);
}

// NOLINTNEXTLINE
TEST_CASE("Binary div") {
    // div disabled for bool
    auto test_div_bool = []<typename T>(Device device) {
        Tensor lhs = full(true, {2, 3}, device);
        Tensor rhs = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = lhs / rhs, TTException);
        CHECK_THROWS_AS(std::ignore = div(lhs, rhs), TTException);
    };
    runner_boolean(test_div_bool);

    auto test_div = []<typename T>(Device device) {
        using U = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        std::vector<T> data_lhs = {1, 2, 3, 4, 5, 6};
        std::vector<T> data_rhs = {2, 3, 4, 5, 6, 1};
        std::vector<U> data_result(data_lhs.size());
        for (std::size_t i = 0; i < data_result.size(); ++i) {
            data_result[i] = static_cast<U>(data_lhs[i]) / static_cast<U>(data_rhs[i]);
        }
        Tensor lhs(data_lhs, {2, 3}, device);
        Tensor rhs(data_rhs, {2, 3}, device);
        Tensor expected(data_result, {2, 3}, device);
        CHECK(allclose((lhs / rhs), expected));
        CHECK(allclose(div(lhs, rhs), expected));
    };
    runner_all_except_bool(test_div);
}

// NOLINTNEXTLINE
TEST_CASE("Binary maximum") {
    auto test_maximum_bool = []<typename T>(Device device) {
        Tensor lhs(std::vector<bool>{true, true, true, false, false, false}, {2, 3}, device);
        Tensor rhs(std::vector<bool>{true, false, true, false, true, false}, {2, 3}, device);
        Tensor expected(std::vector<bool>{true, true, true, false, true, false}, {2, 3}, device);
        CHECK(allclose(maximum(lhs, rhs), expected));
    };
    runner_boolean(test_maximum_bool);

    auto test_maximum1 = []<typename T>(Device device) {
        std::vector<T> data_lhs = {1, 2, 3, 4, 5, 6};
        std::vector<T> data_rhs = {2, 3, 4, 5, 6, 7};
        std::vector<T> data_result(data_lhs.size());
        for (std::size_t i = 0; i < data_result.size(); ++i) {
            data_result[i] = std::max(data_lhs[i], data_rhs[i]);
        }
        Tensor lhs(data_lhs, {2, 3}, device);
        Tensor rhs(data_rhs, {2, 3}, device);
        Tensor expected(data_result, {2, 3}, device);
        Tensor result = maximum(lhs, rhs);
        CHECK(allclose(result, expected));
    };
    runner_all_except_bool(test_maximum1);

    auto test_maximum2 = []<typename T>(Device device) {
        std::vector<T> data_lhs = {1, 2, 30, 40, 5, 60};
        std::vector<T> data_rhs = {2, 30, 4, 5, 6, 7};
        std::vector<T> data_result(data_lhs.size());
        for (std::size_t i = 0; i < data_result.size(); ++i) {
            data_result[i] = std::max(data_lhs[i], data_rhs[i]);
        }
        Tensor lhs(data_lhs, {2, 3}, device);
        Tensor rhs(data_rhs, {2, 3}, device);
        Tensor expected(data_result, {2, 3}, device);
        Tensor result = maximum(lhs, rhs);
        CHECK(allclose(result, expected));
    };
    runner_all_except_bool(test_maximum2);
}

// NOLINTNEXTLINE
TEST_CASE("Binary minimum") {
    auto test_minimum_bool = []<typename T>(Device device) {
        Tensor lhs(std::vector<bool>{true, true, true, false, false, false}, {2, 3}, device);
        Tensor rhs(std::vector<bool>{true, false, true, false, true, false}, {2, 3}, device);
        Tensor expected(std::vector<bool>{true, false, true, false, false, false}, {2, 3}, device);
        CHECK(allclose(minimum(lhs, rhs), expected));
    };
    runner_boolean(test_minimum_bool);

    auto test_minimum1 = []<typename T>(Device device) {
        std::vector<T> data_lhs = {1, 2, 3, 4, 5, 6};
        std::vector<T> data_rhs = {2, 3, 4, 5, 6, 7};
        std::vector<T> data_result(data_lhs.size());
        for (std::size_t i = 0; i < data_result.size(); ++i) {
            data_result[i] = std::min(data_lhs[i], data_rhs[i]);
        }
        Tensor lhs(data_lhs, {2, 3}, device);
        Tensor rhs(data_rhs, {2, 3}, device);
        Tensor expected(data_result, {2, 3}, device);
        Tensor result = minimum(lhs, rhs);
        CHECK(allclose(result, expected));
    };
    runner_all_except_bool(test_minimum1);

    auto test_minimum2 = []<typename T>(Device device) {
        std::vector<T> data_lhs = {1, 2, 30, 40, 5, 60};
        std::vector<T> data_rhs = {2, 30, 4, 5, 6, 7};
        std::vector<T> data_result(data_lhs.size());
        for (std::size_t i = 0; i < data_result.size(); ++i) {
            data_result[i] = std::min(data_lhs[i], data_rhs[i]);
        }
        Tensor lhs(data_lhs, {2, 3}, device);
        Tensor rhs(data_rhs, {2, 3}, device);
        Tensor expected(data_result, {2, 3}, device);
        Tensor result = minimum(lhs, rhs);
        CHECK(allclose(result, expected));
    };
    runner_all_except_bool(test_minimum2);
}

// NOLINTNEXTLINE
TEST_CASE("Binary pow") {
    // pow disabled for bool
    auto test_pow_bool = []<typename T>(Device device) {
        Tensor lhs = full(true, {2, 3}, device);
        Tensor rhs = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = pow(lhs, rhs), TTException);
    };
    runner_boolean(test_pow_bool);

    auto test_pow1 = []<typename T>(Device device) {
        std::vector<T> data_lhs = {1, 2, 3, 4, 5, 6};
        std::vector<T> data_rhs = {6, 5, 4, 3, 2, 1};
        std::vector<T> data_result(data_lhs.size());
        for (std::size_t i = 0; i < data_result.size(); ++i) {
            data_result[i] = static_cast<T>(std::pow(data_lhs[i], data_rhs[i]));
        }
        Tensor lhs(data_lhs, {2, 3}, device);
        Tensor rhs(data_rhs, {2, 3}, device);
        Tensor expected(data_result, {2, 3}, device);
        CHECK(allclose(pow(lhs, rhs), expected));
    };
    runner_all_except_bool(test_pow1);

    auto test_pow2 = []<typename T>(Device device) {
        std::vector<T> data_rhs = {6, 5, 4, 3, 2, 1};
        std::vector<T> data_result(data_rhs.size());
        for (std::size_t i = 0; i < data_result.size(); ++i) {
            data_result[i] = static_cast<T>(std::pow(2, data_rhs[i]));
        }
        Tensor rhs(data_rhs, {2, 3}, device);
        Tensor expected(data_result, {2, 3}, device);
        CHECK(allclose(pow(2, rhs), expected));
    };
    runner_all_except_bool(test_pow2);
}
