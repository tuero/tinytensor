// test_unary.cpp
// Test the unary ops

// Clang is noisy on hard-coded test data
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wimplicit-float-conversion"
#endif

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <tt/device.h>
#include <tt/exception.h>
#include <tt/scalar.h>
#include <tt/tensor.h>

#ifdef TT_TORCH
#include <torch/torch.h>
#undef CHECK
#endif

#include "doctest.h"
#include "test_util.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <tuple>
#include <type_traits>
#include <vector>

using namespace tinytensor;

namespace {
template <typename R, typename T>
constexpr auto SC(T t) {
    return static_cast<R>(t);
}

template <typename T>
std::vector<T> rand_vec(int size, std::mt19937 &gen) {
    std::vector<T> res;
    res.reserve(static_cast<std::size_t>(size));
    std::uniform_int_distribution<int> dis(-5.0, 5.0);
    for (int i = 0; i < size; ++i) {
        res.push_back(static_cast<T>(dis(gen)));
    }
    return res;
}
}    // namespace

// NOLINTNEXTLINE
TEST_CASE("Unary abs") {
    // abs disabled for bool
    auto test_abs_bool = []<typename T>(Device device) {
        Tensor input = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = abs(input), TTException);
    };
    runner_boolean(test_abs_bool);

    auto test_abs = []<typename T>(Device device) {
        std::vector<T> input_values = {1, 2, 3, 4, 5, 6};
        if constexpr (!std::is_same_v<T, uint8_t>) {
            for (std::size_t i = 0; i < input_values.size(); i += 2) {
                input_values[i] *= -1;
            }
        }
        std::vector<T> expected_values = input_values;
        std::transform(input_values.begin(), input_values.end(), expected_values.begin(), [](auto v) {
            return std::abs(v);
        });
        Tensor input(input_values, {2, 3}, device);
        Tensor expected(expected_values, {2, 3}, device);
        CHECK(allclose(abs(input), expected));
    };
    runner_floating_point(test_abs);
    runner_signed_integral(test_abs);
}

// NOLINTNEXTLINE
TEST_CASE("Unary sign") {
    // sign disabled for bool
    auto test_sign_bool = []<typename T>(Device device) {
        Tensor input = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = sign(input), TTException);
    };
    runner_boolean(test_sign_bool);

    auto test_sign = []<typename T>(Device device) {
        std::vector<T> input_values = {1, -2, -3, 4, 5, 0};
        std::vector<T> expected_values = {1, -1, -1, 1, 1, 0};
        Tensor input(input_values, {2, 3}, device);
        Tensor expected(expected_values, {2, 3}, device);
        CHECK(allclose(sign(input), expected));
    };
    runner_signed_integral(test_sign);
    runner_floating_point(test_sign);
}

// NOLINTNEXTLINE
TEST_CASE("Unary logical_not") {
    auto test_logical_not_bool = []<typename T>(Device device) {
        std::vector<bool> input_values = {true, false, false, true, true, false};
        std::vector<bool> expected_values = {false, true, true, false, false, true};
        Tensor input(input_values, {2, 3}, device);
        Tensor expected(expected_values, {2, 3}, device);
        CHECK(allclose(logical_not(input), expected));
        CHECK(allclose(!input, expected));
    };
    runner_boolean(test_logical_not_bool);

    // logical_not disabled for non-bool
    auto test_logical_not = []<typename T>(Device device) {
        Tensor input = full(static_cast<T>(1), {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = logical_not(input), TTException);
        CHECK_THROWS_AS(std::ignore = !input, TTException);
    };
    runner_all_except_bool(test_logical_not);
}

// NOLINTNEXTLINE
TEST_CASE("Unary log") {
    // log disabled for bool
    auto test_log_bool = []<typename T>(Device device) {
        Tensor input = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = log(input), TTException);
    };
    runner_boolean(test_log_bool);

    auto test_log = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        std::vector<R> input_values = {1, 2, 3, 4, 5, 6};
        std::vector<R> expected_values(input_values.size());
        std::transform(input_values.begin(), input_values.end(), expected_values.begin(), [](auto v) {
            return std::log(v);
        });
        Tensor input(input_values, {2, 3}, device);
        Tensor expected(expected_values, {2, 3}, device);
        CHECK(allclose(log(input), expected));
    };
    runner_all_except_bool(test_log);

    // Log on negative numbers is nan, 0 is -inf
    auto test_log_nan = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        std::vector<R> input_values = {-1, -2, -3, -4, -5, -6};
        Tensor input(input_values, {2, 3}, device);
        CHECK(all(isnan(log(input))));
    };
    runner_integral(test_log_nan);
    runner_floating_point(test_log_nan);
    auto test_log_inf = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        std::vector<R> input_values(6, 0);
        Tensor input(input_values, {2, 3}, device);
        CHECK(all(isinf(log(input))));
    };
    runner_integral(test_log_inf);
    runner_floating_point(test_log_inf);
}

// NOLINTNEXTLINE
TEST_CASE("Unary log10") {
    // log10 disabled for bool
    auto test_log10_bool = []<typename T>(Device device) {
        Tensor input = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = log10(input), TTException);
    };
    runner_boolean(test_log10_bool);

    auto test_log10 = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        std::vector<R> input_values = {1, 2, 3, 4, 5, 6};
        std::vector<R> expected_values(input_values.size());
        std::transform(input_values.begin(), input_values.end(), expected_values.begin(), [](auto v) {
            return std::log10(v);
        });
        Tensor input(input_values, {2, 3}, device);
        Tensor expected(expected_values, {2, 3}, device);
        CHECK(allclose(log10(input), expected));
    };
    runner_all_except_bool(test_log10);

    // Log on negative numbers is nan, 0 is -inf
    auto test_log10_nan = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        std::vector<R> input_values = {-1, -2, -3, -4, -5, -6};
        Tensor input(input_values, {2, 3}, device);
        CHECK(all(isnan(log10(input))));
    };
    runner_integral(test_log10_nan);
    runner_floating_point(test_log10_nan);
    auto test_log10_inf = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        std::vector<R> input_values(6, 0);
        Tensor input(input_values, {2, 3}, device);
        CHECK(all(isinf(log(input))));
    };
    runner_integral(test_log10_inf);
    runner_floating_point(test_log10_inf);
}

// NOLINTNEXTLINE
TEST_CASE("Unary log2") {
    // log2 disabled for bool
    auto test_log2_bool = []<typename T>(Device device) {
        Tensor input = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = log2(input), TTException);
    };
    runner_boolean(test_log2_bool);

    auto test_log2 = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        std::vector<R> input_values = {1, 2, 3, 4, 5, 6};
        std::vector<R> expected_values(input_values.size());
        std::transform(input_values.begin(), input_values.end(), expected_values.begin(), [](auto v) {
            return std::log2(v);
        });
        Tensor input(input_values, {2, 3}, device);
        Tensor expected(expected_values, {2, 3}, device);
        CHECK(allclose(log2(input), expected));
    };
    runner_all_except_bool(test_log2);

    // Log on negative numbers is nan, 0 is -fin
    auto test_log2_nan = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        std::vector<R> input_values = {-1, -2, -3, -4, -5, -6};
        Tensor input(input_values, {2, 3}, device);
        CHECK(all(isnan(log2(input))));
    };
    runner_integral(test_log2_nan);
    runner_floating_point(test_log2_nan);
    auto test_log2_inf = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        std::vector<R> input_values(6, 0);
        Tensor input(input_values, {2, 3}, device);
        CHECK(all(isinf(log2(input))));
    };
    runner_integral(test_log2_inf);
    runner_floating_point(test_log2_inf);
}

// NOLINTNEXTLINE
TEST_CASE("Unary log1p") {
    // log1p disabled for bool
    auto test_log1p_bool = []<typename T>(Device device) {
        Tensor input = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = log1p(input), TTException);
    };
    runner_boolean(test_log1p_bool);

    auto test_log1p = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        std::vector<R> input_values = {1, 2, 3, 4, 5, 6};
        std::vector<R> expected_values(input_values.size());
        std::transform(input_values.begin(), input_values.end(), expected_values.begin(), [](auto v) {
            return std::log1p(v);
        });
        Tensor input(input_values, {2, 3}, device);
        Tensor expected(expected_values, {2, 3}, device);
        CHECK(allclose(log1p(input), expected));
    };
    runner_all_except_bool(test_log1p);

    // Log1p on numbers < -1 is nan, -1 is -inf
    auto test_log1p_nan = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        std::vector<R> input_values = {-2, -3, -4, -5, -6, -7};
        Tensor input(input_values, {2, 3}, device);
        CHECK(all(isnan(log1p(input))));
    };
    runner_integral(test_log1p_nan);
    runner_floating_point(test_log1p_nan);
    auto test_log1p_inf = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        std::vector<R> input_values(6, -1);
        Tensor input(input_values, {2, 3}, device);
        CHECK(all(isinf(log1p(input))));
    };
    runner_integral(test_log1p_inf);
    runner_floating_point(test_log1p_inf);
}

// NOLINTNEXTLINE
TEST_CASE("Unary exp") {
    // exp disabled for bool
    auto test_exp_bool = []<typename T>(Device device) {
        Tensor input = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = exp(input), TTException);
    };
    runner_boolean(test_exp_bool);

    auto test_exp = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        std::vector<R> input_values = {1, 2, 3, 4, 5, 6};
        std::vector<R> expected_values(input_values.size());
        std::transform(input_values.begin(), input_values.end(), expected_values.begin(), [](auto v) {
            return std::exp(static_cast<R>(v));
        });
        Tensor input(input_values, {2, 3}, device);
        Tensor expected(expected_values, {2, 3}, device);
        CHECK(allclose(exp(input), expected));
    };
    runner_all_except_bool(test_exp);

    // Large values with exp give inf
    auto test_exp_inf = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        std::vector<R> input_values = {1000, 2000, 3000, 4000, 5000, 6000};
        Tensor input(input_values, {2, 3}, device);
        CHECK(all(isinf(exp(input))));
    };
    runner_all_except_bool(test_exp_inf);
}

// NOLINTNEXTLINE
TEST_CASE("Unary exp2") {
    // exp2 disabled for bool
    auto test_exp2_bool = []<typename T>(Device device) {
        Tensor input = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = exp2(input), TTException);
    };
    runner_boolean(test_exp2_bool);

    auto test_exp2 = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        std::vector<R> input_values = {1, 2, 3, 4, 5, 6};
        std::vector<R> expected_values(input_values.size());
        std::transform(input_values.begin(), input_values.end(), expected_values.begin(), [](auto v) {
            return std::exp2(static_cast<R>(v));
        });
        Tensor input(input_values, {2, 3}, device);
        Tensor expected(expected_values, {2, 3}, device);
        CHECK(allclose(exp2(input), expected));
    };
    runner_all_except_bool(test_exp2);

    // Large values with exp2 give inf
    auto test_exp2_inf = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        std::vector<R> input_values = {2000, 3000, 4000, 5000, 6000, 7000};
        Tensor input(input_values, {2, 3}, device);
        CHECK(all(isinf(exp2(input))));
    };
    runner_all_except_bool(test_exp2_inf);
}

// NOLINTNEXTLINE
TEST_CASE("Unary expm1") {
    // expm1 disabled for bool
    auto test_expm1_bool = []<typename T>(Device device) {
        Tensor input = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = expm1(input), TTException);
    };
    runner_boolean(test_expm1_bool);

    auto test_expm1 = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        std::vector<R> input_values = {1, 2, 3, 4, 5, 6};
        std::vector<R> expected_values(input_values.size());
        std::transform(input_values.begin(), input_values.end(), expected_values.begin(), [](auto v) {
            return std::expm1(static_cast<R>(v));
        });
        Tensor input(input_values, {2, 3}, device);
        Tensor expected(expected_values, {2, 3}, device);
        CHECK(allclose(expm1(input), expected));
    };
    runner_all_except_bool(test_expm1);

    // Large values with expm1 give inf
    auto test_expm1_inf = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        std::vector<R> input_values = {1000, 2000, 3000, 4000, 5000, 6000};
        Tensor input(input_values, {2, 3}, device);
        CHECK(all(isinf(expm1(input))));
    };
    runner_all_except_bool(test_expm1_inf);
}

// NOLINTNEXTLINE
TEST_CASE("Unary sqrt") {
    // sqrt disabled for bool
    auto test_sqrt_bool = []<typename T>(Device device) {
        Tensor input = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = sqrt(input), TTException);
    };
    runner_boolean(test_sqrt_bool);

    auto test_sqrt = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        std::vector<R> input_values = {0, 1, 2, 3, 4, 5};
        std::vector<R> expected_values(input_values.size());
        std::transform(input_values.begin(), input_values.end(), expected_values.begin(), [](auto v) {
            return std::sqrt(static_cast<R>(v));
        });
        Tensor input(input_values, {2, 3}, device);
        Tensor expected(expected_values, {2, 3}, device);
        CHECK(allclose(sqrt(input), expected));
    };
    runner_all_except_bool(test_sqrt);

    // Negative values with sqrt give nan
    auto test_sqrt_inf = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        std::vector<R> input_values = {-1, -2, -3, -4, -5, -6};
        Tensor input(input_values, {2, 3}, device);
        CHECK(all(isnan(sqrt(input))));
    };
    runner_integral(test_sqrt_inf);
    runner_floating_point(test_sqrt_inf);
}

// NOLINTNEXTLINE
TEST_CASE("Unary sin") {
    // sin disabled for bool
    auto test_sin_bool = []<typename T>(Device device) {
        Tensor input = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = sin(input), TTException);
    };
    runner_boolean(test_sin_bool);

    auto test_sin = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        std::vector<T> input_values = {1, 2, 3, 4, 5, 6};
        std::vector<R> expected_values(input_values.size());
        std::transform(input_values.begin(), input_values.end(), expected_values.begin(), [](auto v) {
            return std::sin(static_cast<R>(v));
        });
        Tensor input(input_values, {2, 3}, device);
        Tensor expected(expected_values, {2, 3}, device);
        CHECK(allclose(sin(input), expected));
    };
    runner_all_except_bool(test_sin);
}

// NOLINTNEXTLINE
TEST_CASE("Unary cos") {
    // cos disabled for bool
    auto test_cos_bool = []<typename T>(Device device) {
        Tensor input = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = cos(input), TTException);
    };
    runner_boolean(test_cos_bool);

    auto test_cos = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        std::vector<T> input_values = {1, 2, 3, 4, 5, 6};
        std::vector<R> expected_values(input_values.size());
        std::transform(input_values.begin(), input_values.end(), expected_values.begin(), [](auto v) {
            return std::cos(static_cast<R>(v));
        });
        Tensor input(input_values, {2, 3}, device);
        Tensor expected(expected_values, {2, 3}, device);
        CHECK(allclose(cos(input), expected));
    };
    runner_all_except_bool(test_cos);
}

// NOLINTNEXTLINE
TEST_CASE("Unary tan") {
    // tan disabled for bool
    auto test_tan_bool = []<typename T>(Device device) {
        Tensor input = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = tan(input), TTException);
    };
    runner_boolean(test_tan_bool);

    auto test_tan = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        std::vector<T> input_values = {1, 2, 3, 4, 5, 6};
        std::vector<R> expected_values(input_values.size());
        std::transform(input_values.begin(), input_values.end(), expected_values.begin(), [](auto v) {
            return std::tan(static_cast<R>(v));
        });
        Tensor input(input_values, {2, 3}, device);
        Tensor expected(expected_values, {2, 3}, device);
        CHECK(allclose(tan(input), expected));
    };
    runner_all_except_bool(test_tan);
}

// NOLINTNEXTLINE
TEST_CASE("Unary asin") {
    // asin disabled for bool
    auto test_asin_bool = []<typename T>(Device device) {
        Tensor input = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = asin(input), TTException);
    };
    runner_boolean(test_asin_bool);

    auto test_asin = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        std::vector<T> input_values = {1, 2, 3, 4, 5, 6};
        std::vector<R> expected_values(input_values.size());
        std::transform(input_values.begin(), input_values.end(), expected_values.begin(), [](auto v) {
            return std::asin(static_cast<R>(v));
        });
        Tensor input(input_values, {2, 3}, device);
        Tensor expected(expected_values, {2, 3}, device);
        CHECK(allclose(asin(input), expected, CloseOptions().equal_nan()));
    };
    runner_all_except_bool(test_asin);
}

// NOLINTNEXTLINE
TEST_CASE("Unary acos") {
    // acos disabled for bool
    auto test_acos_bool = []<typename T>(Device device) {
        Tensor input = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = acos(input), TTException);
    };
    runner_boolean(test_acos_bool);

    auto test_acos = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        std::vector<T> input_values = {1, 2, 3, 4, 5, 6};
        std::vector<R> expected_values(input_values.size());
        std::transform(input_values.begin(), input_values.end(), expected_values.begin(), [](auto v) {
            return std::acos(static_cast<R>(v));
        });
        Tensor input(input_values, {2, 3}, device);
        Tensor expected(expected_values, {2, 3}, device);
        CHECK(allclose(acos(input), expected, CloseOptions().equal_nan()));
    };
    runner_all_except_bool(test_acos);
}

// NOLINTNEXTLINE
TEST_CASE("Unary atan") {
    // atan disabled for bool
    auto test_atan_bool = []<typename T>(Device device) {
        Tensor input = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = atan(input), TTException);
    };
    runner_boolean(test_atan_bool);

    auto test_atan = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        std::vector<T> input_values = {1, 2, 3, 4, 5, 6};
        std::vector<R> expected_values(input_values.size());
        std::transform(input_values.begin(), input_values.end(), expected_values.begin(), [](auto v) {
            return std::atan(static_cast<R>(v));
        });
        Tensor input(input_values, {2, 3}, device);
        Tensor expected(expected_values, {2, 3}, device);
        CHECK(allclose(atan(input), expected, CloseOptions().equal_nan()));
    };
    runner_all_except_bool(test_atan);
}

// NOLINTNEXTLINE
TEST_CASE("Unary sinh") {
    // sinh disabled for bool
    auto test_sinh_bool = []<typename T>(Device device) {
        Tensor input = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = sinh(input), TTException);
    };
    runner_boolean(test_sinh_bool);

    auto test_sinh = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        std::vector<T> input_values = {1, 2, 3, 4, 5, 6};
        std::vector<R> expected_values(input_values.size());
        std::transform(input_values.begin(), input_values.end(), expected_values.begin(), [](auto v) {
            return std::sinh(static_cast<R>(v));
        });
        Tensor input(input_values, {2, 3}, device);
        Tensor expected(expected_values, {2, 3}, device);
        CHECK(allclose(sinh(input), expected));
    };
    runner_all_except_bool(test_sinh);
}

// NOLINTNEXTLINE
TEST_CASE("Unary cosh") {
    // cosh disabled for bool
    auto test_cosh_bool = []<typename T>(Device device) {
        Tensor input = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = cosh(input), TTException);
    };
    runner_boolean(test_cosh_bool);

    auto test_cosh = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        std::vector<T> input_values = {1, 2, 3, 4, 5, 6};
        std::vector<R> expected_values(input_values.size());
        std::transform(input_values.begin(), input_values.end(), expected_values.begin(), [](auto v) {
            return std::cosh(static_cast<R>(v));
        });
        Tensor input(input_values, {2, 3}, device);
        Tensor expected(expected_values, {2, 3}, device);
        CHECK(allclose(cosh(input), expected));
    };
    runner_all_except_bool(test_cosh);
}

// NOLINTNEXTLINE
TEST_CASE("Unary tanh") {
    // tan disabled for bool
    auto test_tanh_bool = []<typename T>(Device device) {
        Tensor input = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = tanh(input), TTException);
    };
    runner_boolean(test_tanh_bool);

    auto test_tanh = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        std::vector<T> input_values = {1, 2, 3, 4, 5, 6};
        std::vector<R> expected_values(input_values.size());
        std::transform(input_values.begin(), input_values.end(), expected_values.begin(), [](auto v) {
            return std::tanh(static_cast<R>(v));
        });
        Tensor input(input_values, {2, 3}, device);
        Tensor expected(expected_values, {2, 3}, device);
        CHECK(allclose(tanh(input), expected));
    };
    runner_all_except_bool(test_tanh);
}

// NOLINTNEXTLINE
TEST_CASE("Unary asinh") {
    // asinh disabled for bool
    auto test_asinh_bool = []<typename T>(Device device) {
        Tensor input = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = asinh(input), TTException);
    };
    runner_boolean(test_asinh_bool);

    auto test_asinh = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        std::vector<T> input_values = {1, 2, 3, 4, 5, 6};
        std::vector<R> expected_values(input_values.size());
        std::transform(input_values.begin(), input_values.end(), expected_values.begin(), [](auto v) {
            return std::asinh(static_cast<R>(v));
        });
        Tensor input(input_values, {2, 3}, device);
        Tensor expected(expected_values, {2, 3}, device);
        CHECK(allclose(asinh(input), expected));
    };
    runner_all_except_bool(test_asinh);
}

// NOLINTNEXTLINE
TEST_CASE("Unary acosh") {
    // acosh disabled for bool
    auto test_acosh_bool = []<typename T>(Device device) {
        Tensor input = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = acosh(input), TTException);
    };
    runner_boolean(test_acosh_bool);

    auto test_acosh = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        std::vector<T> input_values = {1, 2, 3, 4, 5, 6};
        std::vector<R> expected_values(input_values.size());
        std::transform(input_values.begin(), input_values.end(), expected_values.begin(), [](auto v) {
            return std::acosh(static_cast<R>(v));
        });
        Tensor input(input_values, {2, 3}, device);
        Tensor expected(expected_values, {2, 3}, device);
        CHECK(allclose(acosh(input), expected));
    };
    runner_all_except_bool(test_acosh);
}

// NOLINTNEXTLINE
TEST_CASE("Unary atanh") {
    // atanh disabled for bool
    auto test_atanh_bool = []<typename T>(Device device) {
        Tensor input = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = atanh(input), TTException);
    };
    runner_boolean(test_atanh_bool);

    auto test_atanh = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        std::vector<T> input_values = {1, 2, 3, 4, 5, 6};
        std::vector<R> expected_values(input_values.size());
        std::transform(input_values.begin(), input_values.end(), expected_values.begin(), [](auto v) {
            return std::atanh(static_cast<R>(v));
        });
        Tensor input(input_values, {2, 3}, device);
        Tensor expected(expected_values, {2, 3}, device);
        CHECK(allclose(atanh(input), expected, CloseOptions().equal_nan()));
    };
    runner_all_except_bool(test_atanh);
}

// NOLINTNEXTLINE
TEST_CASE("Unary erf") {
    // erf disabled for bool
    auto test_erf_bool = []<typename T>(Device device) {
        Tensor input = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = erf(input), TTException);
    };
    runner_boolean(test_erf_bool);

    auto test_erf = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        std::vector<T> input_values = {1, 2, 3, 4, 5, 6};
        std::vector<R> expected_values(input_values.size());
        std::transform(input_values.begin(), input_values.end(), expected_values.begin(), [](auto v) {
            return std::erf(static_cast<R>(v));
        });
        Tensor input(input_values, {2, 3}, device);
        Tensor expected(expected_values, {2, 3}, device);
        CHECK(allclose(erf(input), expected));
    };
    runner_all_except_bool(test_erf);
}

// NOLINTNEXTLINE
TEST_CASE("Unary erfc") {
    // erfc disabled for bool
    auto test_erfc_bool = []<typename T>(Device device) {
        Tensor input = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = erfc(input), TTException);
    };
    runner_boolean(test_erfc_bool);

    auto test_erfc = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        std::vector<T> input_values = {1, 2, 3, 4, 5, 6};
        std::vector<R> expected_values(input_values.size());
        std::transform(input_values.begin(), input_values.end(), expected_values.begin(), [](auto v) {
            return std::erfc(static_cast<R>(v));
        });
        Tensor input(input_values, {2, 3}, device);
        Tensor expected(expected_values, {2, 3}, device);
        CHECK(allclose(erfc(input), expected));
    };
    runner_all_except_bool(test_erfc);
}

// NOLINTNEXTLINE
TEST_CASE("Unary tgamma") {
    // tgamma disabled for bool
    auto test_tgamma_bool = []<typename T>(Device device) {
        Tensor input = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = tgamma(input), TTException);
    };
    runner_boolean(test_tgamma_bool);

    auto test_tgamma = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        std::vector<T> input_values = {1, 2, 3, 4, 5, 6};
        std::vector<R> expected_values(input_values.size());
        std::transform(input_values.begin(), input_values.end(), expected_values.begin(), [](auto v) {
            return std::tgamma(static_cast<R>(v));
        });
        Tensor input(input_values, {2, 3}, device);
        Tensor expected(expected_values, {2, 3}, device);
        // CUDA tgamma isn't as accurate as std
        CHECK(allclose(tgamma(input), expected));
    };
    runner_all_except_bool(test_tgamma);
}

// NOLINTNEXTLINE
TEST_CASE("Unary lgamma") {
    // lgamma disabled for bool
    auto test_lgamma_bool = []<typename T>(Device device) {
        Tensor input = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = lgamma(input), TTException);
    };
    runner_boolean(test_lgamma_bool);

    auto test_lgamma = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        std::vector<T> input_values = {1, 2, 3, 4, 5, 6};
        std::vector<R> expected_values(input_values.size());
        std::transform(input_values.begin(), input_values.end(), expected_values.begin(), [](auto v) {
            return std::lgamma(static_cast<R>(v));
        });
        Tensor input(input_values, {2, 3}, device);
        Tensor expected(expected_values, {2, 3}, device);
        CHECK(allclose(lgamma(input), expected));
    };
    runner_all_except_bool(test_lgamma);
}

// NOLINTNEXTLINE
TEST_CASE("Unary ceil") {
    // ceil disabled for bool
    auto test_ceil_bool = []<typename T>(Device device) {
        Tensor input = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = ceil(input), TTException);
    };
    runner_boolean(test_ceil_bool);

    auto test_ceil = []<typename T>(Device device) {
        std::vector<T> input_values = {SC<T>(1.5), SC<T>(2.1), SC<T>(3.2), SC<T>(4.7), SC<T>(5.8), SC<T>(6.9)};
        std::vector<T> expected_values(input_values.size());
        std::transform(input_values.begin(), input_values.end(), expected_values.begin(), [](auto v) {
            return std::ceil(v);
        });
        Tensor input(input_values, {2, 3}, device);
        Tensor expected(expected_values, {2, 3}, device);
        CHECK(allclose(ceil(input), expected));
    };
    runner_all_except_bool(test_ceil);
}

// NOLINTNEXTLINE
TEST_CASE("Unary floor") {
    // floor disabled for bool
    auto test_floor_bool = []<typename T>(Device device) {
        Tensor input = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = floor(input), TTException);
    };
    runner_boolean(test_floor_bool);

    auto test_floor = []<typename T>(Device device) {
        std::vector<T> input_values = {SC<T>(1.5), SC<T>(2.1), SC<T>(3.2), SC<T>(4.7), SC<T>(5.8), SC<T>(6.9)};
        std::vector<T> expected_values(input_values.size());
        std::transform(input_values.begin(), input_values.end(), expected_values.begin(), [](auto v) {
            return std::floor(v);
        });
        Tensor input(input_values, {2, 3}, device);
        Tensor expected(expected_values, {2, 3}, device);
        CHECK(allclose(floor(input), expected));
    };
    runner_all_except_bool(test_floor);
}

// NOLINTNEXTLINE
TEST_CASE("Unary round") {
    // round disabled for bool
    auto test_round_bool = []<typename T>(Device device) {
        Tensor input = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = round(input), TTException);
    };
    runner_boolean(test_round_bool);

    auto test_round = []<typename T>(Device device) {
        std::vector<T> input_values = {SC<T>(1.5), SC<T>(2.1), SC<T>(3.2), SC<T>(4.7), SC<T>(5.8), SC<T>(6.9)};
        std::vector<T> expected_values(input_values.size());
        std::transform(input_values.begin(), input_values.end(), expected_values.begin(), [](auto v) {
            return std::round(v);
        });
        Tensor input(input_values, {2, 3}, device);
        Tensor expected(expected_values, {2, 3}, device);
        CHECK(allclose(round(input), expected));
    };
    runner_all_except_bool(test_round);
}

// NOLINTNEXTLINE
TEST_CASE("Unary isinf") {
    // isinf disabled for bool
    auto test_isinf_bool = []<typename T>(Device device) {
        Tensor input = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = isinf(input), TTException);
    };
    runner_boolean(test_isinf_bool);

    // Integral values are never infinity
    auto test_infinity_all = []<typename T>(Device device) {
        std::vector<T> input_values(6, std::numeric_limits<T>::infinity());
        std::vector<bool> expected_values(input_values.size(), std::is_floating_point_v<T> ? true : false);
        Tensor input(input_values, {2, 3}, device);
        Tensor expected(expected_values, {2, 3}, device);
        CHECK(allclose(isinf(input), expected));
    };
    runner_all_except_bool(test_infinity_all);

    auto test_infinity_none = []<typename T>(Device device) {
        std::vector<T> input_values = {1, 2, 3, 4, 5, 6};
        std::vector<bool> expected_values(input_values.size(), false);
        Tensor input(input_values, {2, 3}, device);
        Tensor expected(expected_values, {2, 3}, device);
        CHECK(allclose(isinf(input), expected));
    };
    runner_all_except_bool(test_infinity_none);
}

// NOLINTNEXTLINE
TEST_CASE("Unary isnan") {
    // isnan disabled for bool
    auto test_isnan_bool = []<typename T>(Device device) {
        Tensor input = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = isnan(input), TTException);
    };
    runner_boolean(test_isnan_bool);

    // Integral values are never nan
    auto test_isnan_all = []<typename T>(Device device) {
        std::vector<T> input_values(6, std::numeric_limits<T>::quiet_NaN());
        std::vector<bool> expected_values(input_values.size(), std::is_floating_point_v<T> ? true : false);
        Tensor input(input_values, {2, 3}, device);
        Tensor expected(expected_values, {2, 3}, device);
        CHECK(allclose(isnan(input), expected));
    };
    runner_all_except_bool(test_isnan_all);

    auto test_isnan_none = []<typename T>(Device device) {
        std::vector<T> input_values = {1, 2, 3, 4, 5, 6};
        std::vector<bool> expected_values(input_values.size(), false);
        Tensor input(input_values, {2, 3}, device);
        Tensor expected(expected_values, {2, 3}, device);
        CHECK(allclose(isnan(input), expected));
    };
    runner_all_except_bool(test_isnan_none);
}

// NOLINTNEXTLINE
TEST_CASE("Unary isfinite") {
    // isfinite disabled for bool
    auto test_isfinite_bool = []<typename T>(Device device) {
        Tensor input = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = isfinite(input), TTException);
    };
    runner_boolean(test_isfinite_bool);

    auto test_isfinite_all = []<typename T>(Device device) {
        std::vector<T> input_values = {1, 2, 3, 4, 5, 6};
        std::vector<bool> expected_values(input_values.size(), true);
        Tensor input(input_values, {2, 3}, device);
        Tensor expected(expected_values, {2, 3}, device);
        CHECK(allclose(isfinite(input), expected));
    };
    runner_all_except_bool(test_isfinite_all);

    // Integral values are always finite
    auto test_isfinite_none = []<typename T>(Device device) {
        std::vector<T> input_values(6, std::numeric_limits<T>::infinity());
        std::vector<bool> expected_values(input_values.size(), std::is_floating_point_v<T> ? false : true);
        Tensor input(input_values, {2, 3}, device);
        Tensor expected(expected_values, {2, 3}, device);
        CHECK(allclose(isfinite(input), expected));
    };
    runner_all_except_bool(test_isfinite_none);
}

// NOLINTNEXTLINE
TEST_CASE("Unary sigmoid") {
    // sigmoid disabled for bool
    auto test_sigmoid_bool = []<typename T>(Device device) {
        Tensor input = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = sigmoid(input), TTException);
    };
    runner_boolean(test_sigmoid_bool);

    auto test_sigmoid = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        std::vector<T> input_values = {2, 1, 0, 1, 2, 3};
        std::vector<R> expected_values = {0.880797, 0.731059, 0.5, 0.731059, 0.880797, 0.952574};
        if constexpr (!std::is_same_v<T, uint8_t>) {
            for (std::size_t i = 0; i < 2; ++i) {
                if (input_values[i] < 0) {
                    input_values[i] *= -1;
                    expected_values[i] = 1 - expected_values[i];
                }
            }
        }
        Tensor input(input_values, {2, 3}, device);
        Tensor expected(expected_values, {2, 3}, device);
        CHECK(allclose(sigmoid(input), expected));
    };
    runner_all_except_bool(test_sigmoid);
    auto test_sigmoid_extreme = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        std::vector<T> input_values = {-200000000, -1, 0, 1, 2, 200000000};
        std::vector<R> expected_values = {0, 0.268941, 0.5, 0.731059, 0.880797, 1};
        Tensor input(input_values, {2, 3}, device);
        Tensor expected(expected_values, {2, 3}, device);
        CHECK(allclose(sigmoid(input), expected));
    };
    runner_floating_point(test_sigmoid_extreme);
}

// NOLINTNEXTLINE
TEST_CASE("Unary logsigmoid") {
    // log_sigmoid disabled for bool
    auto test_logsigmoid_bool = []<typename T>(Device device) {
        Tensor input = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = log_sigmoid(input), TTException);
    };
    runner_boolean(test_logsigmoid_bool);
}

// NOLINTNEXTLINE
TEST_CASE("Unary hardsigmoid") {
    // hardsigmoid disabled for bool
    auto test_hardsigmoid_bool = []<typename T>(Device device) {
        Tensor input = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = hardsigmoid(input), TTException);
    };
    runner_boolean(test_hardsigmoid_bool);
}

// NOLINTNEXTLINE
TEST_CASE("Unary relu") {
    // hardsigmoid disabled for bool
    auto test_relu_bool = []<typename T>(Device device) {
        Tensor input = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = relu(input), TTException);
    };
    runner_boolean(test_relu_bool);
}

// NOLINTNEXTLINE
TEST_CASE("Unary relu6") {
    // relu6 disabled for bool
    auto test_relu6_bool = []<typename T>(Device device) {
        Tensor input = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = relu(input), TTException);
    };
    runner_boolean(test_relu6_bool);
}

// NOLINTNEXTLINE
TEST_CASE("Unary leaky_relu") {
    // leaky_relu disabled for bool
    auto test_leaky_relu_bool = []<typename T>(Device device) {
        Tensor input = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = leaky_relu(input), TTException);
    };
    runner_boolean(test_leaky_relu_bool);
}

// NOLINTNEXTLINE
TEST_CASE("Unary elu") {
    // elu disabled for bool
    auto test_elu_bool = []<typename T>(Device device) {
        Tensor input = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = elu(input), TTException);
    };
    runner_boolean(test_elu_bool);
}

// NOLINTNEXTLINE
TEST_CASE("Unary selu") {
    // selu disabled for bool
    auto test_selu_bool = []<typename T>(Device device) {
        Tensor input = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = selu(input), TTException);
    };
    runner_boolean(test_selu_bool);
}

// NOLINTNEXTLINE
TEST_CASE("Unary silu") {
    // silu disabled for bool
    auto test_silu_bool = []<typename T>(Device device) {
        Tensor input = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = silu(input), TTException);
    };
    runner_boolean(test_silu_bool);
}

// NOLINTNEXTLINE
TEST_CASE("Unary hardtanh") {
    // hardtanh disabled for bool
    auto test_hardtanh_bool = []<typename T>(Device device) {
        Tensor input = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = hardtanh(input), TTException);
    };
    runner_boolean(test_hardtanh_bool);
}

// NOLINTNEXTLINE
TEST_CASE("Unary softsign") {
    // softsign disabled for bool
    auto test_softsign_bool = []<typename T>(Device device) {
        Tensor input = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = softsign(input), TTException);
    };
    runner_boolean(test_softsign_bool);
}

// NOLINTNEXTLINE
TEST_CASE("Unary softplus") {
    // softplus disabled for bool
    auto test_softplus_bool = []<typename T>(Device device) {
        Tensor input = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = softplus(input), TTException);
    };
    runner_boolean(test_softplus_bool);
}

// NOLINTNEXTLINE
TEST_CASE("Unary softmax") {
    // softmax disabled for bool
    auto test_softmax_bool = []<typename T>(Device device) {
        Tensor input = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = softmax(input, 0), TTException);
    };
    runner_boolean(test_softmax_bool);
}

#ifdef TT_TORCH
// NOLINTNEXTLINE
TEST_CASE("Unary sigmoid Torch") {
    auto test1 = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> _X = rand_vec<T>(4 * 4 * 18, gen);
        Tensor X(_X, {4, 4, 18}, device);

        const auto options_float = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor X_torch = torch::from_blob(_X.data(), {4, 4, 18}, options_float);
        torch::Tensor expected_torch = torch::sigmoid(X_torch);
        std::vector<T> expected_torch_data(
            expected_torch.data_ptr<T>(),
            expected_torch.data_ptr<T>() + expected_torch.numel()
        );

        Tensor expected(expected_torch_data, {4, 4, 18}, device);
        CHECK(allclose(tinytensor::sigmoid(X), expected, CloseOptions().atol(1e-6)));
    };
    runner_single_type<float>(test1);
}

// NOLINTNEXTLINE
TEST_CASE("Unary logsigmoid Torch") {
    auto test1 = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> _X = rand_vec<T>(4 * 4 * 18, gen);
        Tensor X(_X, {4, 4, 18}, device);

        const auto options_float = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor X_torch = torch::from_blob(_X.data(), {4, 4, 18}, options_float);
        torch::Tensor expected_torch = torch::log_sigmoid(X_torch);
        std::vector<T> expected_torch_data(
            expected_torch.data_ptr<T>(),
            expected_torch.data_ptr<T>() + expected_torch.numel()
        );

        Tensor expected(expected_torch_data, {4, 4, 18}, device);
        CHECK(allclose(tinytensor::log_sigmoid(X), expected, CloseOptions().atol(1e-6)));
    };
    runner_single_type<float>(test1);
}

// NOLINTNEXTLINE
TEST_CASE("Unary hardsigmoid Torch") {
    auto test1 = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> _X = rand_vec<T>(4 * 4 * 18, gen);
        Tensor X(_X, {4, 4, 18}, device);

        const auto options_float = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor X_torch = torch::from_blob(_X.data(), {4, 4, 18}, options_float);
        torch::Tensor expected_torch = torch::hardsigmoid(X_torch);
        std::vector<T> expected_torch_data(
            expected_torch.data_ptr<T>(),
            expected_torch.data_ptr<T>() + expected_torch.numel()
        );

        Tensor expected(expected_torch_data, {4, 4, 18}, device);
        CHECK(allclose(tinytensor::hardsigmoid(X), expected, CloseOptions().atol(1e-6)));
    };
    runner_single_type<float>(test1);
}

// NOLINTNEXTLINE
TEST_CASE("Unary softplus Torch") {
    auto test1 = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> _X = rand_vec<T>(4 * 4 * 18, gen);
        Tensor X(_X, {4, 4, 18}, device);

        const auto options_float = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor X_torch = torch::from_blob(_X.data(), {4, 4, 18}, options_float);
        torch::Tensor expected_torch = torch::softplus(X_torch);
        std::vector<T> expected_torch_data(
            expected_torch.data_ptr<T>(),
            expected_torch.data_ptr<T>() + expected_torch.numel()
        );

        Tensor expected(expected_torch_data, {4, 4, 18}, device);
        CHECK(allclose(tinytensor::softplus(X), expected, CloseOptions().atol(1e-6)));
    };
    runner_single_type<float>(test1);

    auto test2 = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> _X = rand_vec<T>(4 * 4 * 18, gen);
        Tensor X(_X, {4, 4, 18}, device);

        const auto options_float = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor X_torch = torch::from_blob(_X.data(), {4, 4, 18}, options_float);
        torch::Tensor expected_torch = torch::softplus(X_torch, 0.5, 3);
        std::vector<T> expected_torch_data(
            expected_torch.data_ptr<T>(),
            expected_torch.data_ptr<T>() + expected_torch.numel()
        );

        Tensor expected(expected_torch_data, {4, 4, 18}, device);
        CHECK(allclose(tinytensor::softplus(X, 0.5, 3), expected, CloseOptions().atol(1e-6)));
    };
    runner_single_type<float>(test2);
}

// NOLINTNEXTLINE
TEST_CASE("Unary relu Torch") {
    auto test1 = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> _X = rand_vec<T>(4 * 4 * 18, gen);
        Tensor X(_X, {4, 4, 18}, device);

        const auto options_float = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor X_torch = torch::from_blob(_X.data(), {4, 4, 18}, options_float);
        torch::Tensor expected_torch = torch::relu(X_torch);
        std::vector<T> expected_torch_data(
            expected_torch.data_ptr<T>(),
            expected_torch.data_ptr<T>() + expected_torch.numel()
        );

        Tensor expected(expected_torch_data, {4, 4, 18}, device);
        CHECK(allclose(tinytensor::relu(X), expected, CloseOptions().atol(1e-6)));
    };
    runner_single_type<float>(test1);
}

// NOLINTNEXTLINE
TEST_CASE("Unary relu6 Torch") {
    auto test1 = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> _X = rand_vec<T>(4 * 4 * 18, gen);
        Tensor X(_X, {4, 4, 18}, device);

        const auto options_float = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor X_torch = torch::from_blob(_X.data(), {4, 4, 18}, options_float);
        torch::Tensor expected_torch = torch::relu6(X_torch);
        std::vector<T> expected_torch_data(
            expected_torch.data_ptr<T>(),
            expected_torch.data_ptr<T>() + expected_torch.numel()
        );

        Tensor expected(expected_torch_data, {4, 4, 18}, device);
        CHECK(allclose(tinytensor::relu6(X), expected, CloseOptions().atol(1e-6)));
    };
    runner_single_type<float>(test1);
}

// NOLINTNEXTLINE
TEST_CASE("Unary leaky_relu Torch") {
    auto test1 = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> _X = rand_vec<T>(4 * 4 * 18, gen);
        Tensor X(_X, {4, 4, 18}, device);

        const auto options_float = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor X_torch = torch::from_blob(_X.data(), {4, 4, 18}, options_float);
        torch::Tensor expected_torch = torch::leaky_relu(X_torch);
        std::vector<T> expected_torch_data(
            expected_torch.data_ptr<T>(),
            expected_torch.data_ptr<T>() + expected_torch.numel()
        );

        Tensor expected(expected_torch_data, {4, 4, 18}, device);
        CHECK(allclose(tinytensor::leaky_relu(X), expected, CloseOptions().atol(1e-6)));
    };
    runner_single_type<float>(test1);

    auto test2 = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> _X = rand_vec<T>(4 * 4 * 18, gen);
        Tensor X(_X, {4, 4, 18}, device);

        const auto options_float = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor X_torch = torch::from_blob(_X.data(), {4, 4, 18}, options_float);
        torch::Tensor expected_torch = torch::leaky_relu(X_torch, 0.1);
        std::vector<T> expected_torch_data(
            expected_torch.data_ptr<T>(),
            expected_torch.data_ptr<T>() + expected_torch.numel()
        );

        Tensor expected(expected_torch_data, {4, 4, 18}, device);
        CHECK(allclose(tinytensor::leaky_relu(X, 0.1), expected, CloseOptions().atol(1e-6)));
    };
    runner_single_type<float>(test2);
}

// NOLINTNEXTLINE
TEST_CASE("Unary elu Torch") {
    auto test1 = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> _X = rand_vec<T>(4 * 4 * 18, gen);
        Tensor X(_X, {4, 4, 18}, device);

        const auto options_float = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor X_torch = torch::from_blob(_X.data(), {4, 4, 18}, options_float);
        torch::Tensor expected_torch = torch::elu(X_torch);
        std::vector<T> expected_torch_data(
            expected_torch.data_ptr<T>(),
            expected_torch.data_ptr<T>() + expected_torch.numel()
        );

        Tensor expected(expected_torch_data, {4, 4, 18}, device);
        CHECK(allclose(tinytensor::elu(X), expected, CloseOptions().atol(1e-6)));
    };
    runner_single_type<float>(test1);

    auto test2 = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> _X = rand_vec<T>(4 * 4 * 18, gen);
        Tensor X(_X, {4, 4, 18}, device);

        const auto options_float = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor X_torch = torch::from_blob(_X.data(), {4, 4, 18}, options_float);
        torch::Tensor expected_torch = torch::elu(X_torch, 2);
        std::vector<T> expected_torch_data(
            expected_torch.data_ptr<T>(),
            expected_torch.data_ptr<T>() + expected_torch.numel()
        );

        Tensor expected(expected_torch_data, {4, 4, 18}, device);
        CHECK(allclose(tinytensor::elu(X, 2), expected, CloseOptions().atol(1e-6)));
    };
    runner_single_type<float>(test2);
}

// NOLINTNEXTLINE
TEST_CASE("Unary selu Torch") {
    auto test1 = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> _X = rand_vec<T>(4 * 4 * 18, gen);
        Tensor X(_X, {4, 4, 18}, device);

        const auto options_float = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor X_torch = torch::from_blob(_X.data(), {4, 4, 18}, options_float);
        torch::Tensor expected_torch = torch::selu(X_torch);
        std::vector<T> expected_torch_data(
            expected_torch.data_ptr<T>(),
            expected_torch.data_ptr<T>() + expected_torch.numel()
        );

        Tensor expected(expected_torch_data, {4, 4, 18}, device);
        CHECK(allclose(tinytensor::selu(X), expected, CloseOptions().atol(1e-6)));
    };
    runner_single_type<float>(test1);
}

// NOLINTNEXTLINE
TEST_CASE("Unary silu Torch") {
    auto test1 = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> _X = rand_vec<T>(4 * 4 * 18, gen);
        Tensor X(_X, {4, 4, 18}, device);

        const auto options_float = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor X_torch = torch::from_blob(_X.data(), {4, 4, 18}, options_float);
        torch::Tensor expected_torch = torch::silu(X_torch);
        std::vector<T> expected_torch_data(
            expected_torch.data_ptr<T>(),
            expected_torch.data_ptr<T>() + expected_torch.numel()
        );

        Tensor expected(expected_torch_data, {4, 4, 18}, device);
        CHECK(allclose(tinytensor::silu(X), expected, CloseOptions().atol(1e-6)));
    };
    runner_single_type<float>(test1);
}

// NOLINTNEXTLINE
TEST_CASE("Unary hardtanh Torch") {
    auto test1 = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> _X = rand_vec<T>(4 * 4 * 18, gen);
        Tensor X(_X, {4, 4, 18}, device);

        const auto options_float = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor X_torch = torch::from_blob(_X.data(), {4, 4, 18}, options_float);
        torch::Tensor expected_torch = torch::hardtanh(X_torch);
        std::vector<T> expected_torch_data(
            expected_torch.data_ptr<T>(),
            expected_torch.data_ptr<T>() + expected_torch.numel()
        );

        Tensor expected(expected_torch_data, {4, 4, 18}, device);
        CHECK(allclose(tinytensor::hardtanh(X), expected, CloseOptions().atol(1e-6)));
    };
    runner_single_type<float>(test1);

    auto test2 = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> _X = rand_vec<T>(4 * 4 * 18, gen);
        Tensor X(_X, {4, 4, 18}, device);

        const auto options_float = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor X_torch = torch::from_blob(_X.data(), {4, 4, 18}, options_float);
        torch::Tensor expected_torch = torch::hardtanh(X_torch, -0.5, 0.5);
        std::vector<T> expected_torch_data(
            expected_torch.data_ptr<T>(),
            expected_torch.data_ptr<T>() + expected_torch.numel()
        );

        Tensor expected(expected_torch_data, {4, 4, 18}, device);
        CHECK(allclose(tinytensor::hardtanh(X, -0.5, 0.5), expected, CloseOptions().atol(1e-6)));
    };
    runner_single_type<float>(test2);
}

// NOLINTNEXTLINE
TEST_CASE("Unary softsign Torch") {
    auto test1 = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> _X = rand_vec<T>(4 * 4 * 18, gen);
        Tensor X(_X, {4, 4, 18}, device);

        const auto options_float = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor X_torch = torch::from_blob(_X.data(), {4, 4, 18}, options_float);
        torch::Tensor expected_torch = torch::nn::functional::softsign(X_torch);

        std::vector<T> expected_torch_data(
            expected_torch.data_ptr<T>(),
            expected_torch.data_ptr<T>() + expected_torch.numel()
        );

        Tensor expected(expected_torch_data, {4, 4, 18}, device);
        CHECK(allclose(tinytensor::softsign(X), expected, CloseOptions().atol(1e-6)));
    };
    runner_single_type<float>(test1);
}

// NOLINTNEXTLINE
TEST_CASE("Unary softmax Torch") {
    auto test1 = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> _X = rand_vec<T>(4 * 4 * 18, gen);
        Tensor X(_X, {4, 4, 18}, device);

        const auto options_float = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor X_torch = torch::from_blob(_X.data(), {4, 4, 18}, options_float);
        torch::Tensor expected_torch = torch::nn::functional::softmax(X_torch, 0);

        std::vector<T> expected_torch_data(
            expected_torch.data_ptr<T>(),
            expected_torch.data_ptr<T>() + expected_torch.numel()
        );

        Tensor expected(expected_torch_data, {4, 4, 18}, device);
        CHECK(allclose(tinytensor::softmax(X, 0), expected, CloseOptions().atol(1e-6)));
    };
    runner_single_type<float>(test1);

    auto test2 = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> _X = rand_vec<T>(4 * 4 * 18, gen);
        Tensor X(_X, {4, 4, 18}, device);

        const auto options_float = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor X_torch = torch::from_blob(_X.data(), {4, 4, 18}, options_float);
        torch::Tensor expected_torch = torch::nn::functional::softmax(X_torch, 1);

        std::vector<T> expected_torch_data(
            expected_torch.data_ptr<T>(),
            expected_torch.data_ptr<T>() + expected_torch.numel()
        );

        Tensor expected(expected_torch_data, {4, 4, 18}, device);
        CHECK(allclose(tinytensor::softmax(X, 1), expected, CloseOptions().atol(1e-6)));
    };
    runner_single_type<float>(test2);

    auto test3 = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> _X = rand_vec<T>(4 * 4 * 18, gen);
        Tensor X(_X, {4, 4, 18}, device);

        const auto options_float = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor X_torch = torch::from_blob(_X.data(), {4, 4, 18}, options_float);
        torch::Tensor expected_torch = torch::nn::functional::softmax(X_torch, 2);

        std::vector<T> expected_torch_data(
            expected_torch.data_ptr<T>(),
            expected_torch.data_ptr<T>() + expected_torch.numel()
        );

        Tensor expected(expected_torch_data, {4, 4, 18}, device);
        CHECK(allclose(tinytensor::softmax(X, 2), expected, CloseOptions().atol(1e-6)));
    };
    runner_single_type<float>(test3);
}

// NOLINTNEXTLINE
TEST_CASE("Unary log_softmax Torch") {
    auto test1 = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> _X = rand_vec<T>(4 * 4 * 18, gen);
        Tensor X(_X, {4, 4, 18}, device);

        const auto options_float = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor X_torch = torch::from_blob(_X.data(), {4, 4, 18}, options_float);
        torch::Tensor expected_torch = torch::nn::functional::log_softmax(X_torch, 0);

        std::vector<T> expected_torch_data(
            expected_torch.data_ptr<T>(),
            expected_torch.data_ptr<T>() + expected_torch.numel()
        );

        Tensor expected(expected_torch_data, {4, 4, 18}, device);
        CHECK(allclose(tinytensor::log_softmax(X, 0), expected, CloseOptions().atol(1e-6)));
    };
    runner_single_type<float>(test1);

    auto test2 = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> _X = rand_vec<T>(4 * 4 * 18, gen);
        Tensor X(_X, {4, 4, 18}, device);

        const auto options_float = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor X_torch = torch::from_blob(_X.data(), {4, 4, 18}, options_float);
        torch::Tensor expected_torch = torch::nn::functional::log_softmax(X_torch, 1);

        std::vector<T> expected_torch_data(
            expected_torch.data_ptr<T>(),
            expected_torch.data_ptr<T>() + expected_torch.numel()
        );

        Tensor expected(expected_torch_data, {4, 4, 18}, device);
        CHECK(allclose(tinytensor::log_softmax(X, 1), expected, CloseOptions().atol(1e-6)));
    };
    runner_single_type<float>(test2);

    auto test3 = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> _X = rand_vec<T>(4 * 4 * 18, gen);
        Tensor X(_X, {4, 4, 18}, device);

        const auto options_float = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor X_torch = torch::from_blob(_X.data(), {4, 4, 18}, options_float);
        torch::Tensor expected_torch = torch::nn::functional::log_softmax(X_torch, 2);

        std::vector<T> expected_torch_data(
            expected_torch.data_ptr<T>(),
            expected_torch.data_ptr<T>() + expected_torch.numel()
        );

        Tensor expected(expected_torch_data, {4, 4, 18}, device);
        CHECK(allclose(tinytensor::log_softmax(X, 2), expected, CloseOptions().atol(1e-6)));
    };
    runner_single_type<float>(test3);
}
#endif
