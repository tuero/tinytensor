// test_unary_inplace.cpp
// Test the unary inplace ops

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
TEST_CASE("Unary abs inplace") {
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
        auto data_ptr = input.data_ptr();
        CHECK_NOTHROW(input.abs_());
        CHECK_EQ(input.data_ptr(), data_ptr);
        CHECK(allclose(input, expected));
    };
    runner_floating_point(test_abs);
    runner_signed_integral(test_abs);
}

// NOLINTNEXTLINE
TEST_CASE("Unary logical_not inplace") {
    auto test_logical_not = []<typename T>(Device device) {
        std::vector<bool> input_values = {true, false, false, true, true, false};
        Tensor input(input_values, {2, 3}, device);

        std::vector<bool> expected_values = {false, true, true, false, false, true};
        Tensor expected(expected_values, {2, 3}, device);
        auto data_ptr = input.data_ptr();
        CHECK_NOTHROW(input.logical_not_());
        CHECK_EQ(input.data_ptr(), data_ptr);
        CHECK(allclose(input, expected));
    };
    runner_boolean(test_logical_not);
}

// NOLINTNEXTLINE
TEST_CASE("Unary sign inplace") {
    // sign inplace disabled for bool
    auto test_sign_bool = []<typename T>(Device device) {
        Tensor input = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = input.sign_(), TTException);
    };
    runner_boolean(test_sign_bool);

    auto test_sign = []<typename T>(Device device) {
        std::vector<T> input_values = {1, -2, -3, 4, 5, 0};
        std::vector<T> expected_values = {1, -1, -1, 1, 1, 0};
        Tensor input(input_values, {2, 3}, device);
        Tensor expected(expected_values, {2, 3}, device);
        auto data_ptr = input.data_ptr();
        CHECK_NOTHROW(input.sign_());
        CHECK_EQ(input.data_ptr(), data_ptr);
        CHECK(allclose(input, expected));
    };
    runner_signed_integral(test_sign);
    runner_floating_point(test_sign);
}

// NOLINTNEXTLINE
TEST_CASE("Unary log inplace") {
    auto test_log = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        if constexpr (std::is_same_v<T, R>) {
            std::vector<T> input_values = {1, 2, 3, 4, 5, 6};
            std::vector<T> expected_values(input_values.size());
            std::transform(input_values.begin(), input_values.end(), expected_values.begin(), [](auto v) {
                return std::log(v);
            });
            Tensor input(input_values, {2, 3}, device);
            auto data_ptr = input.data_ptr();
            input.log_();
            Tensor expected(expected_values, {2, 3}, device);
            CHECK(allclose(input, expected));
            CHECK_EQ(input.data_ptr(), data_ptr);
        } else {
            Tensor input = full(true, {2, 3}, device);
            CHECK_THROWS_AS(input.log_(), TTException);
        }
    };
    runner_all(test_log);
}

// NOLINTNEXTLINE
TEST_CASE("Unary log10 inplace") {
    auto test_log10 = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        if constexpr (std::is_same_v<T, R>) {
            std::vector<T> input_values = {1, 2, 3, 4, 5, 6};
            std::vector<T> expected_values(input_values.size());
            std::transform(input_values.begin(), input_values.end(), expected_values.begin(), [](auto v) {
                return std::log10(v);
            });
            Tensor input(input_values, {2, 3}, device);
            auto data_ptr = input.data_ptr();
            input.log10_();
            Tensor expected(expected_values, {2, 3}, device);
            CHECK(allclose(input, expected));
            CHECK_EQ(input.data_ptr(), data_ptr);
        } else {
            Tensor input = full(true, {2, 3}, device);
            CHECK_THROWS_AS(input.log10_(), TTException);
        }
    };
    runner_all(test_log10);
}

// NOLINTNEXTLINE
TEST_CASE("Unary log2 inplace") {
    auto test_log2 = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        if constexpr (std::is_same_v<T, R>) {
            std::vector<T> input_values = {1, 2, 3, 4, 5, 6};
            std::vector<T> expected_values(input_values.size());
            std::transform(input_values.begin(), input_values.end(), expected_values.begin(), [](auto v) {
                return std::log2(v);
            });
            Tensor input(input_values, {2, 3}, device);
            auto data_ptr = input.data_ptr();
            input.log2_();
            Tensor expected(expected_values, {2, 3}, device);
            CHECK(allclose(input, expected));
            CHECK_EQ(input.data_ptr(), data_ptr);
        } else {
            Tensor input = full(true, {2, 3}, device);
            CHECK_THROWS_AS(input.log2_(), TTException);
        }
    };
    runner_all(test_log2);
}

// NOLINTNEXTLINE
TEST_CASE("Unary log1p inplace") {
    auto test_log1p = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        if constexpr (std::is_same_v<T, R>) {
            std::vector<T> input_values = {1, 2, 3, 4, 5, 6};
            std::vector<T> expected_values(input_values.size());
            std::transform(input_values.begin(), input_values.end(), expected_values.begin(), [](auto v) {
                return std::log1p(v);
            });
            Tensor input(input_values, {2, 3}, device);
            auto data_ptr = input.data_ptr();
            input.log1p_();
            Tensor expected(expected_values, {2, 3}, device);
            CHECK(allclose(input, expected));
            CHECK_EQ(input.data_ptr(), data_ptr);
        } else {
            Tensor input = full(true, {2, 3}, device);
            CHECK_THROWS_AS(input.log1p_(), TTException);
        }
    };
    runner_all(test_log1p);
}

// NOLINTNEXTLINE
TEST_CASE("Unary exp inplace") {
    auto test_exp = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        if constexpr (std::is_same_v<T, R>) {
            std::vector<T> input_values = {1, 2, 3, 4, 5, 6};
            std::vector<T> expected_values(input_values.size());
            std::transform(input_values.begin(), input_values.end(), expected_values.begin(), [](auto v) {
                return std::exp(static_cast<R>(v));
            });
            Tensor input(input_values, {2, 3}, device);
            auto data_ptr = input.data_ptr();
            input.exp_();
            Tensor expected(expected_values, {2, 3}, device);
            CHECK(allclose(input, expected));
            CHECK_EQ(input.data_ptr(), data_ptr);
        } else {
            Tensor input = full(true, {2, 3}, device);
            CHECK_THROWS_AS(input.exp_(), TTException);
        }
    };
    runner_all(test_exp);
}

// NOLINTNEXTLINE
TEST_CASE("Unary exp2 inplace") {
    auto test_exp2 = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        if constexpr (std::is_same_v<T, R>) {
            std::vector<T> input_values = {1, 2, 3, 4, 5, 6};
            std::vector<T> expected_values(input_values.size());
            std::transform(input_values.begin(), input_values.end(), expected_values.begin(), [](auto v) {
                return std::exp2(static_cast<R>(v));
            });
            Tensor input(input_values, {2, 3}, device);
            auto data_ptr = input.data_ptr();
            input.exp2_();
            Tensor expected(expected_values, {2, 3}, device);
            CHECK(allclose(input, expected));
            CHECK_EQ(input.data_ptr(), data_ptr);
        } else {
            Tensor input = full(true, {2, 3}, device);
            CHECK_THROWS_AS(input.exp2_(), TTException);
        }
    };
    runner_all(test_exp2);
}

// NOLINTNEXTLINE
TEST_CASE("Unary expm1 inplace") {
    auto test_expm1 = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        if constexpr (std::is_same_v<T, R>) {
            std::vector<T> input_values = {1, 2, 3, 4, 5, 6};
            std::vector<T> expected_values(input_values.size());
            std::transform(input_values.begin(), input_values.end(), expected_values.begin(), [](auto v) {
                return std::expm1(static_cast<R>(v));
            });
            Tensor input(input_values, {2, 3}, device);
            auto data_ptr = input.data_ptr();
            input.expm1_();
            Tensor expected(expected_values, {2, 3}, device);
            CHECK(allclose(input, expected));
            CHECK_EQ(input.data_ptr(), data_ptr);
        } else {
            Tensor input = full(true, {2, 3}, device);
            CHECK_THROWS_AS(input.expm1_(), TTException);
        }
    };
    runner_all(test_expm1);
}

// NOLINTNEXTLINE
TEST_CASE("Unary sqrt inplace") {
    auto test_sqrt = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        if constexpr (std::is_same_v<T, R>) {
            std::vector<T> input_values = {0, 1, 2, 3, 4, 5};
            std::vector<T> expected_values(input_values.size());
            std::transform(input_values.begin(), input_values.end(), expected_values.begin(), [](auto v) {
                return std::sqrt(static_cast<R>(v));
            });
            Tensor input(input_values, {2, 3}, device);
            auto data_ptr = input.data_ptr();
            input.sqrt_();
            Tensor expected(expected_values, {2, 3}, device);
            CHECK(allclose(input, expected));
            CHECK_EQ(input.data_ptr(), data_ptr);
        } else {
            Tensor input = full(true, {2, 3}, device);
            CHECK_THROWS_AS(input.sqrt_(), TTException);
        }
    };
    runner_all(test_sqrt);
}

// NOLINTNEXTLINE
TEST_CASE("Unary sin inplace") {
    auto test_sin = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        if constexpr (std::is_same_v<T, R>) {
            std::vector<T> input_values = {1, 2, 3, 4, 5, 6};
            std::vector<T> expected_values(input_values.size());
            std::transform(input_values.begin(), input_values.end(), expected_values.begin(), [](auto v) {
                return std::sin(static_cast<R>(v));
            });
            Tensor input(input_values, {2, 3}, device);
            auto data_ptr = input.data_ptr();
            input.sin_();
            Tensor expected(expected_values, {2, 3}, device);
            CHECK(allclose(input, expected));
            CHECK_EQ(input.data_ptr(), data_ptr);
        } else {
            Tensor input = full(true, {2, 3}, device);
            CHECK_THROWS_AS(input.sin_(), TTException);
        }
    };
    runner_all(test_sin);
}

// NOLINTNEXTLINE
TEST_CASE("Unary cos inplace") {
    auto test_cos = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        if constexpr (std::is_same_v<T, R>) {
            std::vector<T> input_values = {1, 2, 3, 4, 5, 6};
            std::vector<T> expected_values(input_values.size());
            std::transform(input_values.begin(), input_values.end(), expected_values.begin(), [](auto v) {
                return std::cos(static_cast<R>(v));
            });
            Tensor input(input_values, {2, 3}, device);
            auto data_ptr = input.data_ptr();
            input.cos_();
            Tensor expected(expected_values, {2, 3}, device);
            CHECK(allclose(input, expected));
            CHECK_EQ(input.data_ptr(), data_ptr);
        } else {
            Tensor input = full(true, {2, 3}, device);
            CHECK_THROWS_AS(input.cos_(), TTException);
        }
    };
    runner_all(test_cos);
}

// NOLINTNEXTLINE
TEST_CASE("Unary tan inplace") {
    auto test_tan = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        if constexpr (std::is_same_v<T, R>) {
            std::vector<T> input_values = {1, 2, 3, 4, 5, 6};
            std::vector<T> expected_values(input_values.size());
            std::transform(input_values.begin(), input_values.end(), expected_values.begin(), [](auto v) {
                return std::tan(static_cast<R>(v));
            });
            Tensor input(input_values, {2, 3}, device);
            auto data_ptr = input.data_ptr();
            input.tan_();
            Tensor expected(expected_values, {2, 3}, device);
            CHECK(allclose(input, expected));
            CHECK_EQ(input.data_ptr(), data_ptr);
        } else {
            Tensor input = full(true, {2, 3}, device);
            CHECK_THROWS_AS(input.tan_(), TTException);
        }
    };
    runner_all(test_tan);
}

// NOLINTNEXTLINE
TEST_CASE("Unary asin inplace") {
    auto test_asin = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        if constexpr (std::is_same_v<T, R>) {
            std::vector<T> input_values = {1, 2, 3, 4, 5, 6};
            std::vector<T> expected_values(input_values.size());
            std::transform(input_values.begin(), input_values.end(), expected_values.begin(), [](auto v) {
                return std::asin(static_cast<R>(v));
            });
            Tensor input(input_values, {2, 3}, device);
            auto data_ptr = input.data_ptr();
            input.asin_();
            Tensor expected(expected_values, {2, 3}, device);
            CHECK(allclose(input, expected, CloseOptions().equal_nan()));
            CHECK_EQ(input.data_ptr(), data_ptr);
        } else {
            Tensor input = full(true, {2, 3}, device);
            CHECK_THROWS_AS(input.asin_(), TTException);
        }
    };
    runner_all(test_asin);
}

// NOLINTNEXTLINE
TEST_CASE("Unary acos inplace") {
    auto test_acos = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        if constexpr (std::is_same_v<T, R>) {
            std::vector<T> input_values = {1, 2, 3, 4, 5, 6};
            std::vector<T> expected_values(input_values.size());
            std::transform(input_values.begin(), input_values.end(), expected_values.begin(), [](auto v) {
                return std::acos(static_cast<R>(v));
            });
            Tensor input(input_values, {2, 3}, device);
            auto data_ptr = input.data_ptr();
            input.acos_();
            Tensor expected(expected_values, {2, 3}, device);
            CHECK(allclose(input, expected, CloseOptions().equal_nan()));
            CHECK_EQ(input.data_ptr(), data_ptr);
        } else {
            Tensor input = full(true, {2, 3}, device);
            CHECK_THROWS_AS(input.acos_(), TTException);
        }
    };
    runner_all(test_acos);
}

// NOLINTNEXTLINE
TEST_CASE("Unary atan inplace") {
    auto test_atan = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        if constexpr (std::is_same_v<T, R>) {
            std::vector<T> input_values = {1, 2, 3, 4, 5, 6};
            std::vector<T> expected_values(input_values.size());
            std::transform(input_values.begin(), input_values.end(), expected_values.begin(), [](auto v) {
                return std::atan(static_cast<R>(v));
            });
            Tensor input(input_values, {2, 3}, device);
            auto data_ptr = input.data_ptr();
            input.atan_();
            Tensor expected(expected_values, {2, 3}, device);
            CHECK(allclose(input, expected));
            CHECK_EQ(input.data_ptr(), data_ptr);
        } else {
            Tensor input = full(true, {2, 3}, device);
            CHECK_THROWS_AS(input.atan_(), TTException);
        }
    };
    runner_all(test_atan);
}

// NOLINTNEXTLINE
TEST_CASE("Unary sinh inplace") {
    auto test_sinh = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        if constexpr (std::is_same_v<T, R>) {
            std::vector<T> input_values = {1, 2, 3, 4, 5, 6};
            std::vector<T> expected_values(input_values.size());
            std::transform(input_values.begin(), input_values.end(), expected_values.begin(), [](auto v) {
                return std::sinh(static_cast<R>(v));
            });
            Tensor input(input_values, {2, 3}, device);
            auto data_ptr = input.data_ptr();
            input.sinh_();
            Tensor expected(expected_values, {2, 3}, device);
            CHECK(allclose(input, expected));
            CHECK_EQ(input.data_ptr(), data_ptr);
        } else {
            Tensor input = full(true, {2, 3}, device);
            CHECK_THROWS_AS(input.sinh_(), TTException);
        }
    };
    runner_all(test_sinh);
}

// NOLINTNEXTLINE
TEST_CASE("Unary cosh inplace") {
    auto test_cosh = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        if constexpr (std::is_same_v<T, R>) {
            std::vector<T> input_values = {1, 2, 3, 4, 5, 6};
            std::vector<T> expected_values(input_values.size());
            std::transform(input_values.begin(), input_values.end(), expected_values.begin(), [](auto v) {
                return std::cosh(static_cast<R>(v));
            });
            Tensor input(input_values, {2, 3}, device);
            auto data_ptr = input.data_ptr();
            input.cosh_();
            Tensor expected(expected_values, {2, 3}, device);
            CHECK(allclose(input, expected));
            CHECK_EQ(input.data_ptr(), data_ptr);
        } else {
            Tensor input = full(true, {2, 3}, device);
            CHECK_THROWS_AS(input.cosh_(), TTException);
        }
    };
    runner_all(test_cosh);
}

// NOLINTNEXTLINE
TEST_CASE("Unary tanh inplace") {
    auto test_tanh = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        if constexpr (std::is_same_v<T, R>) {
            std::vector<T> input_values = {1, 2, 3, 4, 5, 6};
            std::vector<T> expected_values(input_values.size());
            std::transform(input_values.begin(), input_values.end(), expected_values.begin(), [](auto v) {
                return std::tanh(static_cast<R>(v));
            });
            Tensor input(input_values, {2, 3}, device);
            auto data_ptr = input.data_ptr();
            input.tanh_();
            Tensor expected(expected_values, {2, 3}, device);
            CHECK(allclose(input, expected));
            CHECK_EQ(input.data_ptr(), data_ptr);
        } else {
            Tensor input = full(true, {2, 3}, device);
            CHECK_THROWS_AS(input.tanh_(), TTException);
        }
    };
    runner_all(test_tanh);
}

// NOLINTNEXTLINE
TEST_CASE("Unary asinh inplace") {
    auto test_asinh = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        if constexpr (std::is_same_v<T, R>) {
            std::vector<T> input_values = {1, 2, 3, 4, 5, 6};
            std::vector<T> expected_values(input_values.size());
            std::transform(input_values.begin(), input_values.end(), expected_values.begin(), [](auto v) {
                return std::asinh(static_cast<R>(v));
            });
            Tensor input(input_values, {2, 3}, device);
            auto data_ptr = input.data_ptr();
            input.asinh_();
            Tensor expected(expected_values, {2, 3}, device);
            CHECK(allclose(input, expected));
            CHECK_EQ(input.data_ptr(), data_ptr);
        } else {
            Tensor input = full(true, {2, 3}, device);
            CHECK_THROWS_AS(input.asinh_(), TTException);
        }
    };
    runner_all(test_asinh);
}

// NOLINTNEXTLINE
TEST_CASE("Unary acosh inplace") {
    auto test_acosh = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        if constexpr (std::is_same_v<T, R>) {
            std::vector<T> input_values = {1, 2, 3, 4, 5, 6};
            std::vector<T> expected_values(input_values.size());
            std::transform(input_values.begin(), input_values.end(), expected_values.begin(), [](auto v) {
                return std::acosh(static_cast<R>(v));
            });
            Tensor input(input_values, {2, 3}, device);
            auto data_ptr = input.data_ptr();
            input.acosh_();
            Tensor expected(expected_values, {2, 3}, device);
            CHECK(allclose(input, expected));
            CHECK_EQ(input.data_ptr(), data_ptr);
        } else {
            Tensor input = full(true, {2, 3}, device);
            CHECK_THROWS_AS(input.acosh_(), TTException);
        }
    };
    runner_all(test_acosh);
}

// NOLINTNEXTLINE
TEST_CASE("Unary atanh inplace") {
    auto test_atanh = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        if constexpr (std::is_same_v<T, R>) {
            std::vector<T> input_values = {1, 2, 3, 4, 5, 6};
            std::vector<T> expected_values(input_values.size());
            std::transform(input_values.begin(), input_values.end(), expected_values.begin(), [](auto v) {
                return std::atanh(static_cast<R>(v));
            });
            Tensor input(input_values, {2, 3}, device);
            auto data_ptr = input.data_ptr();
            input.atanh_();
            Tensor expected(expected_values, {2, 3}, device);
            CHECK(allclose(input, expected, CloseOptions().equal_nan()));
            CHECK_EQ(input.data_ptr(), data_ptr);
        } else {
            Tensor input = full(true, {2, 3}, device);
            CHECK_THROWS_AS(input.atanh_(), TTException);
        }
    };
    runner_all(test_atanh);
}

// NOLINTNEXTLINE
TEST_CASE("Unary erf inplace") {
    auto test_erf = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        if constexpr (std::is_same_v<T, R>) {
            std::vector<T> input_values = {1, 2, 3, 4, 5, 6};
            std::vector<T> expected_values(input_values.size());
            std::transform(input_values.begin(), input_values.end(), expected_values.begin(), [](auto v) {
                return std::erf(static_cast<R>(v));
            });
            Tensor input(input_values, {2, 3}, device);
            auto data_ptr = input.data_ptr();
            input.erf_();
            Tensor expected(expected_values, {2, 3}, device);
            CHECK(allclose(input, expected));
            CHECK_EQ(input.data_ptr(), data_ptr);
        } else {
            Tensor input = full(true, {2, 3}, device);
            CHECK_THROWS_AS(input.erf_(), TTException);
        }
    };

    runner_all(test_erf);
}

// NOLINTNEXTLINE
TEST_CASE("Unary erfc inplace") {
    auto test_erfc = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        if constexpr (std::is_same_v<T, R>) {
            std::vector<T> input_values = {1, 2, 3, 4, 5, 6};
            std::vector<T> expected_values(input_values.size());
            std::transform(input_values.begin(), input_values.end(), expected_values.begin(), [](auto v) {
                return std::erfc(static_cast<R>(v));
            });
            Tensor input(input_values, {2, 3}, device);
            auto data_ptr = input.data_ptr();
            input.erfc_();
            Tensor expected(expected_values, {2, 3}, device);
            CHECK(allclose(input, expected));
            CHECK_EQ(input.data_ptr(), data_ptr);
        } else {
            Tensor input = full(true, {2, 3}, device);
            CHECK_THROWS_AS(input.erfc_(), TTException);
        }
    };
    runner_all(test_erfc);
}

// NOLINTNEXTLINE
TEST_CASE("Unary tgamma inplace") {
    auto test_tgamma = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        if constexpr (std::is_same_v<T, R>) {
            std::vector<T> input_values = {1, 2, 3, 4, 5, 6};
            std::vector<T> expected_values(input_values.size());
            std::transform(input_values.begin(), input_values.end(), expected_values.begin(), [](auto v) {
                return std::tgamma(static_cast<R>(v));
            });
            Tensor input(input_values, {2, 3}, device);
            auto data_ptr = input.data_ptr();
            input.tgamma_();
            Tensor expected(expected_values, {2, 3}, device);
            // CUDA tgamma isn't as accurate as std
            CHECK(allclose(input, expected));
            CHECK_EQ(input.data_ptr(), data_ptr);
        } else {
            Tensor input = full(true, {2, 3}, device);
            CHECK_THROWS_AS(input.tgamma_(), TTException);
        }
    };
    runner_all(test_tgamma);
}

// NOLINTNEXTLINE
TEST_CASE("Unary lgamma inplace") {
    auto test_lgamma = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        if constexpr (std::is_same_v<T, R>) {
            std::vector<T> input_values = {1, 2, 3, 4, 5, 6};
            std::vector<T> expected_values(input_values.size());
            std::transform(input_values.begin(), input_values.end(), expected_values.begin(), [](auto v) {
                return std::lgamma(static_cast<R>(v));
            });
            Tensor input(input_values, {2, 3}, device);
            auto data_ptr = input.data_ptr();
            input.lgamma_();
            Tensor expected(expected_values, {2, 3}, device);
            CHECK(allclose(input, expected));
            CHECK_EQ(input.data_ptr(), data_ptr);
        } else {
            Tensor input = full(true, {2, 3}, device);
            CHECK_THROWS_AS(input.lgamma_(), TTException);
        }
    };
    runner_all(test_lgamma);
}

// NOLINTNEXTLINE
TEST_CASE("Unary ceil inplace") {
    // ceil disabled for bool
    auto test_ceil_bool = []<typename T>(Device device) {
        Tensor input = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = input.ceil_(), TTException);
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
        auto data_ptr = input.data_ptr();
        CHECK_NOTHROW(input.ceil_());
        CHECK(allclose(input, expected));
        CHECK_EQ(input.data_ptr(), data_ptr);
    };
    runner_all_except_bool(test_ceil);
}

// NOLINTNEXTLINE
TEST_CASE("Unary floor inplace") {
    // floor disabled for bool
    auto test_floor_bool = []<typename T>(Device device) {
        Tensor input = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = input.floor_(), TTException);
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
        auto data_ptr = input.data_ptr();
        CHECK_NOTHROW(input.floor_());
        CHECK(allclose(input, expected));
        CHECK_EQ(input.data_ptr(), data_ptr);
    };
    runner_all_except_bool(test_floor);
}

// NOLINTNEXTLINE
TEST_CASE("Unary round inplace") {
    // round disabled for bool
    auto test_round_bool = []<typename T>(Device device) {
        Tensor input = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = input.round_(), TTException);
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
        auto data_ptr = input.data_ptr();
        CHECK_NOTHROW(input.round_());
        CHECK(allclose(input, expected));
        CHECK_EQ(input.data_ptr(), data_ptr);
    };
    runner_all_except_bool(test_round);
}

// NOLINTNEXTLINE
TEST_CASE("Unary sigmoid inplace") {
    // sigmoid disabled for bool
    auto test_sigmoid_bool = []<typename T>(Device device) {
        Tensor input = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = input.sigmoid_(), TTException);
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
        if constexpr (std::is_same_v<T, R>) {
            auto data_ptr = input.data_ptr();
            input.sigmoid_();
            Tensor expected(expected_values, {2, 3}, device);
            CHECK(allclose(input, expected));
            CHECK_EQ(input.data_ptr(), data_ptr);
        } else {
            CHECK_THROWS_AS(input.sigmoid_(), TTException);
        }
    };
    runner_all_except_bool(test_sigmoid);
}

// NOLINTNEXTLINE
TEST_CASE("Unary logsigmoid") {
    // log_sigmoid disabled for bool
    auto test_logsigmoid_bool = []<typename T>(Device device) {
        Tensor input = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = input.log_sigmoid_(), TTException);
    };
    runner_boolean(test_logsigmoid_bool);
}

// NOLINTNEXTLINE
TEST_CASE("Unary hardsigmoid") {
    // hardsigmoid disabled for bool
    auto test_hardsigmoid_bool = []<typename T>(Device device) {
        Tensor input = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = input.hardsigmoid_(), TTException);
    };
    runner_boolean(test_hardsigmoid_bool);
}

// NOLINTNEXTLINE
TEST_CASE("Unary relu") {
    // hardsigmoid disabled for bool
    auto test_relu_bool = []<typename T>(Device device) {
        Tensor input = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = input.relu_(), TTException);
    };
    runner_boolean(test_relu_bool);
}

// NOLINTNEXTLINE
TEST_CASE("Unary relu6") {
    // relu6 disabled for bool
    auto test_relu6_bool = []<typename T>(Device device) {
        Tensor input = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = input.relu_(), TTException);
    };
    runner_boolean(test_relu6_bool);
}

// NOLINTNEXTLINE
TEST_CASE("Unary leaky_relu") {
    // leaky_relu disabled for bool
    auto test_leaky_relu_bool = []<typename T>(Device device) {
        Tensor input = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = input.leaky_relu_(), TTException);
    };
    runner_boolean(test_leaky_relu_bool);
}

// NOLINTNEXTLINE
TEST_CASE("Unary elu") {
    // elu disabled for bool
    auto test_elu_bool = []<typename T>(Device device) {
        Tensor input = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = input.elu_(), TTException);
    };
    runner_boolean(test_elu_bool);
}

// NOLINTNEXTLINE
TEST_CASE("Unary selu") {
    // selu disabled for bool
    auto test_selu_bool = []<typename T>(Device device) {
        Tensor input = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = input.selu_(), TTException);
    };
    runner_boolean(test_selu_bool);
}

// NOLINTNEXTLINE
TEST_CASE("Unary silu") {
    // silu disabled for bool
    auto test_silu_bool = []<typename T>(Device device) {
        Tensor input = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = input.silu_(), TTException);
    };
    runner_boolean(test_silu_bool);
}

// NOLINTNEXTLINE
TEST_CASE("Unary hardtanh") {
    // hardtanh disabled for bool
    auto test_hardtanh_bool = []<typename T>(Device device) {
        Tensor input = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = input.hardtanh_(), TTException);
    };
    runner_boolean(test_hardtanh_bool);
}

// NOLINTNEXTLINE
TEST_CASE("Unary softsign") {
    // softsign disabled for bool
    auto test_softsign_bool = []<typename T>(Device device) {
        Tensor input = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = input.softsign_(), TTException);
    };
    runner_boolean(test_softsign_bool);
}

// NOLINTNEXTLINE
TEST_CASE("Unary softplus") {
    // softplus disabled for bool
    auto test_softplus_bool = []<typename T>(Device device) {
        Tensor input = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = input.softplus_(), TTException);
    };
    runner_boolean(test_softplus_bool);
}

// NOLINTNEXTLINE
TEST_CASE("Unary softmax") {
    // softmax disabled for bool
    auto test_softmax_bool = []<typename T>(Device device) {
        Tensor input = full(true, {2, 3}, device);
        CHECK_THROWS_AS(std::ignore = input.softmax_(0), TTException);
    };
    runner_boolean(test_softmax_bool);
}

#ifdef TT_TORCH
// NOLINTNEXTLINE
TEST_CASE("Unary sigmoid inplace Torch") {
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
        auto data_ptr = X.data_ptr();
        X.sigmoid_();
        CHECK(allclose(X, expected, CloseOptions().atol(1e-6)));
        CHECK_EQ(X.data_ptr(), data_ptr);
    };
    runner_single_type<float>(test1);
}

// NOLINTNEXTLINE
TEST_CASE("Unary logsigmoid inplace Torch") {
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
        auto data_ptr = X.data_ptr();
        X.log_sigmoid_();
        CHECK(allclose(X, expected, CloseOptions().atol(1e-6)));
        CHECK_EQ(X.data_ptr(), data_ptr);
    };
    runner_single_type<float>(test1);
}

// NOLINTNEXTLINE
TEST_CASE("Unary hardsigmoid inplace Torch") {
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
        auto data_ptr = X.data_ptr();
        X.hardsigmoid_();
        CHECK(allclose(X, expected, CloseOptions().atol(1e-6)));
        CHECK_EQ(X.data_ptr(), data_ptr);
    };
    runner_single_type<float>(test1);
}

// NOLINTNEXTLINE
TEST_CASE("Unary softplus inplace Torch") {
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
        auto data_ptr = X.data_ptr();
        X.softplus_();
        CHECK(allclose(X, expected, CloseOptions().atol(1e-6)));
        CHECK_EQ(X.data_ptr(), data_ptr);
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
        auto data_ptr = X.data_ptr();
        X.softplus_(0.5, 3);
        CHECK(allclose(X, expected, CloseOptions().atol(1e-6)));
        CHECK_EQ(X.data_ptr(), data_ptr);
    };
    runner_single_type<float>(test2);
}

// NOLINTNEXTLINE
TEST_CASE("Unary relu inplace Torch") {
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
        auto data_ptr = X.data_ptr();
        X.relu_();
        CHECK(allclose(X, expected, CloseOptions().atol(1e-6)));
        CHECK_EQ(X.data_ptr(), data_ptr);
    };
    runner_single_type<float>(test1);
}

// NOLINTNEXTLINE
TEST_CASE("Unary relu6 inplace Torch") {
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
        auto data_ptr = X.data_ptr();
        X.relu6_();
        CHECK(allclose(X, expected, CloseOptions().atol(1e-6)));
        CHECK_EQ(X.data_ptr(), data_ptr);
    };
    runner_single_type<float>(test1);
}

// NOLINTNEXTLINE
TEST_CASE("Unary leaky_relu inplace Torch") {
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
        auto data_ptr = X.data_ptr();
        X.leaky_relu_();
        CHECK(allclose(X, expected, CloseOptions().atol(1e-6)));
        CHECK_EQ(X.data_ptr(), data_ptr);
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
        auto data_ptr = X.data_ptr();
        X.leaky_relu_(0.1);
        CHECK(allclose(X, expected, CloseOptions().atol(1e-6)));
        CHECK_EQ(X.data_ptr(), data_ptr);
    };
    runner_single_type<float>(test2);
}

// NOLINTNEXTLINE
TEST_CASE("Unary elu inplace Torch") {
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
        auto data_ptr = X.data_ptr();
        X.elu_();
        CHECK(allclose(X, expected, CloseOptions().atol(1e-6)));
        CHECK_EQ(X.data_ptr(), data_ptr);
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
        auto data_ptr = X.data_ptr();
        X.elu_(2);
        CHECK(allclose(X, expected, CloseOptions().atol(1e-6)));
        CHECK_EQ(X.data_ptr(), data_ptr);
    };
    runner_single_type<float>(test2);
}

// NOLINTNEXTLINE
TEST_CASE("Unary selu inplace Torch") {
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
        auto data_ptr = X.data_ptr();
        X.selu_();
        CHECK(allclose(X, expected, CloseOptions().atol(1e-6)));
        CHECK_EQ(X.data_ptr(), data_ptr);
    };
    runner_single_type<float>(test1);
}

// NOLINTNEXTLINE
TEST_CASE("Unary silu inplace Torch") {
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
        auto data_ptr = X.data_ptr();
        X.silu_();
        CHECK(allclose(X, expected, CloseOptions().atol(1e-6)));
        CHECK_EQ(X.data_ptr(), data_ptr);
    };
    runner_single_type<float>(test1);
}

// NOLINTNEXTLINE
TEST_CASE("Unary hardtanh inplace Torch") {
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
        auto data_ptr = X.data_ptr();
        X.hardtanh_();
        CHECK(allclose(X, expected, CloseOptions().atol(1e-6)));
        CHECK_EQ(X.data_ptr(), data_ptr);
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
        auto data_ptr = X.data_ptr();
        X.hardtanh_(-0.5, 0.5);
        CHECK(allclose(X, expected, CloseOptions().atol(1e-6)));
        CHECK_EQ(X.data_ptr(), data_ptr);
    };
    runner_single_type<float>(test2);
}

// NOLINTNEXTLINE
TEST_CASE("Unary softsign inplace Torch") {
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
        auto data_ptr = X.data_ptr();
        X.softsign_();
        CHECK(allclose(X, expected, CloseOptions().atol(1e-6)));
        CHECK_EQ(X.data_ptr(), data_ptr);
    };
    runner_single_type<float>(test1);
}

// NOLINTNEXTLINE
TEST_CASE("Unary softmax inplace Torch") {
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
        auto data_ptr = X.data_ptr();
        X.softmax_(0);
        CHECK(allclose(X, expected, CloseOptions().atol(1e-6)));
        CHECK_EQ(X.data_ptr(), data_ptr);
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
        auto data_ptr = X.data_ptr();
        X.softmax_(1);
        CHECK(allclose(X, expected, CloseOptions().atol(1e-6)));
        CHECK_EQ(X.data_ptr(), data_ptr);
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
        auto data_ptr = X.data_ptr();
        X.softmax_(2);
        CHECK(allclose(X, expected, CloseOptions().atol(1e-6)));
        CHECK_EQ(X.data_ptr(), data_ptr);
    };
    runner_single_type<float>(test3);
}

// NOLINTNEXTLINE
TEST_CASE("Unary log_softmax inplace Torch") {
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
        auto data_ptr = X.data_ptr();
        X.log_softmax_(0);
        CHECK(allclose(X, expected, CloseOptions().atol(1e-6)));
        CHECK_EQ(X.data_ptr(), data_ptr);
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
        auto data_ptr = X.data_ptr();
        X.log_softmax_(1);
        CHECK(allclose(X, expected, CloseOptions().atol(1e-6)));
        CHECK_EQ(X.data_ptr(), data_ptr);
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
        auto data_ptr = X.data_ptr();
        X.log_softmax_(2);
        CHECK(allclose(X, expected, CloseOptions().atol(1e-6)));
        CHECK_EQ(X.data_ptr(), data_ptr);
    };
    runner_single_type<float>(test3);
}
#endif
