// binary.hpp
// Common element-wise binary kernels

#ifndef TINYTENSOR_BACKEND_COMMON_KERNEL_BINARY_H_
#define TINYTENSOR_BACKEND_COMMON_KERNEL_BINARY_H_

#include <tt/macros.h>
#include <tt/scalar.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

#if defined(__CUDACC__)
#include <nvfunctional>
#else
#include <functional>
#endif

namespace tinytensor::common::kernel::binary {

template <typename T>
struct OpAdd {
    TT_STD_FUNC<T(T, T)> TT_HOST_DEVICE operator()() const {
        return [](auto lhs, auto rhs) { return lhs + rhs; };
    }
};

template <typename T>
struct OpSub {
    TT_STD_FUNC<T(T, T)> TT_HOST_DEVICE operator()() const {
        return [](auto lhs, auto rhs) { return lhs - rhs; };
    }
};

template <typename T>
struct OpMul {
    TT_STD_FUNC<T(T, T)> TT_HOST_DEVICE operator()() const {
        return [](auto lhs, auto rhs) { return lhs * rhs; };
    }
};

template <typename T>
struct OpDiv {
    TT_STD_FUNC<T(T, T)> TT_HOST_DEVICE operator()() const {
        return [](auto lhs, auto rhs) { return lhs / rhs; };
    }
};

template <typename T>
struct OpMinimum {
    static constexpr T padding_value = std::numeric_limits<T>::max();
    TT_STD_FUNC<T(T, T)> TT_HOST_DEVICE operator()() const {
        return [](auto lhs, auto rhs) { return (lhs < rhs) ? lhs : rhs; };
    }
};

template <typename T>
struct OpMaximum {
    static constexpr T padding_value = std::numeric_limits<T>::lowest();
    TT_STD_FUNC<T(T, T)> TT_HOST_DEVICE operator()() const {
        return [](auto lhs, auto rhs) { return (lhs > rhs) ? lhs : rhs; };
    }
};

template <typename T>
struct OpPow {
    TT_STD_FUNC<T(T, T)> TT_HOST_DEVICE operator()() const {
        return [](auto lhs, auto rhs) { return std::pow(lhs, rhs); };
    }
};

template <typename T>
struct OpEq {
    TT_STD_FUNC<bool(T, T)> TT_HOST_DEVICE operator()() const {
        return [](auto lhs, auto rhs) { return lhs == rhs; };
    }
};

// Use Kahan summation for added precision
// Low bits are lost on addition, so we track compensation in c
// https://en.wikipedia.org/wiki/Kahan_summation_algorithm
template <typename T>
struct OpKahanSum {
    static constexpr T padding_value = 0;
    TT_STD_FUNC<T(T, T)> TT_HOST_DEVICE operator()() {
        return [this](auto lhs, auto rhs) {
            T rhs_with_compensation = rhs - c;
            T sum = static_cast<T>(lhs + rhs_with_compensation);    // Promotion can occur for uint8_t
            c = static_cast<T>((sum - lhs) - rhs_with_compensation);
            return sum;
        };
    }
    T c = padding_value;
};

template <typename T>
struct OpNe {
    TT_STD_FUNC<bool(T, T)> TT_HOST_DEVICE operator()() const {
        return [](auto lhs, auto rhs) { return lhs != rhs; };
    }
};

template <typename T>
struct OpLt {
    TT_STD_FUNC<bool(T, T)> TT_HOST_DEVICE operator()() const {
        return [](auto lhs, auto rhs) { return lhs < rhs; };
    }
};

template <typename T>
struct OpLe {
    TT_STD_FUNC<bool(T, T)> TT_HOST_DEVICE operator()() const {
        return [](auto lhs, auto rhs) { return lhs <= rhs; };
    }
};

template <typename T>
struct OpGt {
    TT_STD_FUNC<bool(T, T)> TT_HOST_DEVICE operator()() const {
        return [](auto lhs, auto rhs) { return lhs > rhs; };
    }
};

template <typename T>
struct OpGe {
    TT_STD_FUNC<bool(T, T)> TT_HOST_DEVICE operator()() const {
        return [](auto lhs, auto rhs) { return lhs >= rhs; };
    }
};

template <typename T>
struct OpLogicalOr {
    TT_STD_FUNC<bool(bool, bool)> TT_HOST_DEVICE operator()() const {
        return [](auto lhs, auto rhs) { return lhs || rhs; };
    }
};

template <typename T>
struct OpLogicalAnd {
    TT_STD_FUNC<bool(bool, bool)> TT_HOST_DEVICE operator()() const {
        return [](auto lhs, auto rhs) { return lhs && rhs; };
    }
};

// Cast to ensure it compiles for all types
// we handle invalid types at the top level, not backend level
template <typename T>
struct OpBitwiseOr {
    TT_STD_FUNC<T(T, T)> TT_HOST_DEVICE operator()() const {
        using U = std::conditional_t<IsScalarIntType<T>, T, uint64_t>;
        return [](auto lhs, auto rhs) { return static_cast<U>(lhs) | static_cast<U>(rhs); };
    }
};

template <typename T>
struct OpBitwiseAnd {
    TT_STD_FUNC<T(T, T)> TT_HOST_DEVICE operator()() const {
        using U = std::conditional_t<IsScalarIntType<T>, T, uint64_t>;
        return [](auto lhs, auto rhs) { return static_cast<U>(lhs) & static_cast<U>(rhs); };
    }
};

template <typename T>
struct OpBitwiseXor {
    TT_STD_FUNC<T(T, T)> TT_HOST_DEVICE operator()() const {
        using U = std::conditional_t<IsScalarIntType<T>, T, uint64_t>;
        return [](auto lhs, auto rhs) { return static_cast<U>(lhs) ^ static_cast<U>(rhs); };
    }
};

template <typename T>
struct OpBitwiseLeftShift {
    TT_STD_FUNC<T(T, T)> TT_HOST_DEVICE operator()() const {
        using U = std::conditional_t<IsScalarIntType<T>, T, uint64_t>;
        return [](auto lhs, auto rhs) { return static_cast<U>(lhs) << static_cast<U>(rhs); };
    }
};

template <typename T>
struct OpBitwiseRightShift {
    TT_STD_FUNC<T(T, T)> TT_HOST_DEVICE operator()() const {
        using U = std::conditional_t<IsScalarIntType<T>, T, uint64_t>;
        return [](auto lhs, auto rhs) { return static_cast<U>(lhs) >> static_cast<U>(rhs); };
    }
};

template <typename T>
struct OpModulo {
    TT_STD_FUNC<T(T, T)> TT_HOST_DEVICE operator()() const {
        return [](auto lhs, auto rhs) {
            if constexpr (IsScalarIntType<T>) {
                return lhs % rhs;
            } else {
                return std::fmod(lhs, rhs);
            }
        };
    }
};

}    // namespace tinytensor::common::kernel::binary

#endif    // TINYTENSOR_BACKEND_COMMON_KERNEL_BINARY_H_
