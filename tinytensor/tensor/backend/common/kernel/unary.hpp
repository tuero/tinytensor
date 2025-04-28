// unary.hpp
// Common element-wise unary kernels

#ifndef TINYTENSOR_BACKEND_COMMON_KERNEL_UNARY_H_
#define TINYTENSOR_BACKEND_COMMON_KERNEL_UNARY_H_

#include <tt/macros.h>
#include <tt/scalar.h>

#include <cmath>
#include <cstdint>
#include <numbers>
#include <type_traits>

#if defined(__CUDACC__)
#include <nvfunctional>
#else
#include <functional>
#endif

namespace tinytensor::common::kernel::unary {

// Basic operations
template <typename T>
struct OpIdentity {
    TT_STD_FUNC<T(T)> TT_HOST_DEVICE operator()() const {
        return [](auto val) -> T { return val; };
    }
};

template <typename T>
struct OpNegate {
    TT_STD_FUNC<T(T)> TT_HOST_DEVICE operator()() const {
        return [](auto val) -> T {
            if constexpr (std::is_same_v<T, uint8_t>) {
                return val;
            } else {
                return -val;
            }
        };
    }
};

template <typename T>
struct OpLogicalNot {
    TT_STD_FUNC<bool(T)> TT_HOST_DEVICE operator()() const {
        return [](auto val) -> bool { return !static_cast<bool>(val); };
    }
};

template <typename T>
struct OpAbs {
    TT_STD_FUNC<T(T)> TT_HOST_DEVICE operator()() const {
        return [](auto val) -> T {
            if constexpr (std::is_same_v<T, uint8_t>) {
                return val;
            } else {
                return static_cast<T>(std::abs(val));
            }
        };
    }
};

template <typename T>
struct OpSign {
    TT_STD_FUNC<T(T)> TT_HOST_DEVICE operator()() const {
        return [](auto val) -> T {
            if constexpr (std::is_same_v<T, uint8_t>) {
                return val > T{0} ? T{1} : T{0};
            } else {
                return static_cast<T>((T{0} < val) - (val < T{0}));
            }
        };
    }
};

// Exponential functions
template <typename T>
struct OpLog {
    TT_STD_FUNC<T(T)> TT_HOST_DEVICE operator()() const {
        return [](auto val) -> T { return std::log(val); };
    }
};

template <typename T>
struct OpLog10 {
    TT_STD_FUNC<T(T)> TT_HOST_DEVICE operator()() const {
        return [](auto val) -> T { return std::log10(val); };
    }
};

template <typename T>
struct OpLog2 {
    TT_STD_FUNC<T(T)> TT_HOST_DEVICE operator()() const {
        return [](auto val) -> T { return std::log2(val); };
    }
};

template <typename T>
struct OpLog1p {
    TT_STD_FUNC<T(T)> TT_HOST_DEVICE operator()() const {
        return [](auto val) -> T { return std::log1p(val); };
    }
};

template <typename T>
struct OpExp {
    TT_STD_FUNC<T(T)> TT_HOST_DEVICE operator()() const {
        return [](auto val) -> T { return std::exp(val); };
    }
};

template <typename T>
struct OpExp2 {
    TT_STD_FUNC<T(T)> TT_HOST_DEVICE operator()() const {
        return [](auto val) -> T { return std::exp2(val); };
    }
};

template <typename T>
struct OpExpm1 {
    TT_STD_FUNC<T(T)> TT_HOST_DEVICE operator()() const {
        return [](auto val) -> T { return std::expm1(val); };
    }
};

// Power functions
template <typename T>
struct OpSqrt {
    TT_STD_FUNC<T(T)> TT_HOST_DEVICE operator()() const {
        return [](auto val) -> T { return std::sqrt(val); };
    }
};

// Trigonometric functions
template <typename T>
struct OpSin {
    TT_STD_FUNC<T(T)> TT_HOST_DEVICE operator()() const {
        return [](auto val) -> T { return std::sin(val); };
    }
};

template <typename T>
struct OpCos {
    TT_STD_FUNC<T(T)> TT_HOST_DEVICE operator()() const {
        return [](auto val) -> T { return std::cos(val); };
    }
};

template <typename T>
struct OpTan {
    TT_STD_FUNC<T(T)> TT_HOST_DEVICE operator()() const {
        return [](auto val) -> T { return std::tan(val); };
    }
};

template <typename T>
struct OpASin {
    TT_STD_FUNC<T(T)> TT_HOST_DEVICE operator()() const {
        return [](auto val) -> T { return std::asin(val); };
    }
};

template <typename T>
struct OpACos {
    TT_STD_FUNC<T(T)> TT_HOST_DEVICE operator()() const {
        return [](auto val) -> T { return std::acos(val); };
    }
};

template <typename T>
struct OpATan {
    TT_STD_FUNC<T(T)> TT_HOST_DEVICE operator()() const {
        return [](auto val) -> T { return std::atan(val); };
    }
};

// Hyperbolic functions
template <typename T>
struct OpSinh {
    TT_STD_FUNC<T(T)> TT_HOST_DEVICE operator()() const {
        return [](auto val) -> T { return std::sinh(val); };
    }
};

template <typename T>
struct OpCosh {
    TT_STD_FUNC<T(T)> TT_HOST_DEVICE operator()() const {
        return [](auto val) -> T { return std::cosh(val); };
    }
};

template <typename T>
struct OpTanh {
    TT_STD_FUNC<T(T)> TT_HOST_DEVICE operator()() const {
        return [](auto val) -> T { return std::tanh(val); };
    }
};

template <typename T>
struct OpASinh {
    TT_STD_FUNC<T(T)> TT_HOST_DEVICE operator()() const {
        return [](auto val) -> T { return std::asinh(val); };
    }
};

template <typename T>
struct OpACosh {
    TT_STD_FUNC<T(T)> TT_HOST_DEVICE operator()() const {
        return [](auto val) -> T { return std::acosh(val); };
    }
};

template <typename T>
struct OpATanh {
    TT_STD_FUNC<T(T)> TT_HOST_DEVICE operator()() const {
        return [](auto val) -> T { return std::atanh(val); };
    }
};

// Error and Gamma functions
template <typename T>
struct OpErf {
    TT_STD_FUNC<T(T)> TT_HOST_DEVICE operator()() const {
        return [](auto val) -> T { return std::erf(val); };
    }
};

template <typename T>
struct OpErfc {
    TT_STD_FUNC<T(T)> TT_HOST_DEVICE operator()() const {
        return [](auto val) -> T { return std::erfc(val); };
    }
};

template <typename T>
struct OpTGamma {
    TT_STD_FUNC<T(T)> TT_HOST_DEVICE operator()() const {
        return [](auto val) -> T { return std::tgamma(val); };
    }
};

template <typename T>
struct OpLGamma {
    TT_STD_FUNC<T(T)> TT_HOST_DEVICE operator()() const {
        return [](auto val) -> T { return std::lgamma(val); };
    }
};

// https://math.stackexchange.com/questions/1441753/approximating-the-digamma-function/1441768#1441768
// @NOTE: We don't check for the poles here and return 0/NAN, we just let the computation carry on
template <typename T>
struct OpDiGamma {
    TT_STD_FUNC<T(T)> TT_HOST_DEVICE operator()() const {
        return [](auto val) -> T {
            constexpr T HALF = 0.5;
            constexpr T ONE = 1;
            constexpr T LIMIT = 10;
            constexpr T C1 = -static_cast<T>(1) / static_cast<T>(2);
            constexpr T C2 = -static_cast<T>(1) / static_cast<T>(12);
            constexpr T C3 = static_cast<T>(1) / static_cast<T>(120);
            constexpr T C4 = -static_cast<T>(1) / static_cast<T>(252);
            constexpr T C5 = static_cast<T>(1) / static_cast<T>(240);
            constexpr T C6 = -static_cast<T>(5) / static_cast<T>(660);
            constexpr T C7 = static_cast<T>(691) / static_cast<T>(32760);
            T rem = (val < HALF) ? (std::numbers::pi_v<T> * ONE / std::tan(std::numbers::pi_v<T> * (ONE - val))) : 0;
            val = (val < HALF) ? ONE - val : val;
            while (val < LIMIT) {
                rem -= ONE / val;
                ++val;
            }
            rem += std::log(val);
            rem += static_cast<T>(C1 / val);
            rem += static_cast<T>(C2 / std::pow(val, 2));
            rem += static_cast<T>(C3 / std::pow(val, 4));
            rem += static_cast<T>(C4 / std::pow(val, 6));
            rem += static_cast<T>(C5 / std::pow(val, 8));
            rem += static_cast<T>(C6 / std::pow(val, 10));
            rem += static_cast<T>(C7 / std::pow(val, 12));
            return rem;
        };
    }
};

// Nearest Integer functions
template <typename T>
struct OpCeil {
    TT_STD_FUNC<T(T)> TT_HOST_DEVICE operator()() const {
        return [](auto val) -> T {
            if constexpr (IsScalarIntType<T>) {
                return val;
            } else {
                return std::ceil(val);
            }
        };
    }
};

template <typename T>
struct OpFloor {
    TT_STD_FUNC<T(T)> TT_HOST_DEVICE operator()() const {
        return [](auto val) -> T {
            if constexpr (IsScalarIntType<T>) {
                return val;
            } else {
                return std::floor(val);
            }
        };
    }
};

template <typename T>
struct OpRound {
    TT_STD_FUNC<T(T)> TT_HOST_DEVICE operator()() const {
        return [](auto val) -> T {
            if constexpr (IsScalarIntType<T>) {
                return val;
            } else {
                return std::round(val);
            }
        };
    }
};

// Classification functions
template <typename T>
struct OpIsInf {
    TT_STD_FUNC<bool(T)> TT_HOST_DEVICE operator()() const {
        return [](auto val) -> bool {
            if constexpr (IsScalarIntType<T>) {
                return false;
            } else {
                return std::isinf(val);
            }
        };
    }
};

template <typename T>
struct OpIsNaN {
    TT_STD_FUNC<bool(T)> TT_HOST_DEVICE operator()() const {
        return [](auto val) -> bool {
            if constexpr (IsScalarIntType<T>) {
                return false;
            } else {
                return std::isnan(val);
            }
        };
    }
};

template <typename T>
struct OpIsFinite {
    TT_STD_FUNC<bool(T)> TT_HOST_DEVICE operator()() const {
        return [](auto val) -> bool {
            if constexpr (IsScalarIntType<T>) {
                return true;
            } else {
                return std::isfinite(val);
            }
        };
    }
};

// Util/misc
template <typename T>
constexpr auto _positive_sigmoid(T x) -> T {
    return 1 / (1 + std::exp(-x));
}

template <typename T>
constexpr auto _negative_sigmoid(T x) -> T {
    const auto exp = std::exp(x);
    return exp / (exp + 1);
}

// Numerically stable sigmoid
// 1 / (1+e^-x) == e^x / (e^x + 1)
template <typename T>
constexpr auto sigmoid(T x) -> T {
    return (x > 0) ? _positive_sigmoid(x) : _negative_sigmoid(x);
}

template <typename T>
struct OpSigmoid {
    TT_STD_FUNC<T(T)> TT_HOST_DEVICE operator()() const {
        return [](auto val) -> T { return sigmoid(val); };
    }
};

// PyTorch uses 1 as beta and 20 as threshold for defaults
template <typename T>
struct OpSoftplus {
    TT_STD_FUNC<T(T, T, T)> TT_HOST_DEVICE operator()() const {
        return [](auto x, auto beta, auto threshold) -> T {
            return (beta * x > threshold) ? x : std::log(1 + std::exp(beta * x)) / beta;
        };
    }
};

template <typename T>
struct OpReLU {
    TT_STD_FUNC<T(T)> TT_HOST_DEVICE operator()() const {
        return [](auto x) -> T { return std::max(x, static_cast<T>(0)); };
    }
};

template <typename T>
struct OpReLU6 {
    TT_STD_FUNC<T(T)> TT_HOST_DEVICE operator()() const {
        return [](auto x) -> T { return std::min(std::max(x, static_cast<T>(0)), static_cast<T>(6)); };
    }
};

// Exponential Linear Unit
// PyTorch uses 1 as a default alpha
template <typename T>
struct OpELU {
    TT_STD_FUNC<T(T, T)> TT_HOST_DEVICE operator()() const {
        return [](auto x, auto alpha) -> T { return (x > 0) ? x : alpha * (std::exp(x) - 1); };
    }
};

template <typename T>
struct OpSELU {
    TT_STD_FUNC<T(T)> TT_HOST_DEVICE operator()() const {
        return [](auto x) -> T {
            constexpr T alpha = static_cast<T>(1.67326324235437728);
            constexpr T scale = static_cast<T>(1.05070098735548049);
            return scale * (std::max(static_cast<T>(0), x) + std::min(static_cast<T>(0), alpha * (std::exp(x) - 1)));
        };
    }
};

// Sigmoid Linear Unit
template <typename T>
struct OpSiLU {
    TT_STD_FUNC<T(T)> TT_HOST_DEVICE operator()() const {
        return [](auto x) -> T { return x * sigmoid(x); };
    }
};

template <typename T>
struct OpHardSigmoid {
    TT_STD_FUNC<T(T)> TT_HOST_DEVICE operator()() const {
        return [](auto x) -> T {
            if (x <= -3) {
                return 0;
            } else if (x >= 3) {
                return 1;
            } else {
                return static_cast<T>(x / 6) + static_cast<T>(0.5);
            }
        };
    }
};

// PyTorch uses -1 as min and 1 as max for defaults
template <typename T>
struct OpHardTanh {
    TT_STD_FUNC<T(T, T, T)> TT_HOST_DEVICE operator()() const {
        return [](auto x, auto min, auto max) -> T {
            if (x < min) {
                return min;
            } else if (x > max) {
                return max;
            } else {
                return x;
            }
        };
    }
};

// PyTorch uses 0.01 as a default negative_slope
template <typename T>
struct OpLeakyReLU {
    TT_STD_FUNC<T(T, T)> TT_HOST_DEVICE operator()() const {
        return [](auto x, auto negative_slope) -> T {
            return std::max(x, static_cast<T>(0)) + negative_slope * std::min(x, static_cast<T>(0));
        };
    }
};

template <typename T>
struct OpLogSigmoid {
    TT_STD_FUNC<T(T)> TT_HOST_DEVICE operator()() const {
        return [](auto x) -> T { return std::log(sigmoid(x)); };
    }
};

// Sigmoid Linear Unit
template <typename T>
struct OpSoftsign {
    TT_STD_FUNC<T(T)> TT_HOST_DEVICE operator()() const {
        return [](auto x) -> T { return x / (1 + std::abs(x)); };
    }
};

}    // namespace tinytensor::common::kernel::unary

#endif    // TINYTENSOR_BACKEND_COMMON_KERNEL_UNARY_H_
