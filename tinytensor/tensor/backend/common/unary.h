// unary.h
// Common unary utils

#ifndef TINYTENSOR_BACKEND_COMMON_UNARY_H_
#define TINYTENSOR_BACKEND_COMMON_UNARY_H_

#include <tt/scalar.h>

#include "tensor/backend/common/kernel/unary.hpp"

#include <type_traits>

namespace tinytensor::common::unary {

using namespace kernel::unary;

// Unary operators
enum class UnaryOpT {
    identity,
    negate,
    logical_not,
    abs,
    sign,
    log,
    log10,
    log2,
    log1p,
    exp,
    exp2,
    expm1,
    sqrt,
    sin,
    cos,
    tan,
    asin,
    acos,
    atan,
    sinh,
    cosh,
    tanh,
    asinh,
    acosh,
    atanh,
    erf,
    erfc,
    tgamma,
    lgamma,
    digamma,
    ceil,
    floor,
    round,
    isinf,
    isnan,
    isfinite,
    sigmoid,
    softplus,
    relu,
    relu6,
    leaky_relu,
    elu,
    selu,
    silu,
    hardsigmoid,
    hardtanh,
    log_sigmoid,
    softsign
};

// Unary op to kernel mapping
template <typename T, UnaryOpT Op>
struct OpFactory;

// NOLINTNEXTLINE(*-macro-usage)
#define DECLARE_OP_FACTORY(UNARY_OPT, KERN_OPT) \
    template <typename T>                       \
    struct OpFactory<T, UNARY_OPT> {            \
        using KernelOp = KERN_OPT<T>;           \
    };

DECLARE_OP_FACTORY(UnaryOpT::identity, OpIdentity);
DECLARE_OP_FACTORY(UnaryOpT::negate, OpNegate);
DECLARE_OP_FACTORY(UnaryOpT::logical_not, OpLogicalNot);
DECLARE_OP_FACTORY(UnaryOpT::abs, OpAbs);
DECLARE_OP_FACTORY(UnaryOpT::sign, OpSign);
// Exponential functions
DECLARE_OP_FACTORY(UnaryOpT::log, OpLog);
DECLARE_OP_FACTORY(UnaryOpT::log10, OpLog10);
DECLARE_OP_FACTORY(UnaryOpT::log2, OpLog2);
DECLARE_OP_FACTORY(UnaryOpT::log1p, OpLog1p);
DECLARE_OP_FACTORY(UnaryOpT::exp, OpExp);
DECLARE_OP_FACTORY(UnaryOpT::exp2, OpExp2);
DECLARE_OP_FACTORY(UnaryOpT::expm1, OpExpm1);
// Power functions
DECLARE_OP_FACTORY(UnaryOpT::sqrt, OpSqrt);
// Trigonometric functions
DECLARE_OP_FACTORY(UnaryOpT::sin, OpSin);
DECLARE_OP_FACTORY(UnaryOpT::cos, OpCos);
DECLARE_OP_FACTORY(UnaryOpT::tan, OpTan);
DECLARE_OP_FACTORY(UnaryOpT::asin, OpASin);
DECLARE_OP_FACTORY(UnaryOpT::acos, OpACos);
DECLARE_OP_FACTORY(UnaryOpT::atan, OpATan);
// Hyperbolic functions
DECLARE_OP_FACTORY(UnaryOpT::sinh, OpSinh);
DECLARE_OP_FACTORY(UnaryOpT::cosh, OpCosh);
DECLARE_OP_FACTORY(UnaryOpT::tanh, OpTanh);
DECLARE_OP_FACTORY(UnaryOpT::asinh, OpASinh);
DECLARE_OP_FACTORY(UnaryOpT::acosh, OpACosh);
DECLARE_OP_FACTORY(UnaryOpT::atanh, OpATanh);
// Error and Gamma function
DECLARE_OP_FACTORY(UnaryOpT::erf, OpErf);
DECLARE_OP_FACTORY(UnaryOpT::erfc, OpErfc);
DECLARE_OP_FACTORY(UnaryOpT::tgamma, OpTGamma);
DECLARE_OP_FACTORY(UnaryOpT::lgamma, OpLGamma);
DECLARE_OP_FACTORY(UnaryOpT::digamma, OpDiGamma);
// Nearest Integer
DECLARE_OP_FACTORY(UnaryOpT::ceil, OpCeil);
DECLARE_OP_FACTORY(UnaryOpT::floor, OpFloor);
DECLARE_OP_FACTORY(UnaryOpT::round, OpRound);
// Classification
DECLARE_OP_FACTORY(UnaryOpT::isinf, OpIsInf);
DECLARE_OP_FACTORY(UnaryOpT::isnan, OpIsNaN);
DECLARE_OP_FACTORY(UnaryOpT::isfinite, OpIsFinite);
// Misc activation functions
DECLARE_OP_FACTORY(UnaryOpT::sigmoid, OpSigmoid);
DECLARE_OP_FACTORY(UnaryOpT::softplus, OpSoftplus);
DECLARE_OP_FACTORY(UnaryOpT::relu, OpReLU);
DECLARE_OP_FACTORY(UnaryOpT::relu6, OpReLU6);
DECLARE_OP_FACTORY(UnaryOpT::leaky_relu, OpLeakyReLU);
DECLARE_OP_FACTORY(UnaryOpT::elu, OpELU);
DECLARE_OP_FACTORY(UnaryOpT::selu, OpSELU);
DECLARE_OP_FACTORY(UnaryOpT::silu, OpSiLU);
DECLARE_OP_FACTORY(UnaryOpT::hardsigmoid, OpHardSigmoid);
DECLARE_OP_FACTORY(UnaryOpT::hardtanh, OpHardTanh);
DECLARE_OP_FACTORY(UnaryOpT::log_sigmoid, OpLogSigmoid);
DECLARE_OP_FACTORY(UnaryOpT::softsign, OpSoftsign);
#undef DECLARE_OP_FACTORY

// Result type from unary op
template <typename T, UnaryOpT Op>
struct Result {
    using type = T;
    constexpr static bool CastBeforeOp = true;
    constexpr static auto scalar(ScalarType dtype) -> ScalarType {
        return dtype;
    }
};

// For specified ops, int gets cast to default float and floats keep their input type
// NOLINTNEXTLINE(*-macro-usage)
#define DECLARE_RESULT_TYPE_INT_TO_FLOAT(UNARY_OPT)                                          \
    template <typename T>                                                                    \
    struct Result<T, UNARY_OPT> {                                                            \
        using type = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>; \
        constexpr static bool CastBeforeOp = true;                                           \
        constexpr static auto scalar([[maybe_unused]] ScalarType dtype) -> ScalarType {      \
            return to_scalar<type>::type;                                                    \
        }                                                                                    \
    };

DECLARE_RESULT_TYPE_INT_TO_FLOAT(UnaryOpT::log);
DECLARE_RESULT_TYPE_INT_TO_FLOAT(UnaryOpT::log10);
DECLARE_RESULT_TYPE_INT_TO_FLOAT(UnaryOpT::log2);
DECLARE_RESULT_TYPE_INT_TO_FLOAT(UnaryOpT::log1p);
DECLARE_RESULT_TYPE_INT_TO_FLOAT(UnaryOpT::exp);
DECLARE_RESULT_TYPE_INT_TO_FLOAT(UnaryOpT::exp2);
DECLARE_RESULT_TYPE_INT_TO_FLOAT(UnaryOpT::expm1);
DECLARE_RESULT_TYPE_INT_TO_FLOAT(UnaryOpT::sqrt);
DECLARE_RESULT_TYPE_INT_TO_FLOAT(UnaryOpT::sin);
DECLARE_RESULT_TYPE_INT_TO_FLOAT(UnaryOpT::cos);
DECLARE_RESULT_TYPE_INT_TO_FLOAT(UnaryOpT::tan);
DECLARE_RESULT_TYPE_INT_TO_FLOAT(UnaryOpT::asin);
DECLARE_RESULT_TYPE_INT_TO_FLOAT(UnaryOpT::acos);
DECLARE_RESULT_TYPE_INT_TO_FLOAT(UnaryOpT::atan);
DECLARE_RESULT_TYPE_INT_TO_FLOAT(UnaryOpT::sinh);
DECLARE_RESULT_TYPE_INT_TO_FLOAT(UnaryOpT::cosh);
DECLARE_RESULT_TYPE_INT_TO_FLOAT(UnaryOpT::tanh);
DECLARE_RESULT_TYPE_INT_TO_FLOAT(UnaryOpT::asinh);
DECLARE_RESULT_TYPE_INT_TO_FLOAT(UnaryOpT::acosh);
DECLARE_RESULT_TYPE_INT_TO_FLOAT(UnaryOpT::atanh);
DECLARE_RESULT_TYPE_INT_TO_FLOAT(UnaryOpT::erf);
DECLARE_RESULT_TYPE_INT_TO_FLOAT(UnaryOpT::erfc);
DECLARE_RESULT_TYPE_INT_TO_FLOAT(UnaryOpT::tgamma);
DECLARE_RESULT_TYPE_INT_TO_FLOAT(UnaryOpT::lgamma);
DECLARE_RESULT_TYPE_INT_TO_FLOAT(UnaryOpT::digamma);
DECLARE_RESULT_TYPE_INT_TO_FLOAT(UnaryOpT::sigmoid);
DECLARE_RESULT_TYPE_INT_TO_FLOAT(UnaryOpT::softplus);
DECLARE_RESULT_TYPE_INT_TO_FLOAT(UnaryOpT::relu);
DECLARE_RESULT_TYPE_INT_TO_FLOAT(UnaryOpT::relu6);
DECLARE_RESULT_TYPE_INT_TO_FLOAT(UnaryOpT::leaky_relu);
DECLARE_RESULT_TYPE_INT_TO_FLOAT(UnaryOpT::elu);
DECLARE_RESULT_TYPE_INT_TO_FLOAT(UnaryOpT::selu);
DECLARE_RESULT_TYPE_INT_TO_FLOAT(UnaryOpT::silu);
DECLARE_RESULT_TYPE_INT_TO_FLOAT(UnaryOpT::hardsigmoid);
DECLARE_RESULT_TYPE_INT_TO_FLOAT(UnaryOpT::hardtanh);
DECLARE_RESULT_TYPE_INT_TO_FLOAT(UnaryOpT::log_sigmoid);
DECLARE_RESULT_TYPE_INT_TO_FLOAT(UnaryOpT::softsign);
#undef DECLARE_RESULT_TYPE_INT_TO_FLOAT

// For specified ops, result type is boolean
// NOLINTNEXTLINE(*-macro-usage)
#define DECLARE_RESULT_TYPE_TO_BOOLEAN(UNARY_OPT)                                       \
    template <typename T>                                                               \
    struct Result<T, UNARY_OPT> {                                                       \
        using type = to_ctype_t<kBool>;                                                 \
        constexpr static bool CastBeforeOp = false;                                     \
        constexpr static auto scalar([[maybe_unused]] ScalarType dtype) -> ScalarType { \
            return kBool;                                                               \
        }                                                                               \
    };

DECLARE_RESULT_TYPE_TO_BOOLEAN(UnaryOpT::isinf);
DECLARE_RESULT_TYPE_TO_BOOLEAN(UnaryOpT::isnan);
DECLARE_RESULT_TYPE_TO_BOOLEAN(UnaryOpT::isfinite);
DECLARE_RESULT_TYPE_TO_BOOLEAN(UnaryOpT::logical_not);
#undef DECLARE_RESULT_TYPE_TO_BOOLEAN

}    // namespace tinytensor::common::unary

#endif    // TINYTENSOR_BACKEND_COMMON_UNARY_H_
