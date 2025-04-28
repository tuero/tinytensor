// binary.h
// Common binary utils

#ifndef TINYTENSOR_BACKEND_COMMON_BINARY_H_
#define TINYTENSOR_BACKEND_COMMON_BINARY_H_

#include <tt/scalar.h>

#include "tensor/backend/common/kernel/binary.hpp"

#include <type_traits>

namespace tinytensor::common::binary {

using namespace kernel::binary;

// Binary operators
enum class BinaryOpT {
    add,
    subtract,
    multiply,
    divide,
    minimum,
    maximum,
    pow,
    equal,
    not_equal,
    less_than,
    less_than_eq,
    greater_than,
    greater_than_eq,
    logical_or,
    logical_and,
    bitwise_or,
    bitwise_and,
    bitwise_xor,
    bitwise_left_shift,
    bitwise_right_shift,
    modulo,
};

// Binary op to kernel mapping
template <typename T, BinaryOpT Op>
struct OpFactory;

// NOLINTNEXTLINE(*-macro-usage)
#define DECLARE_OP_FACTORY(BINARY_OPT, KERN_OPT) \
    template <typename T>                        \
    struct OpFactory<T, BINARY_OPT> {            \
        using KernelOp = KERN_OPT<T>;            \
    };

DECLARE_OP_FACTORY(BinaryOpT::add, OpAdd);
DECLARE_OP_FACTORY(BinaryOpT::subtract, OpSub);
DECLARE_OP_FACTORY(BinaryOpT::multiply, OpMul);
DECLARE_OP_FACTORY(BinaryOpT::divide, OpDiv);
DECLARE_OP_FACTORY(BinaryOpT::minimum, OpMinimum);
DECLARE_OP_FACTORY(BinaryOpT::maximum, OpMaximum);
DECLARE_OP_FACTORY(BinaryOpT::pow, OpPow);
DECLARE_OP_FACTORY(BinaryOpT::equal, OpEq);
DECLARE_OP_FACTORY(BinaryOpT::not_equal, OpNe);
DECLARE_OP_FACTORY(BinaryOpT::less_than, OpLt);
DECLARE_OP_FACTORY(BinaryOpT::less_than_eq, OpLe);
DECLARE_OP_FACTORY(BinaryOpT::greater_than, OpGt);
DECLARE_OP_FACTORY(BinaryOpT::greater_than_eq, OpGe);
DECLARE_OP_FACTORY(BinaryOpT::logical_or, OpLogicalOr);
DECLARE_OP_FACTORY(BinaryOpT::logical_and, OpLogicalAnd);
DECLARE_OP_FACTORY(BinaryOpT::bitwise_or, OpBitwiseOr);
DECLARE_OP_FACTORY(BinaryOpT::bitwise_and, OpBitwiseAnd);
DECLARE_OP_FACTORY(BinaryOpT::bitwise_xor, OpBitwiseXor);
DECLARE_OP_FACTORY(BinaryOpT::bitwise_left_shift, OpBitwiseLeftShift);
DECLARE_OP_FACTORY(BinaryOpT::bitwise_right_shift, OpBitwiseRightShift);
DECLARE_OP_FACTORY(BinaryOpT::modulo, OpModulo);
#undef DECLARE_OP_FACTORY

// Result type from binary op
template <typename T, BinaryOpT Op>
struct Result {
    using type = T;
    constexpr static bool CastBeforeOp = false;
    constexpr static auto scalar(ScalarType dtype) -> ScalarType {
        return dtype;
    }
};

// Division casts to default float if integral type
// Bool not allowed to perform division, so we don't have to worry about to_scalar<uint8_t>
template <typename T>
struct Result<T, BinaryOpT::divide> {
    using type = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
    constexpr static bool CastBeforeOp = true;
    constexpr static auto scalar([[maybe_unused]] ScalarType dtype) -> ScalarType {
        return to_scalar<type>::type;
    }
};

// NOLINTNEXTLINE(*-macro-usage)
#define DECLARE_RESULT_TYPE_BOOL(BINARY_OPT, SCALAR_TYPE, CAST_BEFORE)                  \
    template <typename T>                                                               \
    struct Result<T, BINARY_OPT> {                                                      \
        using type = to_ctype_t<SCALAR_TYPE>;                                           \
        constexpr static bool CastBeforeOp = CAST_BEFORE;                               \
        constexpr static auto scalar([[maybe_unused]] ScalarType dtype) -> ScalarType { \
            return SCALAR_TYPE;                                                         \
        }                                                                               \
    };

DECLARE_RESULT_TYPE_BOOL(BinaryOpT::equal, ScalarType::bool8, false);
DECLARE_RESULT_TYPE_BOOL(BinaryOpT::not_equal, ScalarType::bool8, false);
DECLARE_RESULT_TYPE_BOOL(BinaryOpT::less_than, ScalarType::bool8, false);
DECLARE_RESULT_TYPE_BOOL(BinaryOpT::less_than_eq, ScalarType::bool8, false);
DECLARE_RESULT_TYPE_BOOL(BinaryOpT::greater_than, ScalarType::bool8, false);
DECLARE_RESULT_TYPE_BOOL(BinaryOpT::greater_than_eq, ScalarType::bool8, false);
DECLARE_RESULT_TYPE_BOOL(BinaryOpT::logical_or, ScalarType::bool8, true);
DECLARE_RESULT_TYPE_BOOL(BinaryOpT::logical_and, ScalarType::bool8, true);
#undef DECLARE_RESULT_TYPE_BOOL

}    // namespace tinytensor::common::binary

#endif    // TINYTENSOR_BACKEND_COMMON_BINARY_H_
