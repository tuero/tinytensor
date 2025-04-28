// scalar.cpp
// Scalar types

#include <tt/exception.h>
#include <tt/scalar.h>

#include "tensor/backend/common/dispatch.h"

#include <array>
#include <cstddef>
#include <iostream>
#include <string>
#include <variant>

namespace tinytensor {

constexpr std::array<std::array<ScalarType, kNumScalars>, kNumScalars> promote_type_map = {{
    /*           kBool  kU8  kI16  kI32  kI64  kF32  kF64 */
    /* kBool */ {kBool, kU8, kI16, kI32, kI64, kF32, kF64},
    /* kU8   */ {kU8, kU8, kI16, kI32, kI64, kF32, kF64},
    /* kI16  */ {kI16, kI16, kI16, kI32, kI64, kF32, kF64},
    /* kI32  */ {kI32, kI32, kI32, kI32, kI64, kF32, kF64},
    /* kI64  */ {kI64, kI64, kI64, kI64, kI64, kF32, kF64},
    /* kF32  */ {kF32, kF32, kF32, kF32, kF32, kF32, kF64},
    /* kF64  */ {kF64, kF64, kF64, kF64, kF64, kF64, kF64},
}};

auto promote_types(ScalarType type1, ScalarType type2) -> ScalarType {
    auto idx1 = static_cast<std::size_t>(type1);
    auto idx2 = static_cast<std::size_t>(type2);
    if (idx1 >= kNumScalars || idx2 >= kNumScalars) {
        TT_EXCEPTION("Unknown ScalarType");
    }
    return promote_type_map[idx1][idx2];    // NOLINT(*-constant-array-index)
}

auto Scalar::operator+(const Scalar &other) -> Scalar {
    ScalarType result_type = promote_types(dtype(), other.dtype());
    return DISPATCH_ALL_TYPES(result_type, "", [&]() { return Scalar(to<scalar_t>() + other.to<scalar_t>()); });
}

auto Scalar::operator-(const Scalar &other) -> Scalar {
    ScalarType result_type = promote_types(dtype(), other.dtype());
    return DISPATCH_ALL_TYPES(result_type, "", [&]() { return Scalar(to<scalar_t>() - other.to<scalar_t>()); });
}

auto Scalar::operator*(const Scalar &other) -> Scalar {
    ScalarType result_type = promote_types(dtype(), other.dtype());
    return DISPATCH_ALL_TYPES(result_type, "", [&]() { return Scalar(to<scalar_t>() * other.to<scalar_t>()); });
}

auto Scalar::operator/(const Scalar &other) -> Scalar {
    ScalarType result_type = promote_types(dtype(), other.dtype());
    return DISPATCH_ALL_TYPES(result_type, "", [&]() { return Scalar(to<scalar_t>() / other.to<scalar_t>()); });
}

auto Scalar::operator==(const Scalar &other) -> bool {
    ScalarType result_type = promote_types(dtype(), other.dtype());
    return DISPATCH_ALL_TYPES(result_type, "", [&]() { return to<scalar_t>() == other.to<scalar_t>(); });
}

auto Scalar::operator!=(const Scalar &other) -> bool {
    ScalarType result_type = promote_types(dtype(), other.dtype());
    return DISPATCH_ALL_TYPES(result_type, "", [&]() { return to<scalar_t>() != other.to<scalar_t>(); });
}

auto Scalar::operator<(const Scalar &other) -> bool {
    ScalarType result_type = promote_types(dtype(), other.dtype());
    return DISPATCH_ALL_TYPES(result_type, "", [&]() { return to<scalar_t>() < other.to<scalar_t>(); });
}

auto Scalar::operator>(const Scalar &other) -> bool {
    ScalarType result_type = promote_types(dtype(), other.dtype());
    return DISPATCH_ALL_TYPES(result_type, "", [&]() { return to<scalar_t>() > other.to<scalar_t>(); });
}

auto Scalar::operator<=(const Scalar &other) -> bool {
    ScalarType result_type = promote_types(dtype(), other.dtype());
    return DISPATCH_ALL_TYPES(result_type, "", [&]() { return to<scalar_t>() <= other.to<scalar_t>(); });
}
auto Scalar::operator>=(const Scalar &other) -> bool {
    ScalarType result_type = promote_types(dtype(), other.dtype());
    return DISPATCH_ALL_TYPES(result_type, "", [&]() { return to<scalar_t>() >= other.to<scalar_t>(); });
}

auto to_string(const Scalar &scalar) -> std::string {
    return std::visit([&](auto &&value) { return std::to_string(value); }, scalar.data);
}

auto operator<<(std::ostream &os, const ScalarType &dtype) -> std::ostream & {
    return os << to_string(dtype);
}

auto operator<<(std::ostream &os, const Scalar &scalar) -> std::ostream & {
    return os << to_string(scalar);
}

}    // namespace tinytensor
