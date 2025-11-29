// scalar.h
// Scalar types

#ifndef TINYTENSOR_SCALAR_H_
#define TINYTENSOR_SCALAR_H_

#include <tt/concepts.h>
#include <tt/exception.h>
#include <tt/export.h>

#include <cstddef>
#include <cstdint>
#include <format>
#include <ostream>
#include <string>
#include <type_traits>
#include <variant>

namespace tinytensor {

enum class TINYTENSOR_EXPORT ScalarType {
    bool8 = 0,    // 8 bit bool
    u8,           // 8 bit unsigned int
    i16,          // 16 bit int
    i32,          // 32 bit int
    i64,          // 64 bit int
    f32,          // 32 bit float
    f64,          // 64 bit float
};

// Shortnames
constexpr ScalarType kBool = ScalarType::bool8;
constexpr ScalarType kU8 = ScalarType::u8;
constexpr ScalarType kI16 = ScalarType::i16;
constexpr ScalarType kI32 = ScalarType::i32;
constexpr ScalarType kI64 = ScalarType::i64;
constexpr ScalarType kF32 = ScalarType::f32;
constexpr ScalarType kF64 = ScalarType::f64;
constexpr std::size_t kNumScalars = 7;

// Default scalar types
constexpr ScalarType kDefaultInt = ScalarType::i32;
constexpr ScalarType kDefaultFloat = ScalarType::f32;

// Concepts for valid types
template <typename T>
concept IsScalarFloatType = IsAnyOf<T, float, double>;

template <typename T>
concept IsScalarIntType = IsAnyOf<T, uint8_t, int16_t, int32_t, int64_t>;

template <typename T>
concept IsScalarType = IsScalarIntType<T> || IsScalarFloatType<T>;

/**
 * Check if the scalar type is integral
 * @param dtype The dtype to check
 * @return True if the dtype is integral, false otherwise
 */
TINYTENSOR_EXPORT constexpr auto is_integral_dtype(ScalarType dtype) -> bool {
    return dtype == kU8 || dtype == kI16 || dtype == kI32 || dtype == kI64;
}

/**
 * Check if the scalar type is floating point
 * @param dtype The dtype to check
 * @return True if the dtype is floating point, false otherwise
 */
TINYTENSOR_EXPORT constexpr auto is_float_dtype(ScalarType dtype) -> bool {
    return dtype == kF32 || dtype == kF64;
}

/**
 * Convert scalar type to string
 */
TINYTENSOR_EXPORT constexpr auto to_string(ScalarType type) -> std::string {
    switch (type) {
    case ScalarType::bool8:
        return "bool8";
    case ScalarType::u8:
        return "uint8";
    case ScalarType::i16:
        return "int16";
    case ScalarType::i32:
        return "int32";
    case ScalarType::i64:
        return "int64";
    case ScalarType::f32:
        return "float32";
    case ScalarType::f64:
        return "float64";
    }
    TT_EXCEPTION("Unknown ScalarType" + std::to_string(static_cast<std::underlying_type_t<ScalarType>>(type)));
}

TINYTENSOR_EXPORT auto operator<<(std::ostream &os, const ScalarType &dtype) -> std::ostream &;

// Get the promoted type (smallest size and scalar type not smaller than both type1 and type2)
TINYTENSOR_EXPORT auto promote_types(ScalarType type1, ScalarType type2) -> ScalarType;

// ctype to scalar type
template <typename T>
struct to_scalar;

// scalar type to ctype
template <ScalarType T>
struct to_ctype;

// NOLINTNEXTLINE(*-macro-usage)
#define DECLARE_TYPE_TRAITS(CTYPE, SCALAR_TYPE, REQUIRED_BYTES) \
    template <>                                                 \
    struct TINYTENSOR_EXPORT to_scalar<CTYPE> {                 \
        static_assert(sizeof(CTYPE) == REQUIRED_BYTES);         \
        static constexpr ScalarType type = SCALAR_TYPE;         \
    };                                                          \
    template <>                                                 \
    struct TINYTENSOR_EXPORT to_ctype<SCALAR_TYPE> {            \
        static_assert(sizeof(CTYPE) == REQUIRED_BYTES);         \
        using type = CTYPE;                                     \
    };

DECLARE_TYPE_TRAITS(uint8_t, kU8, 1);
DECLARE_TYPE_TRAITS(int16_t, kI16, 2);
DECLARE_TYPE_TRAITS(int32_t, kI32, 4);
DECLARE_TYPE_TRAITS(int64_t, kI64, 8);
DECLARE_TYPE_TRAITS(float, kF32, 4);
DECLARE_TYPE_TRAITS(double, kF64, 8);

// Bool is specially handled
template <>
struct TINYTENSOR_EXPORT to_scalar<bool> {
    static constexpr ScalarType type = kBool;
};
template <>
struct TINYTENSOR_EXPORT to_ctype<kBool> {
    using type = uint8_t;
};

template <ScalarType T>
using to_ctype_t = to_ctype<T>::type;

// Type erased scalar type
class TINYTENSOR_EXPORT Scalar {
public:
    explicit Scalar(bool v)
        : data(static_cast<uint8_t>(v)), type(kBool) {}
    explicit Scalar(uint8_t v)
        : data(v), type(kU8) {}
    explicit Scalar(int16_t v)
        : data(v), type(kI16) {}
    explicit Scalar(int32_t v)
        : data(v), type(kI32) {}
    explicit Scalar(int64_t v)
        : data(v), type(kI64) {}
    explicit Scalar(float v)
        : data(v), type(kF32) {}
    explicit Scalar(double v)
        : data(v), type(kF64) {}

    template <typename T>
        requires(IsScalarType<T> || std::is_same_v<T, bool>)
    explicit Scalar(T v, ScalarType dtype)
        : type(dtype) {
        switch (dtype) {
        case kBool:
            // Ensure converting types to bool keeps within range 0-1
            data = static_cast<uint8_t>(static_cast<bool>(v));
            break;
        case kU8:
            data = static_cast<to_ctype_t<kU8>>(v);
            break;
        case kI16:
            data = static_cast<to_ctype_t<kI16>>(v);
            break;
        case kI32:
            data = static_cast<to_ctype_t<kI32>>(v);
            break;
        case kI64:
            data = static_cast<to_ctype_t<kI64>>(v);
            break;
        case kF32:
            data = static_cast<to_ctype_t<kF32>>(v);
            break;
        case kF64:
            data = static_cast<to_ctype_t<kF64>>(v);
            break;
        }
    }

    [[nodiscard]] constexpr auto dtype() const -> ScalarType {
        return type;
    }

    /**
     * Get and cast the underyling stored value to the given type
     */
    template <typename T>
        requires(IsScalarType<T> || std::is_same_v<T, bool>)
    [[nodiscard]] auto to() const -> T {
        return std::visit([&](auto &&scalar) { return static_cast<T>(scalar); }, data);
    }

    /**
     * Get and cast the underyling stored value to the given type
     * @param dtype The scalar type for the result
     */
    [[nodiscard]] auto to(ScalarType dtype) const -> Scalar {
        return std::visit([&](auto &&scalar) { return Scalar(scalar, dtype); }, data);
    }

    // Operations on Scalars will have result converted to underlying promotion type based on C++ defaults
    auto operator+(const Scalar &other) -> Scalar;
    auto operator-(const Scalar &other) -> Scalar;
    auto operator*(const Scalar &other) -> Scalar;
    auto operator/(const Scalar &other) -> Scalar;
    auto operator==(const Scalar &other) -> bool;
    auto operator!=(const Scalar &other) -> bool;
    auto operator<(const Scalar &other) -> bool;
    auto operator>(const Scalar &other) -> bool;
    auto operator<=(const Scalar &other) -> bool;
    auto operator>=(const Scalar &other) -> bool;

    friend std::formatter<tinytensor::Scalar>;
    friend auto to_string(const Scalar &scalar) -> std::string;

private:
    std::variant<uint8_t, int16_t, int32_t, int64_t, double, float> data;
    ScalarType type;
};

/**
 * Convert Scalar to string
 */
TINYTENSOR_EXPORT auto to_string(const Scalar &scalar) -> std::string;

TINYTENSOR_EXPORT auto operator<<(std::ostream &os, const Scalar &scalar) -> std::ostream &;

// Cast floating or integral Ts to scalar types
template <IsScalarFloatType T>
TINYTENSOR_EXPORT auto cast_to_default(T t) -> Scalar {
    return Scalar(t, kDefaultFloat);
}
template <IsScalarIntType T>
TINYTENSOR_EXPORT auto cast_to_default(T t) -> Scalar {
    return Scalar(t, kDefaultInt);
}

}    // namespace tinytensor

template <>
struct TINYTENSOR_EXPORT std::formatter<tinytensor::ScalarType> : std::formatter<std::string> {
    auto format(const tinytensor::ScalarType &dtype, format_context &ctx) const {
        return formatter<string>::format(std::format("{}", to_string(dtype)), ctx);
    }
};

template <>
struct TINYTENSOR_EXPORT std::formatter<tinytensor::Scalar> : std::formatter<std::string> {
    auto format(const tinytensor::Scalar &scalar, format_context &ctx) const {
        return formatter<string>::format(std::format("{}", to_string(scalar)), ctx);
    }
};

#endif    // TINYTENSOR_SCALAR_H_
