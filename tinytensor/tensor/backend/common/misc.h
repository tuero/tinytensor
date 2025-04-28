// misc.h
// Common misc utils

#ifndef TINYTENSOR_BACKEND_COMMON_MISC_H_
#define TINYTENSOR_BACKEND_COMMON_MISC_H_

#include <tt/scalar.h>

#include "tensor/backend/common/kernel/misc.hpp"

namespace tinytensor::common::misc {

using namespace kernel::misc;

// Binary operators
enum class MiscOpT {
    where,
};

// Binary op to kernel mapping
template <typename T, MiscOpT Op>
struct OpFactory;

// NOLINTNEXTLINE(*-macro-usage)
#define DECLARE_OP_FACTORY(MISC_OPT, KERN_OPT) \
    template <typename T>                      \
    struct OpFactory<T, MISC_OPT> {            \
        using KernelOp = KERN_OPT<T>;          \
    };

DECLARE_OP_FACTORY(MiscOpT::where, OpWhere);
#undef DECLARE_OP_FACTORY

// Result type from binary op
template <typename T, MiscOpT Op>
struct Result {
    using type = T;
};

// Distribution OP to kernel mapping
template <typename T, MiscOpT Op>
struct OpProperties;

// NOLINTNEXTLINE(*-macro-usage)
#define DECLARE_OP_PROPERTIES(MISC_OPT, SUPPORTED_CONCEPT)        \
    template <typename T>                                         \
    struct OpProperties<T, MISC_OPT> {                            \
        static constexpr bool IsSupported = SUPPORTED_CONCEPT<T>; \
    };

DECLARE_OP_PROPERTIES(MiscOpT::where, IsScalarType);
#undef DECLARE_OP_PROPERTIES

}    // namespace tinytensor::common::misc

#endif    // TINYTENSOR_BACKEND_COMMON_MISC_H_
