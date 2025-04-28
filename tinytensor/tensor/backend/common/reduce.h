// reduce.h
// Common reduction utils

#ifndef TINYTENSOR_BACKEND_COMMON_REDUCE_H_
#define TINYTENSOR_BACKEND_COMMON_REDUCE_H_

#include <tt/scalar.h>

#include "tensor/backend/common/kernel/reduce.hpp"

#include <type_traits>

namespace tinytensor::common::reduce {

// Reduction operators
enum class ReduceOpT {
    min,
    argmin,
    max,
    argmax,
    sum,
    all,
    any,
    mean,
};

// Reduction op to kernel mapping
template <typename T, ReduceOpT Op>
struct OpFactory;

// NOLINTNEXTLINE(*-macro-usage)
#define DECLARE_OP_FACTORY(REDUCE_OPT, KERN_OPT) \
    template <typename T>                        \
    struct OpFactory<T, REDUCE_OPT> {            \
        using KernelOp = KERN_OPT<T>;            \
    };

DECLARE_OP_FACTORY(ReduceOpT::min, kernel::reduce::OpMinimum);
DECLARE_OP_FACTORY(ReduceOpT::argmin, kernel::reduce::OpArgMinimum);
DECLARE_OP_FACTORY(ReduceOpT::max, kernel::reduce::OpMaximum);
DECLARE_OP_FACTORY(ReduceOpT::argmax, kernel::reduce::OpArgMaximum);
DECLARE_OP_FACTORY(ReduceOpT::sum, kernel::reduce::OpSum);
DECLARE_OP_FACTORY(ReduceOpT::all, kernel::reduce::OpAnd);
DECLARE_OP_FACTORY(ReduceOpT::any, kernel::reduce::OpOr);
DECLARE_OP_FACTORY(ReduceOpT::mean, kernel::reduce::OpMean);
#undef DECLARE_OP_FACTORY

template <typename T, ReduceOpT Op>
struct Result {
    using type = T;
    using val_type = T;
    using res_type = T;
};

// Sum reductions on low width int types get cast to larger width to circumvent overflows
template <typename T>
struct Result<T, ReduceOpT::sum> {
    using type = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kI64>>;
    using val_type = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kI64>>;
    using res_type = val_type;
};

// Mean reductions on on int types need to cast to default float
template <typename T>
struct Result<T, ReduceOpT::mean> {
    using type = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
    using val_type = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
    using res_type = val_type;
};

// Argmin/max reductions results are indexing ints
template <typename T>
struct Result<T, ReduceOpT::argmin> {
    using type = int;
    using val_type = T;
    using res_type = int;
};
template <typename T>
struct Result<T, ReduceOpT::argmax> {
    using type = int;
    using val_type = T;
    using res_type = int;
};

}    // namespace tinytensor::common::reduce

#endif    // TINYTENSOR_BACKEND_COMMON_REDUCE_H_
