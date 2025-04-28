// distribution.h
// Common Distribution utils

#ifndef TINYTENSOR_BACKEND_COMMON_DISTRIBUTION_H_
#define TINYTENSOR_BACKEND_COMMON_DISTRIBUTION_H_

#include <tt/scalar.h>

#include "tensor/backend/common/kernel/distribution.hpp"

namespace tinytensor::common::distribution {

// Binary operators
enum class DistributionOpT {
    uniform_int,
    uniform_real,
    bernoulli,
    binomial,
    geometric,
    poisson,
    exponential,
    normal,
    lognormal,
    cauchy,
    weibull,
};

// Distribution OP to kernel mapping
template <typename T, DistributionOpT Op>
struct OpFactory;

// NOLINTNEXTLINE(*-macro-usage)
#define DECLARE_OP_FACTORY(DISTRIBUTION_OPT, KERN_OPT) \
    template <typename T>                              \
    struct OpFactory<T, DISTRIBUTION_OPT> {            \
        using KernelOp = KERN_OPT<T>;                  \
    };

DECLARE_OP_FACTORY(DistributionOpT::uniform_int, common::kernel::distribution::OpUniformInt);
DECLARE_OP_FACTORY(DistributionOpT::uniform_real, common::kernel::distribution::OpUniformReal);
DECLARE_OP_FACTORY(DistributionOpT::bernoulli, common::kernel::distribution::OpBernoulli);
DECLARE_OP_FACTORY(DistributionOpT::binomial, common::kernel::distribution::OpBinomial);
DECLARE_OP_FACTORY(DistributionOpT::geometric, common::kernel::distribution::OpGeometric);
DECLARE_OP_FACTORY(DistributionOpT::poisson, common::kernel::distribution::OpPoisson);
DECLARE_OP_FACTORY(DistributionOpT::exponential, common::kernel::distribution::OpExponential);
DECLARE_OP_FACTORY(DistributionOpT::normal, common::kernel::distribution::OpNormal);
DECLARE_OP_FACTORY(DistributionOpT::cauchy, common::kernel::distribution::OpCauchy);
DECLARE_OP_FACTORY(DistributionOpT::lognormal, common::kernel::distribution::OpLogNormal);
DECLARE_OP_FACTORY(DistributionOpT::weibull, common::kernel::distribution::OpWeibull);
#undef DECLARE_OP_FACTORY

// Distribution OP to kernel mapping
template <typename T, DistributionOpT Op>
struct OpProperties;

// NOLINTNEXTLINE(*-macro-usage)
#define DECLARE_OP_PROPERTIES(DISTRIBUTION_OPT, SUPPORTED_CONCEPT) \
    template <typename T>                                          \
    struct OpProperties<T, DISTRIBUTION_OPT> {                     \
        static constexpr bool IsSupported = SUPPORTED_CONCEPT<T>;  \
    };

DECLARE_OP_PROPERTIES(DistributionOpT::uniform_int, IsScalarType);
DECLARE_OP_PROPERTIES(DistributionOpT::uniform_real, IsScalarFloatType);
DECLARE_OP_PROPERTIES(DistributionOpT::bernoulli, IsScalarFloatType);
DECLARE_OP_PROPERTIES(DistributionOpT::binomial, IsScalarFloatType);
DECLARE_OP_PROPERTIES(DistributionOpT::geometric, IsScalarFloatType);
DECLARE_OP_PROPERTIES(DistributionOpT::poisson, IsScalarFloatType);
DECLARE_OP_PROPERTIES(DistributionOpT::exponential, IsScalarFloatType);
DECLARE_OP_PROPERTIES(DistributionOpT::normal, IsScalarFloatType);
DECLARE_OP_PROPERTIES(DistributionOpT::cauchy, IsScalarFloatType);
DECLARE_OP_PROPERTIES(DistributionOpT::lognormal, IsScalarFloatType);
DECLARE_OP_PROPERTIES(DistributionOpT::weibull, IsScalarFloatType);
#undef DECLARE_OP_PROPERTIES

}    // namespace tinytensor::common::distribution

#endif    // TINYTENSOR_BACKEND_COMMON_DISTRIBUTION_H_
