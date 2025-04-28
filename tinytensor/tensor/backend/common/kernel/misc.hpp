// misc.hpp
// Misc element-wise unary kernels

#ifndef TINYTENSOR_BACKEND_MISC_KERNEL_UNARY_H_
#define TINYTENSOR_BACKEND_MISC_KERNEL_UNARY_H_

#include <tt/macros.h>

#if defined(__CUDACC__)
#include <nvfunctional>
#else
#include <functional>
#endif

namespace tinytensor::common::kernel::misc {

// Sigmoid Linear Unit
template <typename T>
struct OpWhere {
    TT_STD_FUNC<T(bool, T, T)> TT_HOST_DEVICE operator()() const {
        return [](bool cond, auto lhs, auto rhs) -> T { return cond ? lhs : rhs; };
    }
};

}    // namespace tinytensor::common::kernel::misc

#endif    // TINYTENSOR_BACKEND_MISC_KERNEL_UNARY_H_
