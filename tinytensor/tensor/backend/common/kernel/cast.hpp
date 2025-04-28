// cast.hpp
// Common element-wise cast kernels

#ifndef TINYTENSOR_BACKEND_COMMON_KERNEL_CAST_H_
#define TINYTENSOR_BACKEND_COMMON_KERNEL_CAST_H_

#include <tt/macros.h>
#include <tt/scalar.h>

#if defined(__CUDACC__)
#include <nvfunctional>
#else
#include <functional>
#endif

namespace tinytensor::common::kernel::cast {

// Util/misc
template <typename T, typename R>
struct OpCast {
    TT_STD_FUNC<R(T)> TT_HOST_DEVICE operator()() const {
        return [](T val) { return static_cast<R>(val); };
    }
};

}    // namespace tinytensor::common::kernel::cast

#endif    // TINYTENSOR_BACKEND_COMMON_KERNEL_CAST_H_
