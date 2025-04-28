// unary.hpp
// Element-wise unary kernel

#ifndef TINYTENSOR_BACKEND_CPU_KERNEL_UNARY_H_
#define TINYTENSOR_BACKEND_CPU_KERNEL_UNARY_H_

#include "tensor/backend/common/span.h"
#include "tensor/backend/common/util.h"
#include "tensor/backend/cpu/data_types.h"

namespace tinytensor::cpu::kernel::unary {

template <bool CastBeforeOp, typename T, typename R, typename OP, typename... Args>
void unary_kernel(const DataInfo<const T> di, HostSpan<R> res, OP op, int N, Args... args) {
    for (int i = 0; i < N; ++i) {
        const auto idx = to_flat_index(i, di.shape, di.stride, di.offset);
        // Some ops need to cast input before op, some after
        if constexpr (CastBeforeOp) {
            res[i] = op()(static_cast<R>(di.data[idx]), args...);
        } else {
            res[i] = static_cast<R>(op()(di.data[idx], args...));
        }
    }
}

template <typename T, typename OP, typename... Args>
void unary_kernel_inplace(DataInfo<T> di, OP op, int N, Args... args) {
    for (int i = 0; i < N; ++i) {
        const auto idx = to_flat_index(i, di.shape, di.stride, di.offset);
        di.data[idx] = op()(di.data[idx], args...);
    }
}

}    // namespace tinytensor::cpu::kernel::unary

#endif    // TINYTENSOR_BACKEND_CPU_KERNEL_UNARY_H_
