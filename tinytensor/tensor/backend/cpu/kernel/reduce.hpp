// unary.hpp
// Reduction kernel

#ifndef TINYTENSOR_BACKEND_CPU_KERNEL_REDUCE_H_
#define TINYTENSOR_BACKEND_CPU_KERNEL_REDUCE_H_

#include "tensor/backend/common/span.h"
#include "tensor/backend/common/util.h"
#include "tensor/backend/cpu/data_types.h"

#include <cassert>

namespace tinytensor::cpu::kernel::reduce {

template <typename T, typename R, typename OP>
void reduce_dim_kernel(const DataInfo<const T> di, HostSpan<R> res, OP op, int dim, int N) {
    using VT = OP::VT;
    using VIT = OP::VIT;
    for (int i = 0; i < N; ++i) {
        auto out_val = OP::padding_value;
        int out_idx = -1;
        const auto start_idx = to_flat_index(i, di.shape, di.stride, di.offset, dim);
        for (int j = 0; j < di.shape[dim]; ++j) {
            const auto idx = to_flat_index(start_idx + j * di.stride[dim], di.shape, di.stride, di.offset);
            const VIT out_val_idx{out_val, out_idx};
            const VIT curr_val_idx{static_cast<VT>(di.data[idx]), j};
            out_idx = op.index()(curr_val_idx, out_val_idx);
            out_val = op.value()(curr_val_idx, out_val_idx);
        }
        res[i] = static_cast<R>(op.res()({out_val, out_idx}));
    }
}

template <typename T, typename R, typename OP>
void reduce_all_kernel(const DataInfo<const T> di, HostSpan<R> res, OP op, int N) {
    using VT = OP::VT;
    using VIT = OP::VIT;
    auto out_val = OP::padding_value;
    int out_idx = -1;
    for (int i = 0; i < N; ++i) {
        const auto idx = to_flat_index(i, di.shape, di.stride, di.offset);
        const VIT out_val_idx{out_val, out_idx};
        const VIT curr_val_idx{static_cast<VT>(di.data[idx]), i};
        out_idx = op.index()(curr_val_idx, out_val_idx);
        out_val = op.value()(curr_val_idx, out_val_idx);
    }
    res[0] = static_cast<R>(op.res()({out_val, out_idx}));
}

}    // namespace tinytensor::cpu::kernel::reduce

#endif    // TINYTENSOR_BACKEND_CPU_REDUCE_H_
