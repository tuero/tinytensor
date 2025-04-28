// distribution.hpp
// Element-wise distribution kernel

#ifndef TINYTENSOR_BACKEND_CPU_KERNEL_DISTRIBUTION_H_
#define TINYTENSOR_BACKEND_CPU_KERNEL_DISTRIBUTION_H_

#include <tt/concepts.h>
#include <tt/random.h>
#include <tt/scalar.h>

#include "tensor/backend/common/span.h"
#include "tensor/backend/common/util.h"
#include "tensor/backend/cpu/data_types.h"

#include <cstdint>

namespace tinytensor::cpu::kernel::distribution {

template <typename T, typename OP, typename... TS>
    requires IsAllOf<T, TS...>
void variadic_param_kernel(
    const HostSpan<const uint64_t> gen_states,
    DataInfo<T> res,
    OP op,
    int N,
    const HostSpan<const TS>... params
) {
    for (int i = 0; i < N; ++i) {
        Generator gen = Generator::from_state(gen_states[i]);
        const auto idx_res = to_flat_index(i, res.shape, res.stride, res.offset);
        res.data[idx_res] = op()(params[i]..., gen);
    }
}

}    // namespace tinytensor::cpu::kernel::distribution

#endif    // TINYTENSOR_BACKEND_CPU_KERNEL_DISTRIBUTION_H_
