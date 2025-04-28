// misc.cpp
// Element-wise misc runner

#include "tensor/backend/cpu/misc.h"

#include <tt/concepts.h>
#include <tt/exception.h>
#include <tt/scalar.h>
#include <tt/tensor.h>

#include "tensor/backend/common/span.h"
#include "tensor/backend/cpu/data_types.h"
#include "tensor/backend/cpu/kernel/misc.hpp"
#include "tensor/backend/cpu/storage_cpu.h"

#include <cassert>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <variant>
#include <vector>

namespace tinytensor::cpu {

using namespace tinytensor::common::misc;
using namespace kernel::misc;

auto where_runner(const Tensor &cond, const Tensor &lhs, const Tensor &rhs) -> Tensor {
    assert(lhs.device() == rhs.device() && lhs.device() == cond.device());
    const int N = cond.numel();
    return std::visit(
        [&](auto &&array_storage) -> Tensor {
            using DT = std::remove_cvref_t<decltype(array_storage)>;    // Storage type
            using T = template_parameter_t<DT>;                         // Underlying type
            using U = to_ctype_t<kBool>;
            using DU = std::vector<U>;

            // Allocate for result
            std::vector<T> result(static_cast<std::size_t>(N));

            // Get operand data spans
            const HostSpan<const T> lhs_span{array_storage};
            const HostSpan<const T> rhs_span{std::get<DT>(rhs.template get_storage<StorageCPU>().storage)};
            const HostSpan<const U> cond_span{std::get<DU>(cond.template get_storage<StorageCPU>().storage)};

            const DataInfo<const T> l{lhs_span, lhs.shape(), lhs.stride(), lhs.offset()};
            const DataInfo<const T> r{rhs_span, rhs.shape(), rhs.stride(), rhs.offset()};
            const DataInfo<const U> c{cond_span, cond.shape(), cond.stride(), cond.offset()};

            // Call kernel
            where_kernel(HostSpan<T>(result), N, c, l, r);
            return {std::make_unique<StorageCPU>(std::move(result)), lhs.dtype(), lhs.shape(), lhs.device()};
        },
        lhs.template get_storage<StorageCPU>().storage
    );
}

auto gather_runner(const Tensor &input, const Tensor &indices, int dim) -> Tensor {
    assert(input.device() == indices.device());
    auto res_shape = input.shape();
    res_shape[dim] = 1;
    const int N = res_shape.numel();
    return std::visit(
        [&](auto &&array_storage) -> Tensor {
            using DT = std::remove_cvref_t<decltype(array_storage)>;    // Storage type
            using T = template_parameter_t<DT>;                         // Underlying type
            using U = to_ctype_t<kDefaultInt>;
            using DU = std::vector<U>;

            // Allocate for result
            std::vector<T> result(static_cast<std::size_t>(N));

            // Get operand data spans
            const HostSpan<const T> input_span{array_storage};
            const HostSpan<const U> idx_span{std::get<DU>(indices.template get_storage<StorageCPU>().storage)};
            const HostSpan<const U> res_shape_span{res_shape};

            const DataInfo<const T> input_info{input_span, input.shape(), input.stride(), input.offset()};
            const DataInfo<const U> idx_info{idx_span, indices.shape(), indices.stride(), indices.offset()};

            // Call kernel
            gather_kernel(HostSpan<T>(result), N, input_info, idx_info, res_shape_span, dim);
            return {std::make_unique<StorageCPU>(std::move(result)), input.dtype(), res_shape, input.device()};
        },
        input.template get_storage<StorageCPU>().storage
    );
}

}    // namespace tinytensor::cpu
