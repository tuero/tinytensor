// assign.cpp
// Assign runner

#include "tensor/backend/cpu/assign.h"

#include <tt/concepts.h>
#include <tt/scalar.h>
#include <tt/tensor.h>

#include "tensor/backend/common/span.h"
#include "tensor/backend/cpu/data_types.h"
#include "tensor/backend/cpu/kernel/assign.hpp"
#include "tensor/backend/cpu/storage_cpu.h"

#include <type_traits>
#include <variant>

namespace tinytensor::cpu {

using namespace kernel::assign;

void assign_runner(Tensor &lhs, const Tensor &rhs) {
    const int N = lhs.numel();
    return std::visit(
        [&](auto &&array_storage) {
            using DT = std::remove_cvref_t<decltype(array_storage)>;    // Storage type
            using T = template_parameter_t<DT>;                         // Underlying type

            // Get operand data spans
            HostSpan<T> lhs_span{array_storage};
            const HostSpan<const T> rhs_span{std::get<DT>(rhs.template get_storage<StorageCPU>().storage)};

            // Set operands to kernel
            DataInfo<T> l{lhs_span, lhs.shape(), lhs.stride(), lhs.offset()};
            const DataInfo<const T> r{rhs_span, rhs.shape(), rhs.stride(), rhs.offset()};

            // Call kernel
            assign_kernel(l, r, N);
        },
        lhs.template get_storage<StorageCPU>().storage
    );
}

}    // namespace tinytensor::cpu
