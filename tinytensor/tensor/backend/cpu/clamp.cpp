// clamp.cpp
// Element-wise clamp runner

#include "tensor/backend/cpu/clamp.h"

#include <tt/concepts.h>
#include <tt/scalar.h>
#include <tt/tensor.h>

#include "tensor/backend/common/span.h"
#include "tensor/backend/cpu/data_types.h"
#include "tensor/backend/cpu/kernel/clamp.hpp"
#include "tensor/backend/cpu/storage_cpu.h"

#include <type_traits>
#include <variant>

namespace tinytensor::cpu {

using namespace kernel::clamp;

void clamp_inplace_runner(Tensor &tensor, const Tensor &min, const Tensor &max) {
    const int N = tensor.numel();
    return std::visit(
        [&](auto &&tensor_storage) {
            using DT = std::remove_cvref_t<decltype(tensor_storage)>;    // Storage type
            using T = template_parameter_t<DT>;                          // Underlying type

            // Get operand data spans
            HostSpan<T> tensor_span{tensor_storage};
            const HostSpan<const T> min_span{std::get<DT>(min.template get_storage<StorageCPU>().storage)};
            const HostSpan<const T> max_span{std::get<DT>(max.template get_storage<StorageCPU>().storage)};

            // Set operands to kernel
            DataInfo<T> a{tensor_span, tensor.shape(), tensor.stride(), tensor.offset()};
            const DataInfo<const T> _min{min_span, min.shape(), min.stride(), min.offset()};
            const DataInfo<const T> _max{max_span, max.shape(), max.stride(), max.offset()};

            // Call kernel
            clamp_kernel(a, _min, _max, N);
        },
        tensor.template get_storage<StorageCPU>().storage
    );
}

}    // namespace tinytensor::cpu
