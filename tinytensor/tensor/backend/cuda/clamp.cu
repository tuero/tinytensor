// clamp.cu
// Element-wise clamp runner

#include <tt/concepts.h>
#include <tt/scalar.h>
#include <tt/tensor.h>

#include "tensor/backend/cuda/clamp.h"
#include "tensor/backend/cuda/config.cuh"
#include "tensor/backend/cuda/data_types.cuh"
#include "tensor/backend/cuda/kernel/clamp.cuh"
#include "tensor/backend/cuda/kernel_launch.cuh"
#include "tensor/backend/cuda/storage_cuda.h"

#include <cassert>
#include <type_traits>
#include <variant>

namespace tinytensor::cuda {

using namespace kernel::clamp;

void clamp_inplace_runner(Tensor &tensor, const Tensor &min, const Tensor &max) {
    assert(tensor.device() == min.device() && tensor.device() == max.device());
    const int N = tensor.numel();
    const int device_id = tensor.device().id;

    // Create device memory for shape + stride for proper indexing
    const auto tensor_shape = MakeDeviceMemory(device_id, tensor.shape());
    const auto tensor_stride = MakeDeviceMemory(device_id, tensor.stride());
    const auto min_shape = MakeDeviceMemory(device_id, min.shape());
    const auto min_stride = MakeDeviceMemory(device_id, min.stride());
    const auto max_shape = MakeDeviceMemory(device_id, max.shape());
    const auto max_stride = MakeDeviceMemory(device_id, max.stride());

    // lhs and rhs need to be same type, so visit on one to reduce codegen
    return std::visit(
        [&](auto &&tensor_dev_memory) {
            using DT = std::remove_cvref_t<decltype(tensor_dev_memory)>;    // Storage type
            using T = template_parameter_t<DT>;                             // Underlying type

            // Get operand data spans
            DeviceSpan<T> tensor_span{tensor_dev_memory};
            const DeviceSpan<const T> min_span{std::get<DT>(min.template get_storage<StorageCUDA>().dev_memory)};
            const DeviceSpan<const T> max_span{std::get<DT>(max.template get_storage<StorageCUDA>().dev_memory)};

            // Set operands to kernel
            DataInfo<T> a{tensor_span, tensor_shape, tensor_stride, tensor.offset()};
            const DataInfo<const T> _min{min_span, min_shape, min_stride, min.offset()};
            const DataInfo<const T> _max{max_span, max_shape, max_stride, max.offset()};

            const auto kernel = clamp_kernel<T>;
            launch(device_id, kernel, grid_1d(N), block_1d(), a, _min, _max, N);
        },
        tensor.template get_storage<StorageCUDA>().dev_memory
    );
}

}    // namespace tinytensor::cuda
