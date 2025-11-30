// assign.cu
// Assign runner

#include <tt/concepts.h>
#include <tt/scalar.h>
#include <tt/tensor.h>

#include "tensor/backend/cuda/assign.h"
#include "tensor/backend/cuda/config.cuh"
#include "tensor/backend/cuda/data_types.cuh"
#include "tensor/backend/cuda/kernel/assign.cuh"
#include "tensor/backend/cuda/kernel_launch.cuh"
#include "tensor/backend/cuda/storage_cuda.h"

#include <cassert>
#include <type_traits>
#include <variant>

namespace tinytensor::cuda {

using namespace kernel::assign;

void assign_runner(Tensor &lhs, const Tensor &rhs) {
    assert(lhs.device() == rhs.device());
    const int N = lhs.numel();
    const int device_id = lhs.device().id;

    // Create device memory for shape + stride for proper indexing
    const auto lhs_shape = MakeDeviceMemory(device_id, lhs.shape());
    const auto lhs_stride = MakeDeviceMemory(device_id, lhs.stride());
    const auto rhs_shape = MakeDeviceMemory(device_id, rhs.shape());
    const auto rhs_stride = MakeDeviceMemory(device_id, rhs.stride());

    // lhs and rhs need to be same type, so visit on one to reduce codegen
    return std::visit(
        [&](auto &&tensor_dev_memory) {
            using DT = std::remove_cvref_t<decltype(tensor_dev_memory)>;    // Storage type
            using T = template_parameter_t<DT>;                             // Underlying type

            // Get operand data spans
            DeviceSpan<T> lhs_span{tensor_dev_memory};
            const DeviceSpan<const T> rhs_span{std::get<DT>(rhs.template get_storage<StorageCUDA>().dev_memory)};

            // Set operands to kernel
            DataInfo<T> l{lhs_span, lhs_shape, lhs_stride, lhs.offset()};
            const DataInfo<const T> r{rhs_span, rhs_shape, rhs_stride, rhs.offset()};

            launch(device_id, assign_kernel<T>, grid_1d(N), block_1d(), l, r, N);
        },
        lhs.template get_storage<StorageCUDA>().dev_memory
    );
}

}    // namespace tinytensor::cuda
