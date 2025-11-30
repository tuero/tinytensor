// cast.cu
// Cast from one type to another

#include <tt/concepts.h>
#include <tt/exception.h>
#include <tt/scalar.h>
#include <tt/tensor.h>

#include "tensor/backend/common/kernel/unary.hpp"
#include "tensor/backend/cuda/config.cuh"
#include "tensor/backend/cuda/data_types.cuh"
#include "tensor/backend/cuda/kernel/unary.cuh"
#include "tensor/backend/cuda/kernel_launch.cuh"
#include "tensor/backend/cuda/storage_cuda.h"

#include <cstddef>
#include <memory>
#include <type_traits>
#include <variant>

namespace tinytensor::cuda {

using namespace common::kernel::unary;

template <typename T, typename R, typename KernelOp>
auto _cast_runner(int device_id, const DataInfo<const T> &tensor_info, int N) -> std::unique_ptr<StorageCUDA> {
    auto res_dev_memory = DeviceMemory<R>::AllocateElements(device_id, static_cast<std::size_t>(N));

    const auto kernel = kernel::unary::unary_kernel<true, T, R, KernelOp>;
    launch(device_id, kernel, grid_1d(N), block_1d(), tensor_info, DeviceSpan<R>{res_dev_memory}, KernelOp{}, N);
    return std::make_unique<StorageCUDA>(std::move(res_dev_memory));
}

auto cast_runner(const Tensor &tensor, ScalarType dtype) -> Tensor {
    const auto res_shape = tensor.shape();
    const auto res_device = tensor.device();
    const int N = tensor.numel();
    const int device_id = tensor.device().id;

    // Create device memory for shape + stride for proper indexing
    const auto shape = MakeDeviceMemory(device_id, tensor.shape());
    const auto stride = MakeDeviceMemory(device_id, tensor.stride());

    return std::visit(
        [&](auto &&tensor_dev_memory) -> Tensor {
            using DT = std::remove_cvref_t<decltype(tensor_dev_memory)>;    // Storage type
            using T = template_parameter_t<DT>;                             // Underlying type

            const DeviceSpan<const T> a{tensor_dev_memory};
            const DataInfo<const T> tensor_info{a, shape, stride, tensor.offset()};

            auto dtype_switch = [&](ScalarType cast_to_type) {
                switch (cast_to_type) {
                case kBool: {
                    using R = to_ctype_t<kBool>;
                    using KernOp = OpIdentity<R>;
                    return _cast_runner<T, R, KernOp>(device_id, tensor_info, N);
                }
                case kU8: {
                    using R = to_ctype_t<kU8>;
                    using KernOp = OpIdentity<R>;
                    return _cast_runner<T, R, KernOp>(device_id, tensor_info, N);
                }
                case kI16: {
                    using R = to_ctype_t<kI16>;
                    using KernOp = OpIdentity<R>;
                    return _cast_runner<T, R, KernOp>(device_id, tensor_info, N);
                }
                case kI32: {
                    using R = to_ctype_t<kI32>;
                    using KernOp = OpIdentity<R>;
                    return _cast_runner<T, R, KernOp>(device_id, tensor_info, N);
                }
                case kI64: {
                    using R = to_ctype_t<kI64>;
                    using KernOp = OpIdentity<R>;
                    return _cast_runner<T, R, KernOp>(device_id, tensor_info, N);
                }
                case kF32: {
                    using R = to_ctype_t<kF32>;
                    using KernOp = OpIdentity<R>;
                    return _cast_runner<T, R, KernOp>(device_id, tensor_info, N);
                }
                case kF64: {
                    using R = to_ctype_t<kF64>;
                    using KernOp = OpIdentity<R>;
                    return _cast_runner<T, R, KernOp>(device_id, tensor_info, N);
                }
                }
                TT_ERROR("Unknown dtype.");
            };

            return {dtype_switch(dtype), dtype, res_shape, res_device};
        },
        tensor.get_storage<StorageCUDA>().dev_memory
    );
}

}    // namespace tinytensor::cuda
