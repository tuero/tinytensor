// misc.cu
// Element-wise misc runner

#include <tt/concepts.h>
#include <tt/scalar.h>
#include <tt/tensor.h>

#include "tensor/backend/cuda/config.cuh"
#include "tensor/backend/cuda/data_types.cuh"
#include "tensor/backend/cuda/kernel/misc.cuh"
#include "tensor/backend/cuda/kernel_launch.cuh"
#include "tensor/backend/cuda/misc.h"
#include "tensor/backend/cuda/storage_cuda.h"

#include <cassert>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <variant>

namespace tinytensor::cuda {

using namespace tinytensor::common::misc;
using namespace kernel::misc;

auto where_runner(const Tensor &cond, const Tensor &lhs, const Tensor &rhs) -> Tensor {
    assert(lhs.device() == rhs.device() && lhs.device() == cond.device());
    const auto res_shape = lhs.shape();
    const auto res_device = lhs.device();
    const int N = lhs.numel();

    // Create device memory for shape + stride for proper indexing
    const auto cond_shape = MakeDeviceMemory(cond.shape());
    const auto cond_stride = MakeDeviceMemory(cond.stride());
    const auto lhs_shape = MakeDeviceMemory(lhs.shape());
    const auto lhs_stride = MakeDeviceMemory(lhs.stride());
    const auto rhs_shape = MakeDeviceMemory(rhs.shape());
    const auto rhs_stride = MakeDeviceMemory(rhs.stride());

    // lhs and rhs need to be same type, so visit on one to reduce codegen
    return std::visit(
        [&](auto &&tensor_dev_memory) -> Tensor {
            using DT = std::remove_cvref_t<decltype(tensor_dev_memory)>;    // Storage type
            using T = template_parameter_t<DT>;                             // Underlying type
            using U = to_ctype_t<kBool>;
            using DU = DeviceMemory<U>;

            // Allocate for result
            auto res_dev_memory = DeviceMemory<T>::AllocateElements(static_cast<std::size_t>(N));

            // Get operand data spans
            const DeviceSpan<const T> lhs_span{tensor_dev_memory};
            const DeviceSpan<const T> rhs_span{std::get<DT>(rhs.template get_storage<StorageCUDA>().dev_memory)};
            const DeviceSpan<const U> cond_span{std::get<DU>(cond.template get_storage<StorageCUDA>().dev_memory)};

            // Set operands to kernel
            const DataInfo<const T> l{lhs_span, lhs_shape, lhs_stride, lhs.offset()};
            const DataInfo<const T> r{rhs_span, rhs_shape, rhs_stride, rhs.offset()};
            const DataInfo<const U> c{cond_span, cond_shape, cond_stride, cond.offset()};

            const auto kernel = where_kernel<T>;
            launch(kernel, grid_1d(N), block_1d(), DeviceSpan<T>{res_dev_memory}, N, c, l, r);
            return {std::make_unique<StorageCUDA>(std::move(res_dev_memory)), lhs.dtype(), res_shape, res_device};
        },
        lhs.template get_storage<StorageCUDA>().dev_memory
    );
}

auto gather_runner(const Tensor &input, const Tensor &indices, int dim) -> Tensor {
    assert(input.device() == indices.device());
    auto res_shape = input.shape();
    res_shape[dim] = 1;
    const auto res_device = input.device();
    const int N = res_shape.numel();

    // Create device memory for shape + stride for proper indexing
    const auto input_shape = MakeDeviceMemory(input.shape());
    const auto input_stride = MakeDeviceMemory(input.stride());
    const auto idx_shape = MakeDeviceMemory(indices.shape());
    const auto idx_stride = MakeDeviceMemory(indices.stride());
    const auto res_dev_shape = MakeDeviceMemory(res_shape);

    // lhs and rhs need to be same type, so visit on one to reduce codegen
    return std::visit(
        [&](auto &&tensor_dev_memory) -> Tensor {
            using DT = std::remove_cvref_t<decltype(tensor_dev_memory)>;    // Storage type
            using T = template_parameter_t<DT>;                             // Underlying type
            using U = to_ctype_t<kDefaultInt>;
            using DU = DeviceMemory<U>;

            // Allocate for result
            auto res_dev_memory = DeviceMemory<T>::AllocateElements(static_cast<std::size_t>(N));

            // Get operand data spans
            const DeviceSpan<const T> input_span{tensor_dev_memory};
            const DeviceSpan<const U> idx_span{std::get<DU>(indices.template get_storage<StorageCUDA>().dev_memory)};
            const DeviceSpan<const U> res_shape_span{res_dev_shape};

            // Set operands to kernel
            const DataInfo<const T> input_info{input_span, input_shape, input_stride, input.offset()};
            const DataInfo<const U> idx_info{idx_span, idx_shape, idx_stride, indices.offset()};

            const auto kernel = gather_kernel<T>;
            launch(
                kernel,
                grid_1d(N),
                block_1d(),
                DeviceSpan<T>{res_dev_memory},
                N,
                input_info,
                idx_info,
                res_shape_span,
                dim
            );
            return {std::make_unique<StorageCUDA>(std::move(res_dev_memory)), input.dtype(), res_shape, res_device};
        },
        input.template get_storage<StorageCUDA>().dev_memory
    );
}

}    // namespace tinytensor::cuda
