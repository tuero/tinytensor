// binary.cu
// Element-wise binary runner

#include <tt/concepts.h>
#include <tt/scalar.h>
#include <tt/tensor.h>

#include "tensor/backend/common/binary.h"
#include "tensor/backend/cuda/binary.h"
#include "tensor/backend/cuda/config.cuh"
#include "tensor/backend/cuda/data_types.cuh"
#include "tensor/backend/cuda/kernel/binary.cuh"
#include "tensor/backend/cuda/kernel_launch.cuh"
#include "tensor/backend/cuda/storage_cuda.h"

#include <cassert>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <variant>

namespace tinytensor::cuda {

using namespace tinytensor::common::binary;
using namespace kernel::binary;

template <BinaryOpT Op>
auto binary_runner(const Tensor &lhs, const Tensor &rhs) -> Tensor {
    assert(lhs.device() == rhs.device());
    const int device_id = lhs.device().id;
    const auto res_shape = lhs.shape();
    const auto res_stride = MakeDeviceMemory(device_id, res_shape.to_stride());
    const auto res_device = lhs.device();
    const int N = lhs.numel();

    // Create device memory for shape + stride for proper indexing
    const auto lhs_shape = MakeDeviceMemory(device_id, lhs.shape());
    const auto lhs_stride = MakeDeviceMemory(device_id, lhs.stride());
    const auto rhs_shape = MakeDeviceMemory(device_id, rhs.shape());
    const auto rhs_stride = MakeDeviceMemory(device_id, rhs.stride());

    // lhs and rhs need to be same type, so visit on one to reduce codegen
    return std::visit(
        [&](auto &&tensor_dev_memory) -> Tensor {
            using DT = std::remove_cvref_t<decltype(tensor_dev_memory)>;    // Storage type
            using T = template_parameter_t<DT>;                             // Underlying type
            using R = Result<T, Op>::type;                                  // Result type
            // Type to kernel is R if casting before, T if casting after
            constexpr bool CastBeforeOp = Result<R, Op>::CastBeforeOp;
            using KernelOp = typename OpFactory<std::conditional_t<CastBeforeOp, R, T>, Op>::KernelOp;

            // Allocate for result
            auto res_dev_memory = DeviceMemory<R>::AllocateElements(device_id, static_cast<std::size_t>(N));

            // Get operand data spans
            const DeviceSpan<const T> lhs_span{tensor_dev_memory};
            const DeviceSpan<const T> rhs_span{std::get<DT>(rhs.template get_storage<StorageCUDA>().dev_memory)};

            // Set operands to kernel
            const DataInfo<const T> l{lhs_span, lhs_shape, lhs_stride, lhs.offset()};
            const DataInfo<const T> r{rhs_span, rhs_shape, rhs_stride, rhs.offset()};

            const auto kernel = binary_kernel<CastBeforeOp, T, R, KernelOp>;
            launch(device_id, kernel, grid_1d(N), block_1d(), l, r, DeviceSpan<R>{res_dev_memory}, KernelOp{}, N);
            return {
                std::make_unique<StorageCUDA>(std::move(res_dev_memory)),
                Result<T, Op>::scalar(lhs.dtype()),
                res_shape,
                res_device
            };
        },
        lhs.template get_storage<StorageCUDA>().dev_memory
    );
}

template <BinaryOpT Op>
void binary_inplace_runner(Tensor &lhs, const Tensor &rhs) {
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
            using KernelOp = typename OpFactory<T, Op>::KernelOp;

            // Get operand data spans
            DeviceSpan<T> lhs_span{tensor_dev_memory};
            const DeviceSpan<const T> rhs_span{std::get<DT>(rhs.template get_storage<StorageCUDA>().dev_memory)};

            // Set operands to kernel
            DataInfo<T> l{lhs_span, lhs_shape, lhs_stride, lhs.offset()};
            const DataInfo<const T> r{rhs_span, rhs_shape, rhs_stride, rhs.offset()};

            const auto kernel = binary_kernel<T, KernelOp>;
            launch(device_id, kernel, grid_1d(N), block_1d(), l, r, KernelOp{}, N);
        },
        lhs.template get_storage<StorageCUDA>().dev_memory
    );
}

template Tensor binary_runner<BinaryOpT::add>(const Tensor &lhs, const Tensor &rhs);
template Tensor binary_runner<BinaryOpT::subtract>(const Tensor &lhs, const Tensor &rhs);
template Tensor binary_runner<BinaryOpT::multiply>(const Tensor &lhs, const Tensor &rhs);
template Tensor binary_runner<BinaryOpT::divide>(const Tensor &lhs, const Tensor &rhs);
template Tensor binary_runner<BinaryOpT::equal>(const Tensor &lhs, const Tensor &rhs);
template Tensor binary_runner<BinaryOpT::not_equal>(const Tensor &lhs, const Tensor &rhs);
template Tensor binary_runner<BinaryOpT::less_than>(const Tensor &lhs, const Tensor &rhs);
template Tensor binary_runner<BinaryOpT::less_than_eq>(const Tensor &lhs, const Tensor &rhs);
template Tensor binary_runner<BinaryOpT::greater_than>(const Tensor &lhs, const Tensor &rhs);
template Tensor binary_runner<BinaryOpT::greater_than_eq>(const Tensor &lhs, const Tensor &rhs);
template Tensor binary_runner<BinaryOpT::logical_or>(const Tensor &lhs, const Tensor &rhs);
template Tensor binary_runner<BinaryOpT::logical_and>(const Tensor &lhs, const Tensor &rhs);
template Tensor binary_runner<BinaryOpT::bitwise_or>(const Tensor &lhs, const Tensor &rhs);
template Tensor binary_runner<BinaryOpT::bitwise_and>(const Tensor &lhs, const Tensor &rhs);
template Tensor binary_runner<BinaryOpT::bitwise_xor>(const Tensor &lhs, const Tensor &rhs);
template Tensor binary_runner<BinaryOpT::bitwise_left_shift>(const Tensor &lhs, const Tensor &rhs);
template Tensor binary_runner<BinaryOpT::bitwise_right_shift>(const Tensor &lhs, const Tensor &rhs);
template Tensor binary_runner<BinaryOpT::modulo>(const Tensor &lhs, const Tensor &rhs);
template Tensor binary_runner<BinaryOpT::minimum>(const Tensor &lhs, const Tensor &rhs);
template Tensor binary_runner<BinaryOpT::maximum>(const Tensor &lhs, const Tensor &rhs);
template Tensor binary_runner<BinaryOpT::pow>(const Tensor &lhs, const Tensor &rhs);

template void binary_inplace_runner<BinaryOpT::add>(Tensor &lhs, const Tensor &rhs);
template void binary_inplace_runner<BinaryOpT::subtract>(Tensor &lhs, const Tensor &rhs);
template void binary_inplace_runner<BinaryOpT::multiply>(Tensor &lhs, const Tensor &rhs);
template void binary_inplace_runner<BinaryOpT::divide>(Tensor &lhs, const Tensor &rhs);

}    // namespace tinytensor::cuda
