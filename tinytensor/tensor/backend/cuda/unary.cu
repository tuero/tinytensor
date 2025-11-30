// unary.cu
// Element-wise unary runner

#include <tt/concepts.h>
#include <tt/scalar.h>
#include <tt/tensor.h>

#include "tensor/backend/common/unary.h"
#include "tensor/backend/cuda/config.cuh"
#include "tensor/backend/cuda/data_types.cuh"
#include "tensor/backend/cuda/kernel/unary.cuh"
#include "tensor/backend/cuda/kernel_launch.cuh"
#include "tensor/backend/cuda/storage_cuda.h"
#include "tensor/backend/cuda/unary.h"

#include <cstddef>
#include <type_traits>
#include <variant>

namespace tinytensor::cuda {

using namespace tinytensor::common::unary;
using namespace kernel::unary;

template <UnaryOpT Op, typename... Params>
auto unary_runner(const Tensor &tensor, Params... params) -> Tensor {
    // Create device memory for shape + stride for proper indexing
    const int N = tensor.numel();
    const int device_id = tensor.device().id;
    const auto shape = MakeDeviceMemory(device_id, tensor.shape());
    const auto stride = MakeDeviceMemory(device_id, tensor.stride());
    return std::visit(
        [&](auto &&dev_memory) -> Tensor {
            using DT = std::remove_cvref_t<decltype(dev_memory)>;    // Storage type
            using T = template_parameter_t<DT>;                      // Underlying type
            using R = Result<T, Op>::type;                           // Result type
            // Type to kernel is R if casting before, T if casting after
            constexpr bool CastBeforeOp = Result<R, Op>::CastBeforeOp;
            using KernelOp = typename OpFactory<std::conditional_t<CastBeforeOp, R, T>, Op>::KernelOp;
            const DataInfo<const T> tensor_info{dev_memory, shape, stride, tensor.offset()};

            // Allocate for result
            auto res_dev_memory = DeviceMemory<R>::AllocateElements(device_id, static_cast<std::size_t>(N));

            const auto kernel = unary_kernel<CastBeforeOp, T, R, KernelOp, Params...>;
            launch(
                device_id,
                kernel,
                grid_1d(N),
                block_1d(),
                tensor_info,
                DeviceSpan<R>{res_dev_memory},
                KernelOp{},
                N,
                static_cast<R>(params)...
            );
            return {
                std::make_unique<StorageCUDA>(std::move(res_dev_memory)),
                Result<T, Op>::scalar(tensor.dtype()),
                tensor.shape(),
                tensor.device()
            };
        },
        tensor.get_storage<StorageCUDA>().dev_memory
    );
}

template <UnaryOpT Op, typename... Params>
void unary_runner_inplace(Tensor &tensor, Params... params) {
    // Create device memory for shape + stride for proper indexing
    const int N = tensor.numel();
    const int device_id = tensor.device().id;
    const auto shape = MakeDeviceMemory(device_id, tensor.shape());
    const auto stride = MakeDeviceMemory(device_id, tensor.stride());
    std::visit(
        [&](auto &&dev_memory) {
            using DT = std::remove_cvref_t<decltype(dev_memory)>;    // Storage type
            using T = template_parameter_t<DT>;                      // Underlying type

            // If result type changes, exception gets thrown at the call site outside backend
            // implementation we thus don't need to instantiate for these types
            using R = Result<T, Op>::type;    // Result type
            if constexpr (std::is_same_v<T, R>) {
                using KernelOp = typename OpFactory<R, Op>::KernelOp;
                const DataInfo<T> tensor_info{dev_memory, shape, stride, tensor.offset()};

                const auto kernel = unary_kernel_inplace<T, KernelOp, Params...>;
                launch(
                    device_id,
                    kernel,
                    grid_1d(N),
                    block_1d(),
                    tensor_info,
                    KernelOp{},
                    N,
                    static_cast<T>(params)...
                );
            }
        },
        tensor.get_storage<StorageCUDA>().dev_memory
    );
}

template Tensor unary_runner<UnaryOpT::identity>(const Tensor &);
template Tensor unary_runner<UnaryOpT::negate>(const Tensor &);
template Tensor unary_runner<UnaryOpT::logical_not>(const Tensor &);
template Tensor unary_runner<UnaryOpT::abs>(const Tensor &);
template Tensor unary_runner<UnaryOpT::sign>(const Tensor &);
template Tensor unary_runner<UnaryOpT::log>(const Tensor &);
template Tensor unary_runner<UnaryOpT::log10>(const Tensor &);
template Tensor unary_runner<UnaryOpT::log2>(const Tensor &);
template Tensor unary_runner<UnaryOpT::log1p>(const Tensor &);
template Tensor unary_runner<UnaryOpT::exp>(const Tensor &);
template Tensor unary_runner<UnaryOpT::exp2>(const Tensor &);
template Tensor unary_runner<UnaryOpT::expm1>(const Tensor &);
template Tensor unary_runner<UnaryOpT::sqrt>(const Tensor &);
template Tensor unary_runner<UnaryOpT::sin>(const Tensor &);
template Tensor unary_runner<UnaryOpT::cos>(const Tensor &);
template Tensor unary_runner<UnaryOpT::tan>(const Tensor &);
template Tensor unary_runner<UnaryOpT::asin>(const Tensor &);
template Tensor unary_runner<UnaryOpT::acos>(const Tensor &);
template Tensor unary_runner<UnaryOpT::atan>(const Tensor &);
template Tensor unary_runner<UnaryOpT::sinh>(const Tensor &);
template Tensor unary_runner<UnaryOpT::cosh>(const Tensor &);
template Tensor unary_runner<UnaryOpT::tanh>(const Tensor &);
template Tensor unary_runner<UnaryOpT::asinh>(const Tensor &);
template Tensor unary_runner<UnaryOpT::acosh>(const Tensor &);
template Tensor unary_runner<UnaryOpT::atanh>(const Tensor &);
template Tensor unary_runner<UnaryOpT::erf>(const Tensor &);
template Tensor unary_runner<UnaryOpT::erfc>(const Tensor &);
template Tensor unary_runner<UnaryOpT::tgamma>(const Tensor &);
template Tensor unary_runner<UnaryOpT::lgamma>(const Tensor &);
template Tensor unary_runner<UnaryOpT::digamma>(const Tensor &);
template Tensor unary_runner<UnaryOpT::ceil>(const Tensor &);
template Tensor unary_runner<UnaryOpT::floor>(const Tensor &);
template Tensor unary_runner<UnaryOpT::round>(const Tensor &);
template Tensor unary_runner<UnaryOpT::isinf>(const Tensor &);
template Tensor unary_runner<UnaryOpT::isnan>(const Tensor &);
template Tensor unary_runner<UnaryOpT::isfinite>(const Tensor &);
template Tensor unary_runner<UnaryOpT::sigmoid>(const Tensor &);
template Tensor unary_runner<UnaryOpT::softplus>(const Tensor &, double, double);
template Tensor unary_runner<UnaryOpT::relu>(const Tensor &);
template Tensor unary_runner<UnaryOpT::relu6>(const Tensor &);
template Tensor unary_runner<UnaryOpT::elu>(const Tensor &, double);
template Tensor unary_runner<UnaryOpT::selu>(const Tensor &);
template Tensor unary_runner<UnaryOpT::silu>(const Tensor &);
template Tensor unary_runner<UnaryOpT::hardsigmoid>(const Tensor &);
template Tensor unary_runner<UnaryOpT::hardtanh>(const Tensor &, double, double);
template Tensor unary_runner<UnaryOpT::leaky_relu>(const Tensor &, double);
template Tensor unary_runner<UnaryOpT::log_sigmoid>(const Tensor &);
template Tensor unary_runner<UnaryOpT::softsign>(const Tensor &);

template void unary_runner_inplace<UnaryOpT::identity>(Tensor &);
template void unary_runner_inplace<UnaryOpT::negate>(Tensor &);
template void unary_runner_inplace<UnaryOpT::logical_not>(Tensor &);
template void unary_runner_inplace<UnaryOpT::abs>(Tensor &);
template void unary_runner_inplace<UnaryOpT::sign>(Tensor &);
template void unary_runner_inplace<UnaryOpT::log>(Tensor &);
template void unary_runner_inplace<UnaryOpT::log10>(Tensor &);
template void unary_runner_inplace<UnaryOpT::log2>(Tensor &);
template void unary_runner_inplace<UnaryOpT::log1p>(Tensor &);
template void unary_runner_inplace<UnaryOpT::exp>(Tensor &);
template void unary_runner_inplace<UnaryOpT::exp2>(Tensor &);
template void unary_runner_inplace<UnaryOpT::expm1>(Tensor &);
template void unary_runner_inplace<UnaryOpT::sqrt>(Tensor &);
template void unary_runner_inplace<UnaryOpT::sin>(Tensor &);
template void unary_runner_inplace<UnaryOpT::cos>(Tensor &);
template void unary_runner_inplace<UnaryOpT::tan>(Tensor &);
template void unary_runner_inplace<UnaryOpT::asin>(Tensor &);
template void unary_runner_inplace<UnaryOpT::acos>(Tensor &);
template void unary_runner_inplace<UnaryOpT::atan>(Tensor &);
template void unary_runner_inplace<UnaryOpT::sinh>(Tensor &);
template void unary_runner_inplace<UnaryOpT::cosh>(Tensor &);
template void unary_runner_inplace<UnaryOpT::tanh>(Tensor &);
template void unary_runner_inplace<UnaryOpT::asinh>(Tensor &);
template void unary_runner_inplace<UnaryOpT::acosh>(Tensor &);
template void unary_runner_inplace<UnaryOpT::atanh>(Tensor &);
template void unary_runner_inplace<UnaryOpT::erf>(Tensor &);
template void unary_runner_inplace<UnaryOpT::erfc>(Tensor &);
template void unary_runner_inplace<UnaryOpT::tgamma>(Tensor &);
template void unary_runner_inplace<UnaryOpT::lgamma>(Tensor &);
template void unary_runner_inplace<UnaryOpT::digamma>(Tensor &);
template void unary_runner_inplace<UnaryOpT::ceil>(Tensor &);
template void unary_runner_inplace<UnaryOpT::floor>(Tensor &);
template void unary_runner_inplace<UnaryOpT::round>(Tensor &);
template void unary_runner_inplace<UnaryOpT::isinf>(Tensor &);
template void unary_runner_inplace<UnaryOpT::isnan>(Tensor &);
template void unary_runner_inplace<UnaryOpT::isfinite>(Tensor &);
template void unary_runner_inplace<UnaryOpT::sigmoid>(Tensor &);
template void unary_runner_inplace<UnaryOpT::softplus>(Tensor &, double, double);
template void unary_runner_inplace<UnaryOpT::relu>(Tensor &);
template void unary_runner_inplace<UnaryOpT::relu6>(Tensor &);
template void unary_runner_inplace<UnaryOpT::elu>(Tensor &, double);
template void unary_runner_inplace<UnaryOpT::selu>(Tensor &);
template void unary_runner_inplace<UnaryOpT::silu>(Tensor &);
template void unary_runner_inplace<UnaryOpT::hardsigmoid>(Tensor &);
template void unary_runner_inplace<UnaryOpT::hardtanh>(Tensor &, double, double);
template void unary_runner_inplace<UnaryOpT::leaky_relu>(Tensor &, double);
template void unary_runner_inplace<UnaryOpT::log_sigmoid>(Tensor &);
template void unary_runner_inplace<UnaryOpT::softsign>(Tensor &);

}    // namespace tinytensor::cuda
