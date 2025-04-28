// conv.cu
// Conv2d runners

#include <tt/concepts.h>
#include <tt/scalar.h>
#include <tt/shape.h>
#include <tt/tensor.h>
#include <tt/util.h>

#include "tensor/backend/common/binary.h"
#include "tensor/backend/common/reduce.h"
#include "tensor/backend/cuda/config.cuh"
#include "tensor/backend/cuda/conv.h"
#include "tensor/backend/cuda/data_types.cuh"
#include "tensor/backend/cuda/kernel/binary.cuh"
#include "tensor/backend/cuda/kernel/conv.cuh"
#include "tensor/backend/cuda/kernel/matmul.cuh"
#include "tensor/backend/cuda/kernel_launch.cuh"
#include "tensor/backend/cuda/storage_cuda.h"

#include <cassert>
#include <cstddef>
#include <memory>
#include <optional>
#include <tuple>
#include <type_traits>
#include <variant>

namespace tinytensor::cuda {

using namespace tinytensor::common::binary;
using namespace tinytensor::common::reduce;
using namespace kernel::conv;
using namespace kernel::matmul;
using namespace kernel::binary;

auto batched_conv2d_forward_runner(
    const Tensor &input,
    const Tensor &weight,
    const std::optional<Tensor> &bias,
    int stride,
    int padding
) -> Tensor {
    assert(input.dim() == 4 && weight.dim() == 4);
    assert(input.size(1) == weight.size(1));     // Input channels match
    assert(weight.size(2) == weight.size(3));    // weight has square kernel
    assert(stride > 0);
    assert(stride >= 0);
    // Input shape properties
    const int B = input.size(0);
    const int C_in = input.size(1);
    const int H_in = input.size(2);
    const int W_in = input.size(3);
    const int kernel_size = weight.size(2);

    // Output shape properties
    const int C_out = weight.size(0);
    const int H_out = (H_in + 2 * padding - kernel_size) / stride + 1;
    const int W_out = (W_in + 2 * padding - kernel_size) / stride + 1;

    // Input unrolled into a matrix, per batch item
    const int input_unrolled_height = C_in * kernel_size * kernel_size;
    const int input_unrolled_width = H_out * W_out;
    const int input_unrolled_size = input_unrolled_height * input_unrolled_width;

    // Properties for result
    const auto res_shape = Shape({B, C_out, H_out, W_out});
    const auto res_device = input.device();

    // Reshape weight for matmuls
    const Tensor w = weight.reshape({C_out, C_in * kernel_size * kernel_size});

    // conv requires contiguous data
    const auto input_cont = input.contiguous();

    // Setup bias if passed
    std::optional<Tensor> expanded_bias;
    std::optional<DeviceMemory<int>> bias_shape;
    std::optional<DeviceMemory<int>> bias_stride;
    if (bias) {
        expanded_bias = bias->contiguous().reshape({C_out, 1, 1}).expand({C_out, H_out, W_out});
        bias_shape = MakeDeviceMemory(expanded_bias->shape());
        bias_stride = MakeDeviceMemory(expanded_bias->stride());
    }

    // input and weight need to be same type, so visit on one to reduce codegen
    return std::visit(
        [&](auto &&tensor_dev_memory) -> Tensor {
            using DT = std::remove_cvref_t<decltype(tensor_dev_memory)>;    // Storage type
            using T = template_parameter_t<DT>;                             // Underlying type

            // Allocate for result
            auto res_dev_memory = DeviceMemory<T>::AllocateElements(static_cast<std::size_t>(res_shape.numel()));

            // Allocate for input unrolled for matmul
            auto input_unrolled_dev_memory =
                DeviceMemory<T>::AllocateElements(static_cast<std::size_t>(input_unrolled_size));

            // Get operand data spans
            // Data can have an offset whilst still being contiguous
            const T *input_data_ptr = tensor_dev_memory.data_ptr();
            const T *w_data_ptr = std::get<DT>(w.template get_storage<StorageCUDA>().dev_memory).data_ptr();
            input_data_ptr += input_cont.offset();    // NOLINT(*-pointer-arithmetic)
            w_data_ptr += w.offset();                 // NOLINT(*-pointer-arithmetic)
            const DeviceSpan<const T> input_span{input_data_ptr, static_cast<std::size_t>(input_cont.numel())};
            const DeviceSpan<const T> w_span{w_data_ptr, static_cast<std::size_t>(w.numel())};
            DeviceSpan<T> input_unrolled_span{input_unrolled_dev_memory};
            T *res_data_ptr = res_dev_memory.data_ptr();

            // Kernel properties for unroll and matmul
            const dim3 block_dim_unroll = block_1d();
            const dim3 grid_dim_unroll(ceil_div(input_unrolled_size, static_cast<int>(block_dim_unroll.x)));

            const int mat_N = C_out;
            const int mat_K = input_unrolled_height;
            const int mat_M = input_unrolled_width;
            const dim3 block_dim_matmul{
                (properties::TILE_WIDTH * properties::TILE_HEIGHT) / (properties::TN * properties::TM)
            };
            const dim3 grid_dim_matmul{
                static_cast<unsigned int>(ceil_div(mat_M, properties::TILE_WIDTH)),
                static_cast<unsigned int>(ceil_div(mat_N, properties::TILE_HEIGHT)),
                1
            };

            // Unroll over each batch element and perform matmul
            for (int b = 0; b < B; ++b) {
                // Unroll this batch item from input to input_unrolled
                launch(
                    unroll_kernel<T>,
                    grid_dim_unroll,
                    block_dim_unroll,
                    input_span,
                    input_unrolled_span,
                    b,
                    C_in,
                    H_in,
                    W_in,
                    kernel_size,
                    stride,
                    padding
                );
                // Perform unrolled matmul
                const DeviceSpan<T> res_span{res_data_ptr, static_cast<std::size_t>(C_out * H_out * W_out)};
                launch(
                    matmul_kernel<T>,
                    grid_dim_matmul,
                    block_dim_matmul,
                    w_span,
                    input_unrolled_span,
                    res_span,
                    mat_N,
                    mat_K,
                    mat_M
                );

                if (bias) {
                    DataInfo<T> res_info{res_span};
                    const DataInfo<const T> bias_info{
                        {std::get<DT>(expanded_bias->template get_storage<StorageCUDA>().dev_memory)},
                        *bias_shape,
                        *bias_stride,
                        expanded_bias->offset()
                    };
                    using KernelOp = typename common::binary::OpFactory<T, BinaryOpT::add>::KernelOp;
                    const auto kernel = binary_kernel<T, KernelOp>;
                    launch(
                        kernel,
                        grid_1d(C_out * H_out * W_out),
                        block_1d(),
                        res_info,
                        bias_info,
                        KernelOp{},
                        C_out * H_out * W_out
                    );
                }

                // Increment pointers to next batch
                res_data_ptr += (C_out * H_out * W_out);    // NOLINT(*-pointer-arithmetic)
            }

            return {
                std::make_unique<StorageCUDA>(std::move(res_dev_memory)),
                to_scalar<T>::type,
                res_shape,
                res_device
            };
        },
        input_cont.template get_storage<StorageCUDA>().dev_memory
    );
}

auto batched_conv2d_backward_runner(
    const Tensor &grad_output,
    const Tensor &input,
    const Tensor &weight,
    const std::optional<Tensor> &bias,
    int stride,
    int padding
) -> std::tuple<Tensor, Tensor, std::optional<Tensor>> {
    assert(input.dim() == 4 && weight.dim() == 4);
    assert(input.size(1) == weight.size(1));     // Input channels match
    assert(weight.size(2) == weight.size(3));    // weight has square kernel
    assert(stride > 0);
    assert(stride >= 0);
    // Input shape properties
    const int B = input.size(0);
    const int C_in = input.size(1);
    const int H_in = input.size(2);
    const int W_in = input.size(3);
    const int kernel_size = weight.size(2);

    // Output shape properties
    const int C_out = weight.size(0);
    const int H_out = (H_in + 2 * padding - kernel_size) / stride + 1;
    const int W_out = (W_in + 2 * padding - kernel_size) / stride + 1;

    // Input unrolled into a matrix, per batch item
    const int input_unrolled_height = C_in * kernel_size * kernel_size;
    const int input_unrolled_width = H_out * W_out;
    const int input_unrolled_size = input_unrolled_height * input_unrolled_width;

    // Properties for result
    const auto res_shape = Shape({B, C_out, H_out, W_out});
    assert(grad_output.shape() == res_shape);

    // Reshape weight for matmuls
    const Tensor w = weight.reshape({C_out, C_in * kernel_size * kernel_size}).permute({1, 0}).contiguous();

    // conv requires contiguous data
    const auto input_cont = input.contiguous();

    // input and weight need to be same type, so visit on one to reduce codegen
    return std::visit(
        [&](auto &&tensor_dev_memory) -> std::tuple<Tensor, Tensor, std::optional<Tensor>> {
            using DT = std::remove_cvref_t<decltype(tensor_dev_memory)>;    // Storage type
            using T = template_parameter_t<DT>;                             // Underlying type
            using KernelOp = typename common::binary::OpFactory<T, BinaryOpT::add>::KernelOp;

            // Allocate for result
            auto grad_input_fold = DeviceMemory<T>::AllocateElements(static_cast<std::size_t>(input.numel()), 0);
            auto grad_w_local = DeviceMemory<T>::AllocateElements(static_cast<std::size_t>(weight.numel()), 0);
            auto grad_w_global = DeviceMemory<T>::AllocateElements(static_cast<std::size_t>(weight.numel()), 0);

            // Allocate for input unrolled for matmul
            auto input_unrolled_dev_memory =
                DeviceMemory<T>::AllocateElements(static_cast<std::size_t>(input_unrolled_size), 0);

            // Get operand data spans
            // Data can have an offset whilst still being contiguous
            const T *input_data_ptr = tensor_dev_memory.data_ptr();
            const T *w_data_ptr = std::get<DT>(w.template get_storage<StorageCUDA>().dev_memory).data_ptr();
            input_data_ptr += input_cont.offset();
            w_data_ptr += w.offset();
            const DeviceSpan<const T> input_span{input_data_ptr, static_cast<std::size_t>(input_cont.numel())};
            const DeviceSpan<const T> w_span{w_data_ptr, static_cast<std::size_t>(w.numel())};
            const DeviceSpan<const T> grad_out_span{
                std::get<DT>(grad_output.template get_storage<StorageCUDA>().dev_memory)
            };
            DeviceSpan<T> g_w_local_span{grad_w_local};
            DeviceSpan<T> g_w_global_span{grad_w_global};

            DeviceSpan<T> input_unrolled_span{input_unrolled_dev_memory};
            T *grad_input_data_ptr = grad_input_fold.data_ptr();
            const T *g_out_data_ptr = grad_out_span.get();

            // Kernel properties for unroll and matmul
            const dim3 block_dim_1d = block_1d();
            const dim3 grid_dim_unroll(ceil_div(input_unrolled_size, static_cast<int>(block_dim_1d.x)));

            const dim3 block_dim_matmul{
                (properties::TILE_WIDTH * properties::TILE_HEIGHT) / (properties::TN * properties::TM)
            };
            const dim3 grid_dim_matmul_w{
                static_cast<unsigned int>(ceil_div(input_unrolled_height, properties::TILE_WIDTH)),
                static_cast<unsigned int>(ceil_div(C_out, properties::TILE_HEIGHT)),
                1
            };
            const dim3 grid_dim_matmul_input{
                static_cast<unsigned int>(ceil_div(input_unrolled_width, properties::TILE_WIDTH)),
                static_cast<unsigned int>(ceil_div(input_unrolled_height, properties::TILE_HEIGHT)),
                1
            };

            // Unroll over each batch element and perform matmul
            for (int b = 0; b < B; ++b) {
                // Unroll this batch item from input to input_unrolled
                launch(
                    unroll_transpose_kernel<T>,
                    grid_dim_unroll,
                    block_dim_1d,
                    input_span,
                    input_unrolled_span,
                    b,
                    C_in,
                    H_in,
                    W_in,
                    kernel_size,
                    stride,
                    padding
                );
                DeviceSpan<T> grad_input_span{grad_input_data_ptr, static_cast<std::size_t>(C_in * H_in * W_in)};
                const DeviceSpan<const T> g_out_slice_span{
                    g_out_data_ptr,
                    static_cast<std::size_t>(C_out * H_out * W_out)
                };

                // grad_w
                // Transpose input_unrolled_span
                // Call matmaul into local_w
                // add local_w into global_w grad
                launch(
                    matmul_kernel<T>,
                    grid_dim_matmul_w,
                    block_dim_matmul,
                    g_out_slice_span,
                    DeviceSpan<const T>{input_unrolled_span},
                    g_w_local_span,
                    C_out,
                    input_unrolled_width,
                    input_unrolled_height
                );
                const auto kernel = binary_kernel<T, KernelOp>;
                launch(
                    kernel,
                    grid_1d(weight.numel()),
                    block_dim_1d,
                    DataInfo<T>{g_w_global_span},
                    DataInfo<const T>{g_w_local_span},
                    KernelOp{},
                    weight.numel()
                );

                // grad_x
                // Result is unrolled into reused buffer input_unrolled_span
                // Need to then roll back up
                launch(
                    matmul_kernel<T>,
                    grid_dim_matmul_input,
                    block_dim_matmul,
                    w_span,
                    g_out_slice_span,
                    input_unrolled_span,
                    input_unrolled_height,
                    C_out,
                    input_unrolled_width
                );
                launch(
                    col2im_kernel<T>,
                    grid_dim_unroll,
                    block_dim_1d,
                    grad_input_span,
                    DeviceSpan<const T>{input_unrolled_span},
                    0,
                    C_in,
                    H_in,
                    W_in,
                    kernel_size,
                    stride,
                    padding
                );

                // Increment pointers to next batch
                g_out_data_ptr += (C_out * H_out * W_out);      // NOLINT(*-pointer-arithmetic)
                grad_input_data_ptr += (C_in * H_in * W_in);    // NOLINT(*-pointer-arithmetic)
            }
            Tensor grad_input(
                std::make_unique<StorageCUDA>(std::move(grad_input_fold)),
                to_scalar<T>::type,
                input.shape(),
                input.device()
            );
            Tensor grad_w(
                std::make_unique<StorageCUDA>(std::move(grad_w_global)),
                to_scalar<T>::type,
                weight.shape(),
                weight.device()
            );
            return {
                grad_input,
                grad_w,
                bias.has_value() ? std::optional<Tensor>{grad_output.sum(3).sum(2).sum(0)} : std::nullopt
            };
        },
        input_cont.template get_storage<StorageCUDA>().dev_memory
    );
}

template <ReduceOpT Op>
auto batched_pool2d_forward_runner(const Tensor &input, int kernel_size, int stride, int padding) -> Tensor {
    assert(input.dim() == 4);
    assert(kernel_size > 0);
    assert(stride > 0);
    assert(padding >= 0);
    // Input shape properties
    const int B = input.size(0);
    const int C = input.size(1);
    const int H_in = input.size(2);
    const int W_in = input.size(3);

    // Output shape properties
    const int H_out = (H_in + 2 * padding - kernel_size) / stride + 1;
    const int W_out = (W_in + 2 * padding - kernel_size) / stride + 1;

    int W_grid = ceil_div(W_out, TW);
    int H_grid = ceil_div(H_out, TW);

    // Properties for result
    const auto res_shape = Shape({B, C, H_out, W_out});
    const auto res_device = input.device();

    // conv requires contiguous data
    const auto input_cont = input.contiguous();

    // input and weight need to be same type, so visit on one to reduce codegen
    return std::visit(
        [&](auto &&tensor_dev_memory) -> Tensor {
            using DT = std::remove_cvref_t<decltype(tensor_dev_memory)>;    // Storage type
            using T = template_parameter_t<DT>;                             // Underlying type
            using R = common::reduce::Result<T, Op>::type;                  // Result type
            using KernelOp = typename common::reduce::OpFactory<R, Op>::KernelOp;

            // Allocate for result
            auto res_dev_memory = DeviceMemory<R>::AllocateElements(static_cast<std::size_t>(res_shape.numel()));

            // Get operand data spans
            // Data can have an offset whilst still being contiguous
            const T *input_data_ptr = tensor_dev_memory.data_ptr();
            input_data_ptr += input_cont.offset();
            const DeviceSpan<const T> input_span{input_data_ptr, static_cast<std::size_t>(input_cont.numel())};

            const dim3 block_dim = block_2d();
            const dim3 grid_dim(B, C, H_grid * W_grid);

            launch(
                pool_kernel<T, R, KernelOp>,
                grid_dim,
                block_dim,
                input_span,
                DeviceSpan<R>{res_dev_memory},
                KernelOp{static_cast<R>(kernel_size * kernel_size)},
                C,
                H_in,
                W_in,
                kernel_size,
                stride,
                padding
            );

            return {
                std::make_unique<StorageCUDA>(std::move(res_dev_memory)),
                to_scalar<R>::type,
                res_shape,
                res_device
            };
        },
        input_cont.template get_storage<StorageCUDA>().dev_memory
    );
}

template <ReduceOpT Op>
auto batched_pool2d_backward_runner(
    const Tensor &grad_output,
    const Tensor &input,
    const Tensor &result,
    int kernel_size,
    int stride,
    int padding
) -> Tensor {
    assert(input.dim() == 4);
    assert(kernel_size > 0);
    assert(stride > 0);
    assert(padding >= 0);
    // Input shape properties
    const int B = input.size(0);
    const int C = input.size(1);
    const int H_in = input.size(2);
    const int W_in = input.size(3);

    // Output shape properties
    const int H_out = (H_in + 2 * padding - kernel_size) / stride + 1;
    const int W_out = (W_in + 2 * padding - kernel_size) / stride + 1;

    int W_grid = ceil_div(W_out, TW);
    int H_grid = ceil_div(H_out, TW);

    // Properties for result
    const auto res_shape = Shape({B, C, H_out, W_out});
    assert(grad_output.shape() == res_shape);

    // conv requires contiguous data
    const auto input_cont = input.contiguous();

    // input and weight need to be same type, so visit on one to reduce codegen
    return std::visit(
        [&](auto &&tensor_dev_memory) -> Tensor {
            using DT = std::remove_cvref_t<decltype(tensor_dev_memory)>;    // Storage type
            using T = template_parameter_t<DT>;                             // Underlying type
            using R = common::reduce::Result<T, Op>::type;                  // Result type
            using DR = DeviceMemory<R>;
            using KernelOp = typename common::reduce::OpFactory<R, Op>::KernelOp;

            // Allocate for result
            auto grad_input_memory = DeviceMemory<R>::AllocateElements(static_cast<std::size_t>(input.numel()));

            // Get operand data spans
            // Data can have an offset whilst still being contiguous
            const T *input_data_ptr = tensor_dev_memory.data_ptr();
            input_data_ptr += input_cont.offset();
            const DeviceSpan<const T> input_span{input_data_ptr, static_cast<std::size_t>(input_cont.numel())};
            const DeviceSpan<const R> res_span{std::get<DR>(result.template get_storage<StorageCUDA>().dev_memory)};
            const DeviceSpan<const R> grad_output_span{
                std::get<DR>(grad_output.template get_storage<StorageCUDA>().dev_memory)
            };

            const dim3 block_dim = block_2d();
            const dim3 grid_dim(B, C, H_grid * W_grid);

            // Cuda has some trouble with template deduction using concept requires
            if constexpr (std::is_same_v<KernelOp, MinOp<R>> || std::is_same_v<KernelOp, MaxOp<R>>) {
                launch(
                    pool_backward_min_max_kernel<T, R>,
                    grid_dim,
                    block_dim,
                    grad_output_span,
                    DeviceSpan<R>{grad_input_memory},
                    input_span,
                    res_span,
                    C,
                    H_in,
                    W_in,
                    kernel_size,
                    stride,
                    padding
                );
            } else if constexpr (std::is_same_v<KernelOp, MeanOp<R>>) {
                launch(
                    pool_backward_mean_kernel<T, R>,
                    grid_dim,
                    block_dim,
                    grad_output_span,
                    DeviceSpan<R>{grad_input_memory},
                    input_span,
                    res_span,
                    C,
                    H_in,
                    W_in,
                    kernel_size,
                    stride,
                    padding
                );
            } else {
                static_assert(dependent_false_v<T>, "Unknown pool backward OP");
            }

            return {
                std::make_unique<StorageCUDA>(std::move(grad_input_memory)),
                to_scalar<R>::type,
                input.shape(),
                input.device()
            };
        },
        input_cont.template get_storage<StorageCUDA>().dev_memory
    );
}

template Tensor batched_pool2d_forward_runner<ReduceOpT::min>(const Tensor &, int, int, int);
template Tensor batched_pool2d_forward_runner<ReduceOpT::max>(const Tensor &, int, int, int);
template Tensor batched_pool2d_forward_runner<ReduceOpT::mean>(const Tensor &, int, int, int);

template Tensor
    batched_pool2d_backward_runner<ReduceOpT::min>(const Tensor &, const Tensor &, const Tensor &, int, int, int);
template Tensor
    batched_pool2d_backward_runner<ReduceOpT::max>(const Tensor &, const Tensor &, const Tensor &, int, int, int);
template Tensor
    batched_pool2d_backward_runner<ReduceOpT::mean>(const Tensor &, const Tensor &, const Tensor &, int, int, int);

}    // namespace tinytensor::cuda
