// index.cu
// Index runner

#include <tt/concepts.h>
#include <tt/scalar.h>
#include <tt/tensor.h>
#include <tt/util.h>

#include "tensor/backend/cuda/data_types.cuh"
#include "tensor/backend/cuda/kernel/index.cuh"
#include "tensor/backend/cuda/kernel_launch.cuh"
#include "tensor/backend/cuda/storage_cuda.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <variant>

namespace tinytensor::cuda {

using namespace kernel::index;

// Get values using mask
auto index_indices_runner(const Tensor &input, const Tensor &indices) -> Tensor {
    assert(input.device() == indices.device());
    using U = to_ctype_t<kDefaultInt>;
    const auto &indices_dev_memory = std::get<DeviceMemory<U>>(indices.get_storage<StorageCUDA>().dev_memory);
    const auto N = indices.numel();

    const auto res_device = input.device();

    // Create device memory for shape + stride for proper indexing
    const auto input_shape = MakeDeviceMemory(input.shape());
    const auto input_stride = MakeDeviceMemory(input.stride());
    const auto indices_shape = MakeDeviceMemory(indices.shape());
    const auto indices_stride = MakeDeviceMemory(indices.stride());

    return std::visit(
        [&](auto &&input_dev_memory) -> Tensor {
            using DT = std::remove_cvref_t<decltype(input_dev_memory)>;    // Storage type
            using T = template_parameter_t<DT>;                            // Underlying type

            // Create device memory for shape + stride for proper indexing
            const DeviceSpan<const T> input_span{input_dev_memory};
            const DataInfo<const T> input_info{input_span, input_shape, input_stride, input.offset()};

            const DeviceSpan<const U> indices_span{indices_dev_memory};
            const DataInfo<const U> indices_info{indices_span, indices_shape, indices_stride, indices.offset()};

            // Allocate for result
            auto res_dev_memory = DeviceMemory<T>::AllocateElements(static_cast<std::size_t>(N));
            auto res_span = DeviceSpan<T>{res_dev_memory};

            int num_blocks = ceil_div(N, SECTION_SIZE);
            dim3 blockDim(SECTION_SIZE);
            dim3 gridDim(num_blocks);

            launch(index_indices_kernel<T>, gridDim, blockDim, input_info, indices_info, DataInfo{res_span}, N);

            return {std::make_unique<StorageCUDA>(std::move(res_dev_memory)), input.dtype(), {N}, res_device};
        },
        input.get_storage<StorageCUDA>().dev_memory
    );
}

// Set indices of input using values
void index_put_indices_runner(Tensor &input, const Tensor &values, const Tensor &indices) {
    assert(input.device() == indices.device());
    using U = to_ctype_t<kDefaultInt>;
    const auto &indices_dev_memory = std::get<DeviceMemory<U>>(indices.get_storage<StorageCUDA>().dev_memory);
    const auto N = indices.numel();

    // Create device memory for shape + stride for proper indexing
    const auto input_shape = MakeDeviceMemory(input.shape());
    const auto input_stride = MakeDeviceMemory(input.stride());
    const auto values_shape = MakeDeviceMemory(values.shape());
    const auto values_stride = MakeDeviceMemory(values.stride());
    const auto indices_shape = MakeDeviceMemory(indices.shape());
    const auto indices_stride = MakeDeviceMemory(indices.stride());

    std::visit(
        [&](auto &&input_dev_memory) {
            using DT = std::remove_cvref_t<decltype(input_dev_memory)>;    // Storage type
            using T = template_parameter_t<DT>;                            // Underlying type

            // Create device memory for shape + stride for proper indexing
            DeviceSpan<T> input_span{input_dev_memory};
            DataInfo<T> input_info{input_span, input_shape, input_stride, input.offset()};
            const DeviceSpan<const T> values_span{std::get<DT>(values.get_storage<StorageCUDA>().dev_memory)};
            const DataInfo<const T> values_info{values_span, values_shape, values_stride, values.offset()};

            const DeviceSpan<const U> indices_span{indices_dev_memory};
            const DataInfo<const U> indices_info{indices_span, indices_shape, indices_stride, indices.offset()};

            int num_blocks = ceil_div(N, SECTION_SIZE);
            dim3 blockDim(SECTION_SIZE);
            dim3 gridDim(num_blocks);

            launch(index_put_indices_kernel<T>, gridDim, blockDim, input_info, values_info, indices_info, N);
        },
        input.get_storage<StorageCUDA>().dev_memory
    );
}

// Get values using mask
auto index_mask_runner(const Tensor &input, const Tensor &mask, int Nr) -> Tensor {
    assert(input.device() == mask.device());
    const auto &mask_dev_memory = std::get<DeviceMemory<uint8_t>>(mask.get_storage<StorageCUDA>().dev_memory);
    const auto Ni = input.numel();

    const auto res_device = input.device();

    // Create device memory for shape + stride for proper indexing
    const auto input_shape = MakeDeviceMemory(input.shape());
    const auto input_stride = MakeDeviceMemory(input.stride());
    const auto mask_shape = MakeDeviceMemory(mask.shape());
    const auto mask_stride = MakeDeviceMemory(mask.stride());

    return std::visit(
        [&](auto &&input_dev_memory) -> Tensor {
            using DT = std::remove_cvref_t<decltype(input_dev_memory)>;    // Storage type
            using T = template_parameter_t<DT>;                            // Underlying type

            // Create device memory for shape + stride for proper indexing
            const DeviceSpan<const T> input_span{input_dev_memory};
            const DataInfo<const T> input_info{input_span, input_shape, input_stride, input.offset()};

            const DeviceSpan<const uint8_t> mask_span{mask_dev_memory};
            const DataInfo<const uint8_t> mask_info{mask_span, mask_shape, mask_stride, mask.offset()};

            // Allocate for result
            auto res_dev_memory = DeviceMemory<T>::AllocateElements(static_cast<std::size_t>(Nr));
            auto res_span = DeviceSpan<T>{res_dev_memory};

            // Hierarchy of prefix sums depending on input size
            if (Ni <= SECTION_SIZE) {
                launch(kernel_mask_small<T>, {1}, {SECTION_SIZE}, input_info, mask_info, DataInfo{res_span}, Ni);
            } else if (Ni <= SECTION_SIZE * SECTION_SIZE) {
                int num_blocks = ceil_div(Ni, SECTION_SIZE);
                dim3 blockDim(SECTION_SIZE);
                dim3 gridDim(num_blocks);
                auto prefix_sum_dev_memory = DeviceMemory<int>::AllocateElements(static_cast<std::size_t>(Ni));
                auto scan_block_sum_dev_memory =
                    DeviceMemory<int>::AllocateElements(static_cast<std::size_t>(num_blocks));

                auto prefix_sum_span = DeviceSpan<int>{prefix_sum_dev_memory};
                auto scan_block_sum_span = DeviceSpan<int>{scan_block_sum_dev_memory};
                launch(
                    kernel_mask_medium_phase1,
                    gridDim,
                    blockDim,
                    mask_info,
                    prefix_sum_span,
                    scan_block_sum_span,
                    Ni
                );
                launch(kernel_mask_medium_phase2, {1}, blockDim, scan_block_sum_span, num_blocks);
                launch(
                    kernel_mask_medium_phase3<T>,
                    gridDim,
                    blockDim,
                    input_info,
                    mask_info,
                    prefix_sum_span,
                    scan_block_sum_span,
                    DataInfo{res_span},
                    Ni
                );
            } else {
                int num_blocks_phase1 = ceil_div(Ni, SECTION_SIZE);
                int num_blocks_phase2 = ceil_div(num_blocks_phase1, SECTION_SIZE);
                dim3 blockDim(SECTION_SIZE);

                auto prefix_sum_dev_memory = DeviceMemory<int>::AllocateElements(static_cast<std::size_t>(Ni));
                auto scan_block_sum1_dev_memory =
                    DeviceMemory<int>::AllocateElements(static_cast<std::size_t>(num_blocks_phase1));
                auto scan_block_sum2_dev_memory =
                    DeviceMemory<int>::AllocateElements(static_cast<std::size_t>(num_blocks_phase2));

                auto prefix_sum_span = DeviceSpan<int>{prefix_sum_dev_memory};
                auto scan_block_sum1_span = DeviceSpan<int>{scan_block_sum1_dev_memory};
                auto scan_block_sum2_span = DeviceSpan<int>{scan_block_sum2_dev_memory};
                launch(
                    kernel_mask_medium_phase1,
                    num_blocks_phase1,
                    blockDim,
                    mask_info,
                    prefix_sum_span,
                    scan_block_sum1_span,
                    Ni
                );
                launch(
                    kernel_mask_large_phase2,
                    num_blocks_phase2,
                    blockDim,
                    scan_block_sum1_span,
                    scan_block_sum2_span,
                    num_blocks_phase1
                );
                launch(kernel_mask_medium_phase2, {1}, blockDim, scan_block_sum2_span, num_blocks_phase2);
                launch(
                    kernel_mask_large_phase3,
                    num_blocks_phase2,
                    blockDim,
                    scan_block_sum1_span,
                    scan_block_sum2_span,
                    num_blocks_phase1
                );
                launch(
                    kernel_mask_medium_phase3<T>,
                    num_blocks_phase1,
                    blockDim,
                    input_info,
                    mask_info,
                    prefix_sum_span,
                    scan_block_sum1_span,
                    DataInfo{res_span},
                    Ni
                );
            }

            return {std::make_unique<StorageCUDA>(std::move(res_dev_memory)), input.dtype(), {Nr}, res_device};
        },
        input.get_storage<StorageCUDA>().dev_memory
    );
}

// Put values into input using mask
void index_put_mask_runner(Tensor &input, const Tensor &values, const Tensor &mask) {
    assert(input.device() == values.device() && input.device() == mask.device());
    const auto &mask_dev_memory = std::get<DeviceMemory<uint8_t>>(mask.get_storage<StorageCUDA>().dev_memory);
    const auto Ni = input.numel();

    // Create device memory for shape + stride for proper indexing
    const auto input_shape = MakeDeviceMemory(input.shape());
    const auto input_stride = MakeDeviceMemory(input.stride());
    const auto values_shape = MakeDeviceMemory(values.shape());
    const auto values_stride = MakeDeviceMemory(values.stride());
    const auto mask_shape = MakeDeviceMemory(mask.shape());
    const auto mask_stride = MakeDeviceMemory(mask.stride());

    std::visit(
        [&](auto &&input_dev_memory) {
            using DT = std::remove_cvref_t<decltype(input_dev_memory)>;    // Storage type
            using T = template_parameter_t<DT>;                            // Underlying type

            // Create device memory for shape + stride for proper indexing
            DeviceSpan<T> input_span{input_dev_memory};
            DataInfo<T> input_info{input_span, input_shape, input_stride, input.offset()};
            const DeviceSpan<const T> values_span{std::get<DT>(values.get_storage<StorageCUDA>().dev_memory)};
            const DataInfo<const T> values_info{values_span, values_shape, values_stride, values.offset()};

            const DeviceSpan<const uint8_t> mask_span{mask_dev_memory};
            const DataInfo<const uint8_t> mask_info{mask_span, mask_shape, mask_stride, mask.offset()};

            // Hierarchy of prefix sums depending on input size
            if (Ni <= SECTION_SIZE) {
                launch(kernel_input_put_mask_small<T>, {1}, {SECTION_SIZE}, input_info, values_info, mask_info, Ni);
            } else if (Ni <= SECTION_SIZE * SECTION_SIZE) {
                int num_blocks = ceil_div(Ni, SECTION_SIZE);
                dim3 blockDim(SECTION_SIZE);
                dim3 gridDim(num_blocks);
                auto prefix_sum_dev_memory = DeviceMemory<int>::AllocateElements(static_cast<std::size_t>(Ni));
                auto scan_block_sum_dev_memory =
                    DeviceMemory<int>::AllocateElements(static_cast<std::size_t>(num_blocks));

                auto prefix_sum_span = DeviceSpan<int>{prefix_sum_dev_memory};
                auto scan_block_sum_span = DeviceSpan<int>{scan_block_sum_dev_memory};
                launch(
                    kernel_mask_medium_phase1,
                    gridDim,
                    blockDim,
                    mask_info,
                    prefix_sum_span,
                    scan_block_sum_span,
                    Ni
                );
                launch(kernel_mask_medium_phase2, {1}, blockDim, scan_block_sum_span, num_blocks);
                launch(
                    kernel_input_put_mask_medium_phase3<T>,
                    gridDim,
                    blockDim,
                    input_info,
                    values_info,
                    mask_info,
                    prefix_sum_span,
                    scan_block_sum_span,
                    Ni
                );
            } else {
                int num_blocks_phase1 = ceil_div(Ni, SECTION_SIZE);
                int num_blocks_phase2 = ceil_div(num_blocks_phase1, SECTION_SIZE);
                dim3 blockDim(SECTION_SIZE);

                auto prefix_sum_dev_memory = DeviceMemory<int>::AllocateElements(static_cast<std::size_t>(Ni));
                auto scan_block_sum1_dev_memory =
                    DeviceMemory<int>::AllocateElements(static_cast<std::size_t>(num_blocks_phase1));
                auto scan_block_sum2_dev_memory =
                    DeviceMemory<int>::AllocateElements(static_cast<std::size_t>(num_blocks_phase2));

                auto prefix_sum_span = DeviceSpan<int>{prefix_sum_dev_memory};
                auto scan_block_sum1_span = DeviceSpan<int>{scan_block_sum1_dev_memory};
                auto scan_block_sum2_span = DeviceSpan<int>{scan_block_sum2_dev_memory};
                launch(
                    kernel_mask_medium_phase1,
                    num_blocks_phase1,
                    blockDim,
                    mask_info,
                    prefix_sum_span,
                    scan_block_sum1_span,
                    Ni
                );
                launch(
                    kernel_mask_large_phase2,
                    num_blocks_phase2,
                    blockDim,
                    scan_block_sum1_span,
                    scan_block_sum2_span,
                    num_blocks_phase1
                );
                launch(kernel_mask_medium_phase2, {1}, blockDim, scan_block_sum2_span, num_blocks_phase2);
                launch(
                    kernel_mask_large_phase3,
                    num_blocks_phase2,
                    blockDim,
                    scan_block_sum1_span,
                    scan_block_sum2_span,
                    num_blocks_phase1
                );
                launch(
                    kernel_input_put_mask_medium_phase3<T>,
                    num_blocks_phase1,
                    blockDim,
                    input_info,
                    values_info,
                    mask_info,
                    prefix_sum_span,
                    scan_block_sum1_span,
                    Ni
                );
            }
        },
        input.get_storage<StorageCUDA>().dev_memory
    );
}

}    // namespace tinytensor::cuda
