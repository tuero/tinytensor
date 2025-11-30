// reduce.cu
// Reduction runner

#include <tt/concepts.h>
#include <tt/scalar.h>
#include <tt/shape.h>
#include <tt/tensor.h>
#include <tt/util.h>

#include "tensor/backend/common/reduce.h"
#include "tensor/backend/cuda/config.cuh"
#include "tensor/backend/cuda/data_types.cuh"
#include "tensor/backend/cuda/kernel/reduce.cuh"
#include "tensor/backend/cuda/kernel_launch.cuh"
#include "tensor/backend/cuda/reduce.h"
#include "tensor/backend/cuda/storage_cuda.h"

#include <cassert>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <variant>

namespace tinytensor::cuda {

using namespace tinytensor::common::reduce;
using namespace kernel::reduce;

template <ReduceOpT Op>
auto reduce_dim_runner(const Tensor &tensor, int dim) -> Tensor {
    assert(dim >= 0 && dim < tensor.dim());
    auto res_shape = tensor.shape();
    const int RN = res_shape[dim];
    res_shape[dim] = 1;
    const int N = res_shape.numel();

    // Create device memory for shape + stride for proper indexing
    const int device_id = tensor.device().id;
    const auto shape = MakeDeviceMemory(device_id, tensor.shape());
    const auto stride = MakeDeviceMemory(device_id, tensor.stride());

    return std::visit(
        [&](auto &&dev_memory) -> Tensor {
            using DT = std::remove_cvref_t<decltype(dev_memory)>;    // Storage type
            using T = template_parameter_t<DT>;                      // Underlying type
            using V = Result<T, Op>::val_type;                       // Op value type
            using R = Result<T, Op>::res_type;                       // Result type
            using KernelOp = typename OpFactory<V, Op>::KernelOp;

            // Allocate for result
            auto res_dev_memory = DeviceMemory<R>::AllocateElements(device_id, static_cast<std::size_t>(N));

            const DataInfo<const T> input_info{dev_memory, shape, stride, tensor.offset()};

            // Only one kernel is launched, as each tread is an output in the reduced tensor
            // where each thread sums over the reduced dim
            const auto kernel = reduce_dim_kernel<T, R, KernelOp>;
            launch(
                device_id,
                kernel,
                grid_1d(N),
                block_1d(),
                input_info,
                DeviceSpan<R>{res_dev_memory},
                KernelOp{static_cast<V>(RN)},
                dim,
                N
            );
            return {
                std::make_unique<StorageCUDA>(std::move(res_dev_memory)),
                std::is_same_v<T, R> ? tensor.dtype() : to_scalar<R>::type,
                res_shape,
                tensor.device()
            };
        },
        tensor.get_storage<StorageCUDA>().dev_memory
    );
}

// Reduce entire tensor into a single value
// Depending on the number of elements in the tensor, multiple reductions will be required
// Multiple reductions have to allocate a scratch space for in/out reductions,
// but we try to prevent that if we only need a single reduction
template <ReduceOpT Op>
auto reduce_all_runner(const Tensor &tensor) -> Tensor {
    const auto res_device = tensor.device();
    int N = tensor.numel();
    // Create device memory for shape + stride for proper indexing
    const int device_id = tensor.device().id;
    const auto arr_shape = MakeDeviceMemory(device_id, tensor.shape());
    const auto arr_stride = MakeDeviceMemory(device_id, tensor.stride());

    return std::visit(
        [&](auto &&dev_memory) -> Tensor {
            using DT = std::remove_cvref_t<decltype(dev_memory)>;    // Storage type
            using T = template_parameter_t<DT>;                      // Underlying type
            using V = Result<T, Op>::val_type;                       // Op value type
            using R = Result<T, Op>::res_type;                       // Result type
            using KernelOp = typename OpFactory<V, Op>::KernelOp;

            // Kernel properties
            auto grid_dim = grid_1d(N, 2);
            const auto block_dim = block_1d();

            // Result
            auto res_memory = DeviceMemory<R>::AllocateElements(device_id, 1);

            DataInfo<const T> input_info{dev_memory, arr_shape, arr_stride, tensor.offset()};

            // If grid_dim.x <= 1, i.e., only one reduction required, we can simply store directly
            // into result
            if (grid_dim.x <= 1) {
                launch(
                    device_id,
                    reduce_all_kernel<T, R, KernelOp>,
                    grid_dim,
                    block_dim,
                    input_info,
                    DeviceSpan<R>{res_memory},
                    KernelOp{static_cast<V>(N)},
                    static_cast<std::size_t>(N)
                );
                return {std::make_unique<StorageCUDA>(std::move(res_memory)), to_scalar<R>::type, {1}, res_device};
            }

            // Otherwise, we need multiple reduction passes and need to allocate temporary buffers
            // to reduce into
            auto reduction_memory_in = DeviceMemory<R>::AllocateElements(device_id, grid_dim.x);
            auto reduction_memory_out = DeviceMemory<R>::AllocateElements(device_id, grid_dim.x);

            launch(
                device_id,
                kernel::reduce::reduce_all_kernel<T, R, KernelOp>,
                grid_dim,
                block_dim,
                input_info,
                DeviceSpan<R>{reduction_memory_out},
                KernelOp{},
                static_cast<std::size_t>(N)
            );

            while (grid_dim.x > 1) {
                N = static_cast<int>(grid_dim.x);
                grid_dim.x = ceil_div(grid_dim.x, block_dim.x);
                std::swap(reduction_memory_in, reduction_memory_out);
                // All buffers are type R now
                launch(
                    device_id,
                    kernel::reduce::reduce_all_kernel<R, R, KernelOp>,
                    grid_dim,
                    block_dim,
                    DataInfo<const R>{reduction_memory_in},
                    (grid_dim.x <= 1)    // If last reduce, place directly into result
                        ? DeviceSpan<R>{res_memory}
                        : DeviceSpan<R>{reduction_memory_out},
                    KernelOp{},
                    static_cast<std::size_t>(N)
                );
            }

            return {
                std::make_unique<StorageCUDA>(std::move(res_memory)),
                std::is_same_v<T, R> ? tensor.dtype() : to_scalar<R>::type,
                {1},
                res_device
            };
        },
        tensor.get_storage<StorageCUDA>().dev_memory
    );
}

template Tensor reduce_dim_runner<ReduceOpT::min>(const Tensor &tensor, int dim);
template Tensor reduce_dim_runner<ReduceOpT::argmin>(const Tensor &tensor, int dim);
template Tensor reduce_dim_runner<ReduceOpT::max>(const Tensor &tensor, int dim);
template Tensor reduce_dim_runner<ReduceOpT::argmax>(const Tensor &tensor, int dim);
template Tensor reduce_dim_runner<ReduceOpT::sum>(const Tensor &tensor, int dim);
template Tensor reduce_dim_runner<ReduceOpT::all>(const Tensor &tensor, int dim);
template Tensor reduce_dim_runner<ReduceOpT::any>(const Tensor &tensor, int dim);

template Tensor reduce_all_runner<ReduceOpT::min>(const Tensor &tensor);
template Tensor reduce_all_runner<ReduceOpT::argmin>(const Tensor &tensor);
template Tensor reduce_all_runner<ReduceOpT::max>(const Tensor &tensor);
template Tensor reduce_all_runner<ReduceOpT::argmax>(const Tensor &tensor);
template Tensor reduce_all_runner<ReduceOpT::sum>(const Tensor &tensor);
template Tensor reduce_all_runner<ReduceOpT::all>(const Tensor &tensor);
template Tensor reduce_all_runner<ReduceOpT::any>(const Tensor &tensor);

}    // namespace tinytensor::cuda
