// reduce.cpp
// Reduction runner

#include "tensor/backend/cpu/reduce.h"

#include <tt/concepts.h>
#include <tt/scalar.h>
#include <tt/shape.h>
#include <tt/tensor.h>

#include "tensor/backend/common/reduce.h"
#include "tensor/backend/common/span.h"
#include "tensor/backend/cpu/data_types.h"
#include "tensor/backend/cpu/kernel/reduce.hpp"
#include "tensor/backend/cpu/storage_cpu.h"

#include <cstddef>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace tinytensor::cpu {

using namespace tinytensor::common::reduce;
using namespace kernel::reduce;

template <ReduceOpT Op>
auto reduce_dim_runner(const Tensor &tensor, int dim) -> Tensor {
    auto res_shape = tensor.shape();
    const int RN = res_shape[dim];
    res_shape[dim] = 1;
    const int N = res_shape.numel();
    return std::visit(
        [&](auto &&tensor_storage) -> Tensor {
            using DT = std::remove_cvref_t<decltype(tensor_storage)>;    // Storage type
            using T = template_parameter_t<std::remove_cvref_t<DT>>;     // Underlying type
            using V = Result<T, Op>::val_type;                           // Op value type
            using R = Result<T, Op>::res_type;                           // Result type
            using KernelOp = typename OpFactory<V, Op>::KernelOp;

            const HostSpan<const T> a{tensor_storage};
            const auto shape = HostSpan<const int>(tensor.shape());
            const auto stride = HostSpan<const int>(tensor.stride());
            const DataInfo<const T> data_info{a, shape, stride, tensor.offset()};

            // Allocate for result
            std::vector<R> result(static_cast<std::size_t>(N));

            reduce_dim_kernel(data_info, HostSpan<R>{result}, KernelOp{static_cast<V>(RN)}, dim, N);

            return {
                std::make_unique<StorageCPU>(std::move(result)),
                std::is_same_v<T, R> ? tensor.dtype() : to_scalar<R>::type,
                res_shape,
                tensor.device()
            };
        },
        tensor.get_storage<StorageCPU>().storage
    );
}

template <ReduceOpT Op>
auto reduce_all_runner(const Tensor &tensor) -> Tensor {
    const int N = tensor.shape().numel();
    return std::visit(
        [&](auto &&tensor_storage) -> Tensor {
            using DT = std::remove_cvref_t<decltype(tensor_storage)>;    // Storage type
            using T = template_parameter_t<std::remove_cvref_t<DT>>;     // Underlying type
            using V = Result<T, Op>::val_type;                           // Op value type
            using R = Result<T, Op>::res_type;                           // Result type
            using KernelOp = typename OpFactory<V, Op>::KernelOp;

            const HostSpan<const T> a{tensor_storage};
            const auto shape = HostSpan<const int>(tensor.shape());
            const auto stride = HostSpan<const int>(tensor.stride());
            const DataInfo<const T> data_info{a, shape, stride, tensor.offset()};

            // Allocate for result
            std::vector<R> result(static_cast<std::size_t>(1));

            reduce_all_kernel(data_info, HostSpan<R>{result}, KernelOp{static_cast<V>(N)}, N);
            return {
                std::make_unique<StorageCPU>(std::move(result)),
                std::is_same_v<T, R> ? tensor.dtype() : to_scalar<R>::type,
                {1},
                tensor.device()
            };
        },
        tensor.get_storage<StorageCPU>().storage
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

}    // namespace tinytensor::cpu
