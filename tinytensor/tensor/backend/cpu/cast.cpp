// cast.cpp
// Cast from one type to another

#include "tensor/backend/cpu/cast.h"

#include <tt/concepts.h>
#include <tt/exception.h>
#include <tt/scalar.h>
#include <tt/tensor.h>

#include "tensor/backend/common/kernel/unary.hpp"
#include "tensor/backend/common/span.h"
#include "tensor/backend/cpu/data_types.h"
#include "tensor/backend/cpu/kernel/unary.hpp"
#include "tensor/backend/cpu/storage_cpu.h"

#include <cstddef>
#include <memory>
#include <type_traits>
#include <variant>
#include <vector>

namespace tinytensor::cpu {

using namespace common::kernel::unary;

namespace {
template <typename T, typename R, typename KernelOp>
auto _cast_runner(const DataInfo<const T> &tensor_info, int N) -> std::unique_ptr<StorageCPU> {
    std::vector<R> result(static_cast<std::size_t>(N));
    kernel::unary::unary_kernel<true, T, R>(tensor_info, HostSpan<R>(result), KernelOp{}, N);
    return std::make_unique<StorageCPU>(std::move(result));
}
}    // namespace

auto cast_runner(const Tensor &tensor, ScalarType dtype) -> Tensor {
    const int N = tensor.shape().numel();

    // Create device memory for shape + stride for proper indexing
    const auto shape = HostSpan<const int>(tensor.shape());
    const auto stride = HostSpan<const int>(tensor.stride());

    return std::visit(
        [&](auto &&tensor_storage) -> Tensor {
            using DT = std::remove_cvref_t<decltype(tensor_storage)>;    // Storage type
            using T = template_parameter_t<DT>;                          // Underlying type

            const HostSpan<const T> a{tensor_storage};
            const DataInfo<const T> tensor_info{a, shape, stride, tensor.offset()};

            auto dtype_switch = [&](ScalarType cast_to_type) {
                switch (cast_to_type) {
                case kBool: {
                    using R = to_ctype_t<kBool>;
                    using KernOp = OpIdentity<R>;
                    return _cast_runner<T, R, KernOp>(tensor_info, N);
                }
                case kU8: {
                    using R = to_ctype_t<kU8>;
                    using KernOp = OpIdentity<R>;
                    return _cast_runner<T, R, KernOp>(tensor_info, N);
                }
                case kI16: {
                    using R = to_ctype_t<kI16>;
                    using KernOp = OpIdentity<R>;
                    return _cast_runner<T, R, KernOp>(tensor_info, N);
                }
                case kI32: {
                    using R = to_ctype_t<kI32>;
                    using KernOp = OpIdentity<R>;
                    return _cast_runner<T, R, KernOp>(tensor_info, N);
                }
                case kI64: {
                    using R = to_ctype_t<kI64>;
                    using KernOp = OpIdentity<R>;
                    return _cast_runner<T, R, KernOp>(tensor_info, N);
                }
                case kF32: {
                    using R = to_ctype_t<kF32>;
                    using KernOp = OpIdentity<R>;
                    return _cast_runner<T, R, KernOp>(tensor_info, N);
                }
                case kF64: {
                    using R = to_ctype_t<kF64>;
                    using KernOp = OpIdentity<R>;
                    return _cast_runner<T, R, KernOp>(tensor_info, N);
                }
                }
                TT_ERROR("Unknown dtype.");
            };

            return {dtype_switch(dtype), dtype, tensor.shape(), tensor.device()};
        },
        tensor.get_storage<StorageCPU>().storage
    );
}

}    // namespace tinytensor::cpu
