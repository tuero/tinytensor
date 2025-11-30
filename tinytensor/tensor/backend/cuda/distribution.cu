// distribution.cu
// Element-wise distribution runner

#include <tt/concepts.h>
#include <tt/exception.h>
#include <tt/random.h>
#include <tt/scalar.h>
#include <tt/tensor.h>

#include "tensor/backend/common/distribution.h"
#include "tensor/backend/cuda/config.cuh"
#include "tensor/backend/cuda/data_types.cuh"
#include "tensor/backend/cuda/distribution.h"
#include "tensor/backend/cuda/kernel/distribution.cuh"
#include "tensor/backend/cuda/kernel_launch.cuh"
#include "tensor/backend/cuda/storage_cuda.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <tuple>
#include <type_traits>
#include <variant>
#include <vector>

namespace tinytensor::cuda {

using namespace tinytensor::common::distribution;
using namespace kernel::distribution;

namespace {
auto create_gen_states(Generator &gen, std::size_t N) -> std::vector<uint64_t> {
    std::vector<uint64_t> states;
    states.reserve(N);
    for (std::size_t i = 0; i < N; ++i) {
        states.push_back(gen());
    }
    return states;
}
}    // namespace

template <DistributionOpT Op, typename... Params>
    requires IsAllOf<Tensor, Params...>
auto _dist_runner(Generator &gen, const Params &...params) -> Tensor {
    std::tuple<Params...> params_cont(params.contiguous()...);
    const int N = std::get<0>(params_cont).numel();
    const auto gen_states = create_gen_states(gen, static_cast<std::size_t>(N));
    const int device_id = std::get<0>(params_cont).device().id;
    return std::visit(
        [&](auto &&param_dev_memory) -> Tensor {
            using DT = std::remove_cvref_t<decltype(param_dev_memory)>;    // Storage type
            using T = template_parameter_t<DT>;                            // Underlying type

            if constexpr (OpProperties<T, Op>::IsSupported) {
                using KernelOp = typename OpFactory<T, Op>::KernelOp;

                // Allocate for result
                auto gen_states_dev_memory = DeviceMemory<uint64_t>::AllocateVec(device_id, gen_states);
                auto res_dev_memory = DeviceMemory<T>::AllocateElements(device_id, static_cast<std::size_t>(N));

                // Get operand data spans
                const auto param_spans = std::apply(
                    [](const auto &...args) {
                        auto make_span = [](auto &&arg) -> DeviceSpan<const T> {
                            const T *data_ptr =
                                std::get<DT>(arg.template get_storage<StorageCUDA>().dev_memory).data_ptr();
                            data_ptr += arg.offset();    // NOLINT(*-pointer-arithmetic)
                            return DeviceSpan<const T>{data_ptr, static_cast<std::size_t>(arg.numel())};
                        };
                        return std::make_tuple(make_span(args)...);
                    },
                    params_cont
                );
                const DeviceSpan<const uint64_t> gen_span{gen_states_dev_memory};
                DataInfo<T> res{DeviceSpan<T>{res_dev_memory}};

                // Call kernel
                std::apply(
                    [&](auto &&...args) {
                        const auto kernel = variadic_param_kernel<T, KernelOp, decltype(args)...>;
                        launch(
                            device_id,
                            kernel,
                            grid_1d(N),
                            block_1d(),
                            gen_span,
                            res,
                            KernelOp{},
                            N,
                            std::forward<decltype(args)>(args)...
                        );
                    },
                    param_spans
                );
                return {
                    std::make_unique<StorageCUDA>(std::move(res_dev_memory)),
                    std::get<0>(params_cont).dtype(),
                    std::get<0>(params_cont).shape(),
                    std::get<0>(params_cont).device()
                };
            } else {
                TT_ERROR("Unrecoverable: Unsupported type.");
            }
        },
        std::get<0>(params_cont).template get_storage<StorageCUDA>().dev_memory
    );
}

template <common::distribution::DistributionOpT Op>
auto dist_runner(Generator &gen, const Tensor &p1) -> Tensor {
    return _dist_runner<Op>(gen, p1);
}
template <common::distribution::DistributionOpT Op>
auto dist_runner(Generator &gen, const Tensor &p1, const Tensor &p2) -> Tensor {
    return _dist_runner<Op>(gen, p1, p2);
}

template <DistributionOpT Op, typename... Params>
    requires IsAllOf<Tensor, Params...>
void _dist_inplace_runner(Tensor &tensor, Generator &gen, const Params &...params) {
    std::tuple<Params...> params_cont(params.contiguous()...);
    const int N = std::get<0>(params_cont).numel();
    const auto gen_states = create_gen_states(gen, static_cast<std::size_t>(N));
    const int device_id = std::get<0>(params_cont).device().id;

    // Create device memory for shape + stride for proper indexing
    const auto shape = MakeDeviceMemory(device_id, tensor.shape());
    const auto stride = MakeDeviceMemory(device_id, tensor.stride());
    return std::visit(
        [&](auto &&dev_memory) {
            using DT = std::remove_cvref_t<decltype(dev_memory)>;    // Storage type
            using T = template_parameter_t<DT>;                      // Underlying type

            if constexpr (OpProperties<T, Op>::IsSupported) {
                using KernelOp = typename OpFactory<T, Op>::KernelOp;

                auto gen_states_dev_memory = DeviceMemory<uint64_t>::AllocateVec(device_id, gen_states);
                const DeviceSpan<const uint64_t> gen_span{gen_states_dev_memory};
                const DataInfo<T> res{dev_memory, shape, stride, tensor.offset()};

                // Get operand data spans
                const auto param_spans = std::apply(
                    [](const auto &...args) {
                        auto make_span = [](auto &&arg) -> DeviceSpan<const T> {
                            const T *data_ptr =
                                std::get<DT>(arg.template get_storage<StorageCUDA>().dev_memory).data_ptr();
                            data_ptr += arg.offset();    // NOLINT(*-pointer-arithmetic)
                            return DeviceSpan<const T>{data_ptr, static_cast<std::size_t>(arg.numel())};
                        };
                        return std::make_tuple(make_span(args)...);
                    },
                    params_cont
                );

                // Call kernel
                std::apply(
                    [&](auto &&...args) {
                        const auto kernel = variadic_param_kernel<T, KernelOp, decltype(args)...>;
                        launch(
                            device_id,
                            kernel,
                            grid_1d(N),
                            block_1d(),
                            gen_span,
                            res,
                            KernelOp{},
                            N,
                            std::forward<decltype(args)>(args)...
                        );
                    },
                    param_spans
                );
            } else {
                TT_ERROR("Unrecoverable: Unsupported type.");
            }
        },
        tensor.template get_storage<StorageCUDA>().dev_memory
    );
}

template <common::distribution::DistributionOpT Op>
void dist_inplace_runner(Tensor &tensor, Generator &gen, const Tensor &p1) {
    return _dist_inplace_runner<Op>(tensor, gen, p1);
}
template <common::distribution::DistributionOpT Op>
void dist_inplace_runner(Tensor &tensor, Generator &gen, const Tensor &p1, const Tensor &p2) {
    return _dist_inplace_runner<Op>(tensor, gen, p1, p2);
}

template Tensor dist_runner<DistributionOpT::uniform_int>(Generator &, const Tensor &, const Tensor &);
template Tensor dist_runner<DistributionOpT::uniform_real>(Generator &, const Tensor &, const Tensor &);
template Tensor dist_runner<DistributionOpT::bernoulli>(Generator &, const Tensor &);
template Tensor dist_runner<DistributionOpT::binomial>(Generator &, const Tensor &, const Tensor &);
template Tensor dist_runner<DistributionOpT::geometric>(Generator &, const Tensor &);
template Tensor dist_runner<DistributionOpT::poisson>(Generator &, const Tensor &);
template Tensor dist_runner<DistributionOpT::exponential>(Generator &, const Tensor &);
template Tensor dist_runner<DistributionOpT::normal>(Generator &, const Tensor &, const Tensor &);
template Tensor dist_runner<DistributionOpT::cauchy>(Generator &, const Tensor &, const Tensor &);
template Tensor dist_runner<DistributionOpT::lognormal>(Generator &, const Tensor &, const Tensor &);
template Tensor dist_runner<DistributionOpT::weibull>(Generator &, const Tensor &, const Tensor &);

template void dist_inplace_runner<DistributionOpT::uniform_int>(Tensor &, Generator &, const Tensor &, const Tensor &);
template void dist_inplace_runner<DistributionOpT::uniform_real>(Tensor &, Generator &, const Tensor &, const Tensor &);
template void dist_inplace_runner<DistributionOpT::bernoulli>(Tensor &, Generator &, const Tensor &);
template void dist_inplace_runner<DistributionOpT::binomial>(Tensor &, Generator &, const Tensor &, const Tensor &);
template void dist_inplace_runner<DistributionOpT::geometric>(Tensor &, Generator &, const Tensor &);
template void dist_inplace_runner<DistributionOpT::poisson>(Tensor &, Generator &, const Tensor &);
template void dist_inplace_runner<DistributionOpT::exponential>(Tensor &, Generator &, const Tensor &);
template void dist_inplace_runner<DistributionOpT::normal>(Tensor &, Generator &, const Tensor &, const Tensor &);
template void dist_inplace_runner<DistributionOpT::cauchy>(Tensor &, Generator &, const Tensor &, const Tensor &);
template void dist_inplace_runner<DistributionOpT::lognormal>(Tensor &, Generator &, const Tensor &, const Tensor &);
template void dist_inplace_runner<DistributionOpT::weibull>(Tensor &, Generator &, const Tensor &, const Tensor &);

}    // namespace tinytensor::cuda
