// distribution.cpp
// Element-wise distribution runner

#include "tensor/backend/cpu/distribution.h"

#include <tt/concepts.h>
#include <tt/random.h>
#include <tt/exception.h>
#include <tt/scalar.h>
#include <tt/tensor.h>

#include "tensor/backend/common/distribution.h"
#include "tensor/backend/common/span.h"
#include "tensor/backend/cpu/data_types.h"
#include "tensor/backend/cpu/kernel/distribution.hpp"
#include "tensor/backend/cpu/storage_cpu.h"

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <type_traits>
#include <variant>
#include <vector>

namespace tinytensor::cpu {

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
auto dist_runner(Generator &gen, const Params &...params) -> Tensor {
    std::tuple<Params...> params_cont(params.contiguous()...);
    const int N = std::get<0>(params_cont).numel();
    const auto gen_states = create_gen_states(gen, static_cast<std::size_t>(N));
    return std::visit(
        [&](auto &&param_storage) -> Tensor {
            using DT = std::remove_cvref_t<decltype(param_storage)>;    // Storage type
            using T = template_parameter_t<DT>;                         // Underlying type

            // Only compile for types which are supported by this op
            if constexpr (OpProperties<T, Op>::IsSupported) {
                using KernelOp = typename OpFactory<T, Op>::KernelOp;

                // Allocate for result
                std::vector<T> result(static_cast<std::size_t>(N));

                // Get operand data infos
                const auto param_data_infos = std::apply(
                    [](const auto &...args) {
                        auto make_span = [](auto &&arg) -> HostSpan<const T> {
                            const T *data_ptr = std::get<DT>(arg.template get_storage<StorageCPU>().storage).data();
                            data_ptr += arg.offset();    // NOLINT(*-pointer-arithmetic)
                            return HostSpan<const T>{data_ptr, static_cast<std::size_t>(arg.numel())};
                        };
                        return std::make_tuple(make_span(args)...);
                    },
                    params_cont
                );
                const HostSpan<const uint64_t> gen_span{gen_states};
                DataInfo<T> r{result};

                // Call kernel
                std::apply(
                    [&](const auto &...args) { variadic_param_kernel(gen_span, r, KernelOp{}, N, args...); },
                    param_data_infos
                );

                return {
                    std::make_unique<StorageCPU>(std::move(result)),
                    std::get<0>(params_cont).dtype(),
                    std::get<0>(params_cont).shape(),
                    std::get<0>(params_cont).device()
                };
            } else {
                TT_ERROR("Unrecoverable: Unsupported type.");
            }
        },
        std::get<0>(params_cont).template get_storage<StorageCPU>().storage
    );
}

template <DistributionOpT Op, typename... Params>
    requires IsAllOf<Tensor, Params...>
void dist_inplace_runner(Tensor &tensor, Generator &gen, const Params &...params) {
    std::tuple<Params...> params_cont(params.contiguous()...);
    const int N = std::get<0>(params_cont).numel();
    const auto gen_states = create_gen_states(gen, static_cast<std::size_t>(N));
    const bool is_contiguous = tensor.is_contiguous();
    std::visit(
        [&](auto &&tensor_storage) {
            using DT = std::remove_cvref_t<decltype(tensor_storage)>;    // Storage type
            using T = template_parameter_t<DT>;                          // Underlying type

            // Only compile for types which are supported by this op
            if constexpr (OpProperties<T, Op>::IsSupported) {
                using KernelOp = typename OpFactory<T, Op>::KernelOp;

                // Get operand data infos
                const auto param_data_infos = std::apply(
                    [](const auto &...args) {
                        auto make_span = [](auto &&arg) -> HostSpan<const T> {
                            const T *data_ptr = std::get<DT>(arg.template get_storage<StorageCPU>().storage).data();
                            data_ptr += arg.offset();    // NOLINT(*-pointer-arithmetic)
                            return HostSpan<const T>{data_ptr, static_cast<std::size_t>(arg.numel())};
                        };
                        return std::make_tuple(make_span(args)...);
                    },
                    params_cont
                );
                const HostSpan<const uint64_t> gen_span{gen_states};

                const HostSpan<T> a{tensor_storage};
                const auto shape = is_contiguous ? HostSpan<const int>{} : HostSpan<const int>(tensor.shape());
                const auto stride = is_contiguous ? HostSpan<const int>{} : HostSpan<const int>(tensor.stride());
                const DataInfo<T> res{a, shape, stride, tensor.offset()};

                // Call kernel
                std::apply(
                    [&](const auto &...args) { variadic_param_kernel(gen_span, res, KernelOp{}, N, args...); },
                    param_data_infos
                );
            } else {
                TT_ERROR("Unrecoverable: Unsupported type.");
            }
        },
        tensor.template get_storage<StorageCPU>().storage
    );
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

}    // namespace tinytensor::cpu
