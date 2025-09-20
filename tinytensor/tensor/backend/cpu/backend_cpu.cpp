// backend_cpu.cpp
// CPU backend

#include "tensor/backend/cpu/backend_cpu.h"

#include <tt/concepts.h>
#include <tt/exception.h>
#include <tt/random.h>
#include <tt/scalar.h>
#include <tt/tensor.h>

#include "tensor/backend/common/binary.h"
#include "tensor/backend/common/dispatch.h"
#include "tensor/backend/common/distribution.h"
#include "tensor/backend/common/reduce.h"
#include "tensor/backend/common/span.h"
#include "tensor/backend/common/unary.h"
#include "tensor/backend/cpu/assign.h"
#include "tensor/backend/cpu/binary.h"
#include "tensor/backend/cpu/cast.h"
#include "tensor/backend/cpu/clamp.h"
#include "tensor/backend/cpu/conv.h"
#include "tensor/backend/cpu/distribution.h"
#include "tensor/backend/cpu/index.h"
#include "tensor/backend/cpu/matmul.h"
#include "tensor/backend/cpu/misc.h"
#include "tensor/backend/cpu/reduce.h"
#include "tensor/backend/cpu/storage_cpu.h"
#include "tensor/backend/cpu/unary.h"
#include "tensor/print.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <format>
#include <memory>
#include <numeric>
#include <optional>
#include <ostream>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace tinytensor::cpu {

using namespace common::binary;
using namespace common::distribution;
using namespace common::reduce;
using namespace common::unary;

auto BackendCPU::from_vec(const std::vector<bool> &data, [[maybe_unused]] int device_id) const -> StoragePtr {
    std::vector<uint8_t> _data(data.size());
    std::ranges::transform(data, _data.begin(), [&](bool value) { return static_cast<uint8_t>(value); });
    return from_vec(_data, device_id);
}
auto BackendCPU::from_vec(const std::vector<kU8CType> &data, [[maybe_unused]] int device_id) const -> StoragePtr {
    return std::make_unique<StorageCPU>(data);
}
auto BackendCPU::from_vec(std::vector<kU8CType> &&data, [[maybe_unused]] int device_id) const -> StoragePtr {
    return std::make_unique<StorageCPU>(std::move(data));
}
auto BackendCPU::from_vec(const std::vector<kI16CType> &data, [[maybe_unused]] int device_id) const -> StoragePtr {
    return std::make_unique<StorageCPU>(data);
}
auto BackendCPU::from_vec(std::vector<kI16CType> &&data, [[maybe_unused]] int device_id) const -> StoragePtr {
    return std::make_unique<StorageCPU>(std::move(data));
}
auto BackendCPU::from_vec(const std::vector<kI32CType> &data, [[maybe_unused]] int device_id) const -> StoragePtr {
    return std::make_unique<StorageCPU>(data);
}
auto BackendCPU::from_vec(std::vector<kI32CType> &&data, [[maybe_unused]] int device_id) const -> StoragePtr {
    return std::make_unique<StorageCPU>(std::move(data));
}
auto BackendCPU::from_vec(const std::vector<kI64CType> &data, [[maybe_unused]] int device_id) const -> StoragePtr {
    return std::make_unique<StorageCPU>(data);
}
auto BackendCPU::from_vec(std::vector<kI64CType> &&data, [[maybe_unused]] int device_id) const -> StoragePtr {
    return std::make_unique<StorageCPU>(std::move(data));
}
auto BackendCPU::from_vec(const std::vector<kF32CType> &data, [[maybe_unused]] int device_id) const -> StoragePtr {
    return std::make_unique<StorageCPU>(data);
}
auto BackendCPU::from_vec(std::vector<kF32CType> &&data, [[maybe_unused]] int device_id) const -> StoragePtr {
    return std::make_unique<StorageCPU>(std::move(data));
}
auto BackendCPU::from_vec(const std::vector<kF64CType> &data, [[maybe_unused]] int device_id) const -> StoragePtr {
    return std::make_unique<StorageCPU>(data);
}
auto BackendCPU::from_vec(std::vector<kF64CType> &&data, [[maybe_unused]] int device_id) const -> StoragePtr {
    return std::make_unique<StorageCPU>(std::move(data));
}
auto BackendCPU::from_scalar(const Scalar scalar, [[maybe_unused]] int device_id) const -> StoragePtr {
    return DISPATCH_ALL_TYPES(scalar.dtype(), "BackendCPU::from_scalar", [&]() {
        return std::make_unique<StorageCPU>(std::vector<scalar_t>{scalar.to<scalar_t>()});
    });
}
auto BackendCPU::full(const Scalar &value, std::size_t N, [[maybe_unused]] int device_id) const -> StoragePtr {
    return DISPATCH_ALL_TYPES(value.dtype(), "BackendCPU::full", [&]() {
        return std::make_unique<StorageCPU>(std::vector<scalar_t>(N, value.to<scalar_t>()));
    });
}
auto BackendCPU::arange(std::size_t N, ScalarType dtype, [[maybe_unused]] int device_id) const -> StoragePtr {
    return DISPATCH_ALL_TYPES(dtype, "BackendCPU::arange", [&]() {
        std::vector<scalar_t> data(N, 0);
        std::iota(data.begin(), data.end(), 0);
        return std::make_unique<StorageCPU>(std::move(data));
    });
}

// NOLINTNEXTLINE(*-macro-usage)
#define DECLARE_TO_VEC(TYPE)                                                              \
    void BackendCPU::to_vec(const Tensor &tensor, std::vector<TYPE> &data_out) const {    \
        std::visit(                                                                       \
            [&](auto &&tensor_storage) {                                                  \
                using DT = std::remove_cvref_t<decltype(tensor_storage)>;                 \
                using T = template_parameter_t<DT>;                                       \
                if constexpr (std::is_same_v<TYPE, T>) {                                  \
                    data_out = tensor_storage;                                            \
                } else {                                                                  \
                    TT_EXCEPTION(                                                         \
                        std::format(                                                      \
                            "Scalar type of tensor {:s} does not match to_vec type {:s}", \
                            tensor.dtype(),                                               \
                            to_scalar<TYPE>::type                                         \
                        )                                                                 \
                    );                                                                    \
                }                                                                         \
            },                                                                            \
            tensor.get_storage<StorageCPU>().storage                                      \
        );                                                                                \
    }

DECLARE_TO_VEC(kU8CType);
DECLARE_TO_VEC(kI16CType);
DECLARE_TO_VEC(kI32CType);
DECLARE_TO_VEC(kI64CType);
DECLARE_TO_VEC(kF32CType);
DECLARE_TO_VEC(kF64CType);
#undef DECLARE_TO_VEC

auto BackendCPU::item(const Tensor &tensor) const -> Scalar {
    return std::visit(
        [&](auto &&a) -> Scalar { return Scalar{a[static_cast<std::size_t>(tensor.offset())]}; },
        tensor.get_storage<StorageCPU>().storage
    );
}

auto BackendCPU::data_ptr(const Tensor &tensor) const -> uintptr_t {
    return std::visit(
        // NOLINTNEXTLINE(*-reinterpret-cast)
        [&](auto &&a) -> uintptr_t { return reinterpret_cast<uintptr_t>((void *)a.data()); },
        tensor.get_storage<StorageCPU>().storage
    );
}

// Indexing
auto BackendCPU::index_mask(const Tensor &input, const Tensor &mask, int N_mask) const -> Tensor {
    return index_mask_runner(input, mask, N_mask);
}
auto BackendCPU::index_indices(const Tensor &input, const Tensor &indices) const -> Tensor {
    return index_indices_runner(input, indices);
}
void BackendCPU::index_put_mask(Tensor &input, const Tensor &values, const Tensor &mask) const {
    index_put_mask_runner(input, values, mask);
}
void BackendCPU::index_put_indices(Tensor &input, const Tensor &values, const Tensor &indices) const {
    index_put_indices_runner(input, values, indices);
}
auto BackendCPU::gather(const Tensor &input, const Tensor &indices, int idx) const -> Tensor {
    return gather_runner(input, indices, idx);
}

auto BackendCPU::to(const Tensor &tensor, ScalarType dtype) const -> Tensor {
    return cast_runner(tensor, dtype);
}

auto BackendCPU::print(std::ostream &os, const Tensor &tensor) const -> std::ostream & {
    bool fixed_formatting = true;

    // At least a single non-zero finite value exists
    std::optional<Tensor> abs_non_zero = std::nullopt;
    const auto finite_non_zero_mask = isfinite(tensor) && (tensor != 0);
    if (finite_non_zero_mask.any()) {
        abs_non_zero = tensor[finite_non_zero_mask];
    }
    return std::visit(
        [&](auto &&tensor_storage) -> std::ostream & {
            using DT = std::remove_cvref_t<decltype(tensor_storage)>;    // Storage type
            using T = template_parameter_t<DT>;                          // underlying type
            if (abs_non_zero && abs_non_zero->numel() > 0) {
                const auto min_val = min(*abs_non_zero).item<double>();
                const auto max_val = max(*abs_non_zero).item<double>();
                fixed_formatting = (min_val > detail::SUPPRESS_MIN) && (max_val < detail::SUPPRESS_MAX);
            }
            const HostSpan<const T> a{tensor_storage};
            return (
                print_data<T>(
                    a,
                    tensor.shape(),
                    tensor.stride(),
                    tensor.offset(),
                    fixed_formatting,
                    tensor.dtype() == kBool,
                    os
                )
                << ", CPU " << tensor.dtype()
            );
        },
        tensor.get_storage<StorageCPU>().storage
    );
}

void BackendCPU::assign(Tensor &lhs, const Tensor &rhs) const {
    assign_runner(lhs, rhs);
}

// ------------------------------------------------
// Tensor Creation - Distributions
// ------------------------------------------------
auto BackendCPU::uniform_int(const Tensor &low, const Tensor &high, Generator &gen) const -> Tensor {
    return dist_runner<DistributionOpT::uniform_int>(gen, low, high);
}
auto BackendCPU::uniform_real(const Tensor &low, const Tensor &high, Generator &gen) const -> Tensor {
    return dist_runner<DistributionOpT::uniform_real>(gen, low, high);
}
auto BackendCPU::bernoulli(const Tensor &p, Generator &gen) const -> Tensor {
    return dist_runner<DistributionOpT::bernoulli>(gen, p);
}
auto BackendCPU::binomial(const Tensor &p, const Tensor &num_draws, Generator &gen) const -> Tensor {
    return dist_runner<DistributionOpT::binomial>(gen, p, num_draws);
}
auto BackendCPU::geometric(const Tensor &p, Generator &gen) const -> Tensor {
    return dist_runner<DistributionOpT::geometric>(gen, p);
}
auto BackendCPU::poisson(const Tensor &lambda, Generator &gen) const -> Tensor {
    return dist_runner<DistributionOpT::poisson>(gen, lambda);
}
auto BackendCPU::exponential(const Tensor &lambda, Generator &gen) const -> Tensor {
    return dist_runner<DistributionOpT::exponential>(gen, lambda);
}
auto BackendCPU::normal(const Tensor &mu, const Tensor &std, Generator &gen) const -> Tensor {
    return dist_runner<DistributionOpT::normal>(gen, mu, std);
}
auto BackendCPU::cauchy(const Tensor &loc, const Tensor &scale, Generator &gen) const -> Tensor {
    return dist_runner<DistributionOpT::cauchy>(gen, loc, scale);
}
auto BackendCPU::lognormal(const Tensor &mu, const Tensor &std, Generator &gen) const -> Tensor {
    return dist_runner<DistributionOpT::lognormal>(gen, mu, std);
}
auto BackendCPU::weibull(const Tensor &scale, const Tensor &shape, Generator &gen) const -> Tensor {
    return dist_runner<DistributionOpT::weibull>(gen, scale, shape);
}

void BackendCPU::uniform_int_(Tensor &tensor, const Tensor &low, const Tensor &high, Generator &gen) const {
    dist_inplace_runner<DistributionOpT::uniform_int>(tensor, gen, low, high);
}
void BackendCPU::uniform_real_(Tensor &tensor, const Tensor &low, const Tensor &high, Generator &gen) const {
    dist_inplace_runner<DistributionOpT::uniform_real>(tensor, gen, low, high);
}
void BackendCPU::bernoulli_(Tensor &tensor, const Tensor &p, Generator &gen) const {
    dist_inplace_runner<DistributionOpT::bernoulli>(tensor, gen, p);
}
void BackendCPU::binomial_(Tensor &tensor, const Tensor &p, const Tensor &num_draws, Generator &gen) const {
    dist_inplace_runner<DistributionOpT::binomial>(tensor, gen, p, num_draws);
}
void BackendCPU::geometric_(Tensor &tensor, const Tensor &p, Generator &gen) const {
    dist_inplace_runner<DistributionOpT::geometric>(tensor, gen, p);
}
void BackendCPU::poisson_(Tensor &tensor, const Tensor &lambda, Generator &gen) const {
    dist_inplace_runner<DistributionOpT::poisson>(tensor, gen, lambda);
}
void BackendCPU::exponential_(Tensor &tensor, const Tensor &lambda, Generator &gen) const {
    dist_inplace_runner<DistributionOpT::exponential>(tensor, gen, lambda);
}
void BackendCPU::normal_(Tensor &tensor, const Tensor &mu, const Tensor &std, Generator &gen) const {
    dist_inplace_runner<DistributionOpT::normal>(tensor, gen, mu, std);
}
void BackendCPU::cauchy_(Tensor &tensor, const Tensor &loc, const Tensor &scale, Generator &gen) const {
    dist_inplace_runner<DistributionOpT::cauchy>(tensor, gen, loc, scale);
}
void BackendCPU::lognormal_(Tensor &tensor, const Tensor &mu, const Tensor &std, Generator &gen) const {
    dist_inplace_runner<DistributionOpT::lognormal>(tensor, gen, mu, std);
}
void BackendCPU::weibull_(Tensor &tensor, const Tensor &scale, const Tensor &shape, Generator &gen) const {
    dist_inplace_runner<DistributionOpT::weibull>(tensor, gen, scale, shape);
}

// ------------------------------------------------
// Binary Operators
// ------------------------------------------------
auto BackendCPU::eq(const Tensor &lhs, const Tensor &rhs) const -> Tensor {
    return binary_runner<BinaryOpT::equal>(lhs, rhs);
}
auto BackendCPU::ne(const Tensor &lhs, const Tensor &rhs) const -> Tensor {
    return binary_runner<BinaryOpT::not_equal>(lhs, rhs);
}
auto BackendCPU::lt(const Tensor &lhs, const Tensor &rhs) const -> Tensor {
    return binary_runner<BinaryOpT::less_than>(lhs, rhs);
}
auto BackendCPU::le(const Tensor &lhs, const Tensor &rhs) const -> Tensor {
    return binary_runner<BinaryOpT::less_than_eq>(lhs, rhs);
}
auto BackendCPU::gt(const Tensor &lhs, const Tensor &rhs) const -> Tensor {
    return binary_runner<BinaryOpT::greater_than>(lhs, rhs);
}
auto BackendCPU::ge(const Tensor &lhs, const Tensor &rhs) const -> Tensor {
    return binary_runner<BinaryOpT::greater_than_eq>(lhs, rhs);
}
auto BackendCPU::logical_or(const Tensor &lhs, const Tensor &rhs) const -> Tensor {
    return binary_runner<BinaryOpT::logical_or>(lhs, rhs);
}
auto BackendCPU::logical_and(const Tensor &lhs, const Tensor &rhs) const -> Tensor {
    return binary_runner<BinaryOpT::logical_and>(lhs, rhs);
}
auto BackendCPU::bitwise_or(const Tensor &lhs, const Tensor &rhs) const -> Tensor {
    return binary_runner<BinaryOpT::bitwise_or>(lhs, rhs);
}
auto BackendCPU::bitwise_and(const Tensor &lhs, const Tensor &rhs) const -> Tensor {
    return binary_runner<BinaryOpT::bitwise_and>(lhs, rhs);
}
auto BackendCPU::bitwise_xor(const Tensor &lhs, const Tensor &rhs) const -> Tensor {
    return binary_runner<BinaryOpT::bitwise_xor>(lhs, rhs);
}
auto BackendCPU::bitwise_left_shift(const Tensor &lhs, const Tensor &rhs) const -> Tensor {
    return binary_runner<BinaryOpT::bitwise_left_shift>(lhs, rhs);
}
auto BackendCPU::bitwise_right_shift(const Tensor &lhs, const Tensor &rhs) const -> Tensor {
    return binary_runner<BinaryOpT::bitwise_right_shift>(lhs, rhs);
}
auto BackendCPU::modulo(const Tensor &lhs, const Tensor &rhs) const -> Tensor {
    return binary_runner<BinaryOpT::modulo>(lhs, rhs);
}
auto BackendCPU::add(const Tensor &lhs, const Tensor &rhs) const -> Tensor {
    return binary_runner<BinaryOpT::add>(lhs, rhs);
}
auto BackendCPU::sub(const Tensor &lhs, const Tensor &rhs) const -> Tensor {
    return binary_runner<BinaryOpT::subtract>(lhs, rhs);
}
auto BackendCPU::mul(const Tensor &lhs, const Tensor &rhs) const -> Tensor {
    return binary_runner<BinaryOpT::multiply>(lhs, rhs);
}
auto BackendCPU::div(const Tensor &lhs, const Tensor &rhs) const -> Tensor {
    return binary_runner<BinaryOpT::divide>(lhs, rhs);
}
auto BackendCPU::maximum(const Tensor &lhs, const Tensor &rhs) const -> Tensor {
    return binary_runner<BinaryOpT::maximum>(lhs, rhs);
}
auto BackendCPU::minimum(const Tensor &lhs, const Tensor &rhs) const -> Tensor {
    return binary_runner<BinaryOpT::minimum>(lhs, rhs);
}
auto BackendCPU::pow(const Tensor &lhs, const Tensor &rhs) const -> Tensor {
    return binary_runner<BinaryOpT::pow>(lhs, rhs);
}
auto BackendCPU::batched_matmul(const Tensor &lhs, const Tensor &rhs) const -> Tensor {
    return batched_matmul_runner(lhs, rhs);
}

void BackendCPU::add_inplace(Tensor &lhs, const Tensor &rhs) const {
    binary_inplace_runner<BinaryOpT::add>(lhs, rhs);
}
void BackendCPU::sub_inplace(Tensor &lhs, const Tensor &rhs) const {
    binary_inplace_runner<BinaryOpT::subtract>(lhs, rhs);
}
void BackendCPU::mul_inplace(Tensor &lhs, const Tensor &rhs) const {
    binary_inplace_runner<BinaryOpT::multiply>(lhs, rhs);
}
void BackendCPU::div_inplace(Tensor &lhs, const Tensor &rhs) const {
    binary_inplace_runner<BinaryOpT::divide>(lhs, rhs);
}

// ------------------------------------------------
// Reduction operations
// ------------------------------------------------
auto BackendCPU::min(const Tensor &tensor) const -> Tensor {
    return reduce_all_runner<ReduceOpT::min>(tensor);
}
auto BackendCPU::min(const Tensor &tensor, int dim) const -> Tensor {
    return reduce_dim_runner<ReduceOpT::min>(tensor, dim);
}

auto BackendCPU::argmin(const Tensor &tensor) const -> Tensor {
    return reduce_all_runner<ReduceOpT::argmin>(tensor);
}
auto BackendCPU::argmin(const Tensor &tensor, int dim) const -> Tensor {
    return reduce_dim_runner<ReduceOpT::argmin>(tensor, dim);
}

auto BackendCPU::max(const Tensor &tensor) const -> Tensor {
    return reduce_all_runner<ReduceOpT::max>(tensor);
}
auto BackendCPU::max(const Tensor &tensor, int dim) const -> Tensor {
    return reduce_dim_runner<ReduceOpT::max>(tensor, dim);
}

auto BackendCPU::argmax(const Tensor &tensor) const -> Tensor {
    return reduce_all_runner<ReduceOpT::argmax>(tensor);
}
auto BackendCPU::argmax(const Tensor &tensor, int dim) const -> Tensor {
    return reduce_dim_runner<ReduceOpT::argmax>(tensor, dim);
}

auto BackendCPU::sum(const Tensor &tensor) const -> Tensor {
    return reduce_all_runner<ReduceOpT::sum>(tensor);
}
auto BackendCPU::sum(const Tensor &tensor, int dim) const -> Tensor {
    return reduce_dim_runner<ReduceOpT::sum>(tensor, dim);
}

auto BackendCPU::all(const Tensor &tensor) const -> bool {
    return reduce_all_runner<ReduceOpT::all>(tensor).item<uint8_t>();
}
auto BackendCPU::all(const Tensor &tensor, int dim) const -> Tensor {
    return reduce_dim_runner<ReduceOpT::all>(tensor, dim);
}

auto BackendCPU::any(const Tensor &tensor) const -> bool {
    return reduce_all_runner<ReduceOpT::any>(tensor).item<uint8_t>();
}
auto BackendCPU::any(const Tensor &tensor, int dim) const -> Tensor {
    return reduce_dim_runner<ReduceOpT::any>(tensor, dim);
}

// ------------------------------------------------
// Unary operations
// ------------------------------------------------
auto BackendCPU::identity(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::identity>(tensor);
}
void BackendCPU::identity_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::identity>(tensor);
}

auto BackendCPU::negate(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::negate>(tensor);
}
void BackendCPU::negate_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::negate>(tensor);
}

auto BackendCPU::logical_not(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::logical_not>(tensor);
}
void BackendCPU::logical_not_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::logical_not>(tensor);
}

auto BackendCPU::abs(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::abs>(tensor);
}
void BackendCPU::abs_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::abs>(tensor);
}

auto BackendCPU::sign(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::sign>(tensor);
}
void BackendCPU::sign_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::sign>(tensor);
}

auto BackendCPU::log(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::log>(tensor);
}
void BackendCPU::log_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::log>(tensor);
}

auto BackendCPU::log10(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::log10>(tensor);
}
void BackendCPU::log10_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::log10>(tensor);
}

auto BackendCPU::log2(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::log2>(tensor);
}
void BackendCPU::log2_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::log2>(tensor);
}

auto BackendCPU::log1p(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::log1p>(tensor);
}
void BackendCPU::log1p_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::log1p>(tensor);
}

auto BackendCPU::exp(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::exp>(tensor);
}
void BackendCPU::exp_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::exp>(tensor);
}

auto BackendCPU::exp2(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::exp2>(tensor);
}
void BackendCPU::exp2_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::exp2>(tensor);
}

auto BackendCPU::expm1(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::expm1>(tensor);
}
void BackendCPU::expm1_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::expm1>(tensor);
}

auto BackendCPU::sqrt(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::sqrt>(tensor);
}
void BackendCPU::sqrt_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::sqrt>(tensor);
}

auto BackendCPU::sin(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::sin>(tensor);
}
void BackendCPU::sin_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::sin>(tensor);
}

auto BackendCPU::cos(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::cos>(tensor);
}
void BackendCPU::cos_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::cos>(tensor);
}

auto BackendCPU::tan(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::tan>(tensor);
}
void BackendCPU::tan_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::tan>(tensor);
}

auto BackendCPU::asin(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::asin>(tensor);
}
void BackendCPU::asin_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::asin>(tensor);
}

auto BackendCPU::acos(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::acos>(tensor);
}
void BackendCPU::acos_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::acos>(tensor);
}

auto BackendCPU::atan(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::atan>(tensor);
}
void BackendCPU::atan_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::atan>(tensor);
}

auto BackendCPU::sinh(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::sinh>(tensor);
}
void BackendCPU::sinh_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::sinh>(tensor);
}

auto BackendCPU::cosh(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::cosh>(tensor);
}
void BackendCPU::cosh_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::cosh>(tensor);
}

auto BackendCPU::tanh(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::tanh>(tensor);
}
void BackendCPU::tanh_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::tanh>(tensor);
}

auto BackendCPU::asinh(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::asinh>(tensor);
}
void BackendCPU::asinh_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::asinh>(tensor);
}

auto BackendCPU::acosh(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::acosh>(tensor);
}
void BackendCPU::acosh_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::acosh>(tensor);
}

auto BackendCPU::atanh(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::atanh>(tensor);
}
void BackendCPU::atanh_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::atanh>(tensor);
}

auto BackendCPU::erf(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::erf>(tensor);
}
void BackendCPU::erf_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::erf>(tensor);
}

auto BackendCPU::erfc(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::erfc>(tensor);
}
void BackendCPU::erfc_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::erfc>(tensor);
}

auto BackendCPU::tgamma(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::tgamma>(tensor);
}
void BackendCPU::tgamma_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::tgamma>(tensor);
}

auto BackendCPU::lgamma(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::lgamma>(tensor);
}
void BackendCPU::lgamma_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::lgamma>(tensor);
}

auto BackendCPU::digamma(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::digamma>(tensor);
}
void BackendCPU::digamma_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::digamma>(tensor);
}

auto BackendCPU::ceil(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::ceil>(tensor);
}
void BackendCPU::ceil_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::ceil>(tensor);
}

auto BackendCPU::floor(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::floor>(tensor);
}
void BackendCPU::floor_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::floor>(tensor);
}

auto BackendCPU::round(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::round>(tensor);
}
void BackendCPU::round_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::round>(tensor);
}

auto BackendCPU::isinf(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::isinf>(tensor);
}
auto BackendCPU::isnan(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::isnan>(tensor);
}
auto BackendCPU::isfinite(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::isfinite>(tensor);
}

// ------------------------------------------------
// Activation functions
// ------------------------------------------------
auto BackendCPU::sigmoid(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::sigmoid>(tensor);
}
void BackendCPU::sigmoid_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::sigmoid>(tensor);
}

auto BackendCPU::log_sigmoid(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::log_sigmoid>(tensor);
}
void BackendCPU::log_sigmoid_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::log_sigmoid>(tensor);
}

auto BackendCPU::hardsigmoid(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::hardsigmoid>(tensor);
}
void BackendCPU::hardsigmoid_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::hardsigmoid>(tensor);
}

auto BackendCPU::softplus(const Tensor &tensor, double beta, double threshold) const -> Tensor {
    return unary_runner<UnaryOpT::softplus>(tensor, beta, threshold);
}
void BackendCPU::softplus_(Tensor &tensor, double beta, double threshold) const {
    unary_runner_inplace<UnaryOpT::softplus>(tensor, beta, threshold);
}

auto BackendCPU::relu(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::relu>(tensor);
}
void BackendCPU::relu_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::relu>(tensor);
}

auto BackendCPU::relu6(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::relu6>(tensor);
}
void BackendCPU::relu6_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::relu6>(tensor);
}

auto BackendCPU::leaky_relu(const Tensor &tensor, double negative_slope) const -> Tensor {
    return unary_runner<UnaryOpT::leaky_relu>(tensor, negative_slope);
}
void BackendCPU::leaky_relu_(Tensor &tensor, double negative_slope) const {
    unary_runner_inplace<UnaryOpT::leaky_relu>(tensor, negative_slope);
}

auto BackendCPU::elu(const Tensor &tensor, double alpha) const -> Tensor {
    return unary_runner<UnaryOpT::elu>(tensor, alpha);
}
void BackendCPU::elu_(Tensor &tensor, double alpha) const {
    unary_runner_inplace<UnaryOpT::elu>(tensor, alpha);
}

auto BackendCPU::selu(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::selu>(tensor);
}
void BackendCPU::selu_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::selu>(tensor);
}

auto BackendCPU::silu(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::silu>(tensor);
}
void BackendCPU::silu_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::silu>(tensor);
}

auto BackendCPU::hardtanh(const Tensor &tensor, double min, double max) const -> Tensor {
    return unary_runner<UnaryOpT::hardtanh>(tensor, min, max);
}
void BackendCPU::hardtanh_(Tensor &tensor, double min, double max) const {
    unary_runner_inplace<UnaryOpT::hardtanh>(tensor, min, max);
}

auto BackendCPU::softsign(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::softsign>(tensor);
}
void BackendCPU::softsign_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::softsign>(tensor);
}

auto BackendCPU::softmax(const Tensor &tensor, int dim) const -> Tensor {
    auto result = tensor.clone();
    softmax_(result, dim);
    return result;
}
void BackendCPU::softmax_(Tensor &tensor, int dim) const {
    const auto m = tensor.max(dim, true).expand(tensor.shape());
    tensor -= m;
    tensor.exp_();
    const auto denom = tensor.sum(dim, true).expand(tensor.shape());
    tensor /= denom;
}

auto BackendCPU::log_softmax(const Tensor &tensor, int dim) const -> Tensor {
    auto result = tensor.clone();
    log_softmax_(result, dim);
    return result;
}
void BackendCPU::log_softmax_(Tensor &tensor, int dim) const {
    const auto m = tensor.max(dim, true).expand(tensor.shape());
    const auto logsumexp = log(exp(tensor - m).sum(dim, true).expand(tensor.shape()));
    tensor -= m;
    tensor -= logsumexp;
}

// ------------------------------------------------
// Util/misc
// ------------------------------------------------
auto BackendCPU::where(const Tensor &cond, const Tensor &lhs, const Tensor &rhs) const -> Tensor {
    return where_runner(cond, lhs, rhs);
}

void BackendCPU::clamp_(Tensor &tensor, const Tensor &min, const Tensor &max) const {
    clamp_inplace_runner(tensor, min, max);
}

auto BackendCPU::conv2d(
    const Tensor &input,
    const Tensor &weight,
    const std::optional<Tensor> &bias,
    int stride,
    int padding
) const -> Tensor {
    return batched_conv2d_forward_runner(input, weight, bias, stride, padding);
}
auto BackendCPU::conv2d_backward(
    const Tensor &grad_output,
    const Tensor &input,
    const Tensor &weight,
    const std::optional<Tensor> &bias,
    int stride,
    int padding
) const -> std::tuple<Tensor, Tensor, std::optional<Tensor>> {
    return batched_conv2d_backward_runner(grad_output, input, weight, bias, stride, padding);
}

auto BackendCPU::max_pool2d(const Tensor &input, int kernel_size, int stride, int padding) const -> Tensor {
    return batched_pool2d_forward_runner<ReduceOpT::max>(input, kernel_size, stride, padding);
}
auto BackendCPU::max_pool2d_backward(
    const Tensor &grad_output,
    const Tensor &input,
    const Tensor &result,
    int kernel_size,
    int stride,
    int padding
) const -> Tensor {
    return batched_pool2d_backward_runner<ReduceOpT::max>(grad_output, input, result, kernel_size, stride, padding);
}

auto BackendCPU::min_pool2d(const Tensor &input, int kernel_size, int stride, int padding) const -> Tensor {
    return batched_pool2d_forward_runner<ReduceOpT::min>(input, kernel_size, stride, padding);
}
auto BackendCPU::min_pool2d_backward(
    const Tensor &grad_output,
    const Tensor &input,
    const Tensor &result,
    int kernel_size,
    int stride,
    int padding
) const -> Tensor {
    return batched_pool2d_backward_runner<ReduceOpT::min>(grad_output, input, result, kernel_size, stride, padding);
}

auto BackendCPU::avg_pool2d(const Tensor &input, int kernel_size, int stride, int padding) const -> Tensor {
    return batched_pool2d_forward_runner<ReduceOpT::mean>(input, kernel_size, stride, padding);
}
auto BackendCPU::avg_pool2d_backward(
    const Tensor &grad_output,
    const Tensor &input,
    const Tensor &result,
    int kernel_size,
    int stride,
    int padding
) const -> Tensor {
    return batched_pool2d_backward_runner<ReduceOpT::mean>(grad_output, input, result, kernel_size, stride, padding);
}

auto BackendCPU::current_memory_allocated([[maybe_unused]] int device_id) const -> uint64_t {
    return StorageCPU::current_bytes_allocated;
}
auto BackendCPU::total_memory_allocated([[maybe_unused]] int device_id) const -> uint64_t {
    return StorageCPU::total_bytes_allocated;
}

}    // namespace tinytensor::cpu
