// backend_cuda.cpp
// CUDA backend

#include "tensor/backend/cuda/backend_cuda.h"

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
#include "tensor/backend/cuda/assign.h"
#include "tensor/backend/cuda/binary.h"
#include "tensor/backend/cuda/cast.h"
#include "tensor/backend/cuda/clamp.h"
#include "tensor/backend/cuda/conv.h"
#include "tensor/backend/cuda/distribution.h"
#include "tensor/backend/cuda/index.h"
#include "tensor/backend/cuda/matmul.h"
#include "tensor/backend/cuda/misc.h"
#include "tensor/backend/cuda/reduce.h"
#include "tensor/backend/cuda/storage_cuda.h"
#include "tensor/backend/cuda/unary.h"
#include "tensor/print.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <format>
#include <iostream>
#include <memory>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace tinytensor::cuda {

using namespace common::binary;
using namespace common::distribution;
using namespace common::reduce;
using namespace common::unary;

// NOLINTNEXTLINE(*-macro-usage)
#define CHECK_DEVICE(device_id)                                                     \
    const auto device_count = get_device_count();                                   \
    if (device_id < 0 || device_id >= device_count) {                               \
        TT_EXCEPTION(                                                               \
            std::format(                                                            \
                "Device id {:d} is out of range. Expected to be between [0, {:d}]", \
                device_id,                                                          \
                device_count - 1                                                    \
            )                                                                       \
        );                                                                          \
    }

auto BackendCUDA::from_vec(const std::vector<bool> &data, int device_id) const -> StoragePtr {
    CHECK_DEVICE(device_id);
    std::vector<uint8_t> _data(data.size());
    std::ranges::transform(data, _data.begin(), [&](bool value) { return static_cast<uint8_t>(value); });
    return from_vec(_data, device_id);
}
auto BackendCUDA::from_vec(const std::vector<kU8CType> &data, int device_id) const -> StoragePtr {
    CHECK_DEVICE(device_id);
    return std::make_unique<StorageCUDA>(device_id, data);
}
auto BackendCUDA::from_vec(std::vector<kU8CType> &&data, int device_id) const -> StoragePtr {
    CHECK_DEVICE(device_id);
    return std::make_unique<StorageCUDA>(device_id, std::move(data));
}
auto BackendCUDA::from_vec(const std::vector<kI16CType> &data, int device_id) const -> StoragePtr {
    CHECK_DEVICE(device_id);
    return std::make_unique<StorageCUDA>(device_id, data);
}
auto BackendCUDA::from_vec(std::vector<kI16CType> &&data, int device_id) const -> StoragePtr {
    CHECK_DEVICE(device_id);
    return std::make_unique<StorageCUDA>(device_id, std::move(data));
}
auto BackendCUDA::from_vec(const std::vector<kI32CType> &data, int device_id) const -> StoragePtr {
    CHECK_DEVICE(device_id);
    return std::make_unique<StorageCUDA>(device_id, data);
}
auto BackendCUDA::from_vec(std::vector<kI32CType> &&data, int device_id) const -> StoragePtr {
    CHECK_DEVICE(device_id);
    return std::make_unique<StorageCUDA>(device_id, std::move(data));
}
auto BackendCUDA::from_vec(const std::vector<kI64CType> &data, int device_id) const -> StoragePtr {
    CHECK_DEVICE(device_id);
    return std::make_unique<StorageCUDA>(device_id, data);
}
auto BackendCUDA::from_vec(std::vector<kI64CType> &&data, int device_id) const -> StoragePtr {
    CHECK_DEVICE(device_id);
    return std::make_unique<StorageCUDA>(device_id, std::move(data));
}
auto BackendCUDA::from_vec(const std::vector<kF32CType> &data, int device_id) const -> StoragePtr {
    CHECK_DEVICE(device_id);
    return std::make_unique<StorageCUDA>(device_id, data);
}
auto BackendCUDA::from_vec(std::vector<kF32CType> &&data, int device_id) const -> StoragePtr {
    CHECK_DEVICE(device_id);
    return std::make_unique<StorageCUDA>(device_id, std::move(data));
}
auto BackendCUDA::from_vec(const std::vector<kF64CType> &data, int device_id) const -> StoragePtr {
    CHECK_DEVICE(device_id);
    return std::make_unique<StorageCUDA>(device_id, data);
}
auto BackendCUDA::from_vec(std::vector<kF64CType> &&data, int device_id) const -> StoragePtr {
    CHECK_DEVICE(device_id);
    return std::make_unique<StorageCUDA>(device_id, std::move(data));
}
auto BackendCUDA::from_scalar(const Scalar scalar, int device_id) const -> StoragePtr {
    CHECK_DEVICE(device_id);
    return DISPATCH_ALL_TYPES(scalar.dtype(), "BackendCUDA::from_scalar", [&]() {
        return std::make_unique<StorageCUDA>(device_id, 1, scalar.to<scalar_t>());
    });
}
auto BackendCUDA::full(const Scalar &value, std::size_t N, int device_id) const -> StoragePtr {
    CHECK_DEVICE(device_id);
    return DISPATCH_ALL_TYPES(value.dtype(), "BackendCUDA::full", [&]() {
        return std::make_unique<StorageCUDA>(device_id, N, value.to<scalar_t>());
    });
}
auto BackendCUDA::arange(std::size_t N, ScalarType dtype, int device_id) const -> StoragePtr {
    CHECK_DEVICE(device_id);
    return DISPATCH_ALL_TYPES(dtype, "BackendCUDA::arange", [&]() {
        return std::make_unique<StorageCUDA>(StorageCUDA::arange<scalar_t>(device_id, N));
    });
}

// NOLINTNEXTLINE(*-macro-usage)
#define DECLARE_TO_VEC(TYPE)                                                              \
    void BackendCUDA::to_vec(const Tensor &tensor, std::vector<TYPE> &data_out) const {   \
        std::visit(                                                                       \
            [&](auto &&dev_memory) {                                                      \
                using DT = std::remove_cvref_t<decltype(dev_memory)>;                     \
                using T = template_parameter_t<DT>;                                       \
                if constexpr (std::is_same_v<TYPE, T>) {                                  \
                    const auto hm = dev_memory.to_vec();                                  \
                    data_out = dev_memory.to_vec();                                       \
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
            tensor.get_storage<StorageCUDA>().dev_memory                                  \
        );                                                                                \
    }

DECLARE_TO_VEC(kU8CType);
DECLARE_TO_VEC(kI16CType);
DECLARE_TO_VEC(kI32CType);
DECLARE_TO_VEC(kI64CType);
DECLARE_TO_VEC(kF32CType);
DECLARE_TO_VEC(kF64CType);
#undef DECLARE_TO_VEC

auto BackendCUDA::item(const Tensor &tensor) const -> Scalar {
    return std::visit(
        [&](auto &&dev_memory) -> Scalar { return Scalar{dev_memory.item(tensor.offset())}; },
        tensor.get_storage<StorageCUDA>().dev_memory
    );
}

auto BackendCUDA::data_ptr(const Tensor &tensor) const -> uintptr_t {
    return std::visit(
        [&](auto &&dev_memory) -> uintptr_t {
            return reinterpret_cast<uintptr_t>((void *)dev_memory.data_ptr());    // NOLINT(*-reinterpret-cast)
        },
        tensor.get_storage<StorageCUDA>().dev_memory
    );
}

// Indexing
auto BackendCUDA::index_mask(const Tensor &input, const Tensor &mask, int N_mask) const -> Tensor {
    return index_mask_runner(input, mask, N_mask);
}
auto BackendCUDA::index_indices(const Tensor &input, const Tensor &indices) const -> Tensor {
    return index_indices_runner(input, indices);
}
void BackendCUDA::index_put_mask(Tensor &input, const Tensor &values, const Tensor &mask) const {
    index_put_mask_runner(input, values, mask);
}
void BackendCUDA::index_put_indices(Tensor &input, const Tensor &values, const Tensor &indices) const {
    index_put_indices_runner(input, values, indices);
}
auto BackendCUDA::gather(const Tensor &input, const Tensor &indices, int idx) const -> Tensor {
    return gather_runner(input, indices, idx);
}

auto BackendCUDA::to(const Tensor &tensor, ScalarType dtype) const -> Tensor {
    return cast_runner(tensor, dtype);
}

auto BackendCUDA::print(std::ostream &os, const Tensor &tensor) const -> std::ostream & {
    bool fixed_formatting = true;

    // At least a single non-zero finite value exists
    std::optional<Tensor> abs_non_zero = std::nullopt;
    const auto finite_non_zero_mask = isfinite(tensor) && (tensor != 0);
    if (finite_non_zero_mask.any()) {
        abs_non_zero = tensor[finite_non_zero_mask];
    }
    return std::visit(
        [&](auto &&dev_memory) -> std::ostream & {
            using DT = std::remove_cvref_t<decltype(dev_memory)>;    // Storage type
            using T = template_parameter_t<DT>;                      // Underlying type

            if (abs_non_zero && abs_non_zero->numel() > 0) {
                const auto min_val = min(*abs_non_zero).item<double>();
                const auto max_val = max(*abs_non_zero).item<double>();
                fixed_formatting = (min_val > detail::SUPPRESS_MIN) && (max_val < detail::SUPPRESS_MAX);
            }
            const auto hm = dev_memory.to_vec();
            return (
                print_data(
                    HostSpan<const T>(hm),
                    tensor.shape(),
                    tensor.stride(),
                    tensor.offset(),
                    fixed_formatting,
                    tensor.dtype() == kBool,
                    os
                )
                << std::format(", CUDA:{:d} {:s}", tensor.device().id, tensor.dtype())
            );
        },
        tensor.get_storage<StorageCUDA>().dev_memory
    );
}

void BackendCUDA::assign(Tensor &lhs, const Tensor &rhs) const {
    assign_runner(lhs, rhs);
}

// ------------------------------------------------
// Tensor Creation - Distributions
// ------------------------------------------------
auto BackendCUDA::uniform_int(const Tensor &low, const Tensor &high, Generator &gen) const -> Tensor {
    return dist_runner<DistributionOpT::uniform_int>(gen, low, high);
}
auto BackendCUDA::uniform_real(const Tensor &low, const Tensor &high, Generator &gen) const -> Tensor {
    return dist_runner<DistributionOpT::uniform_real>(gen, low, high);
}
auto BackendCUDA::bernoulli(const Tensor &p, Generator &gen) const -> Tensor {
    return dist_runner<DistributionOpT::bernoulli>(gen, p);
}
auto BackendCUDA::binomial(const Tensor &p, const Tensor &num_draws, Generator &gen) const -> Tensor {
    return dist_runner<DistributionOpT::binomial>(gen, p, num_draws);
}
auto BackendCUDA::geometric(const Tensor &p, Generator &gen) const -> Tensor {
    return dist_runner<DistributionOpT::geometric>(gen, p);
}
auto BackendCUDA::poisson(const Tensor &lambda, Generator &gen) const -> Tensor {
    return dist_runner<DistributionOpT::poisson>(gen, lambda);
}
auto BackendCUDA::exponential(const Tensor &lambda, Generator &gen) const -> Tensor {
    return dist_runner<DistributionOpT::exponential>(gen, lambda);
}
auto BackendCUDA::normal(const Tensor &mu, const Tensor &std, Generator &gen) const -> Tensor {
    return dist_runner<DistributionOpT::normal>(gen, mu, std);
}
auto BackendCUDA::cauchy(const Tensor &loc, const Tensor &scale, Generator &gen) const -> Tensor {
    return dist_runner<DistributionOpT::cauchy>(gen, loc, scale);
}
auto BackendCUDA::lognormal(const Tensor &mu, const Tensor &std, Generator &gen) const -> Tensor {
    return dist_runner<DistributionOpT::lognormal>(gen, mu, std);
}
auto BackendCUDA::weibull(const Tensor &scale, const Tensor &shape, Generator &gen) const -> Tensor {
    return dist_runner<DistributionOpT::weibull>(gen, scale, shape);
}

void BackendCUDA::uniform_int_(Tensor &tensor, const Tensor &low, const Tensor &high, Generator &gen) const {
    dist_inplace_runner<DistributionOpT::uniform_int>(tensor, gen, low, high);
}
void BackendCUDA::uniform_real_(Tensor &tensor, const Tensor &low, const Tensor &high, Generator &gen) const {
    dist_inplace_runner<DistributionOpT::uniform_real>(tensor, gen, low, high);
}
void BackendCUDA::bernoulli_(Tensor &tensor, const Tensor &p, Generator &gen) const {
    dist_inplace_runner<DistributionOpT::bernoulli>(tensor, gen, p);
}
void BackendCUDA::binomial_(Tensor &tensor, const Tensor &p, const Tensor &num_draws, Generator &gen) const {
    dist_inplace_runner<DistributionOpT::binomial>(tensor, gen, p, num_draws);
}
void BackendCUDA::geometric_(Tensor &tensor, const Tensor &p, Generator &gen) const {
    dist_inplace_runner<DistributionOpT::geometric>(tensor, gen, p);
}
void BackendCUDA::poisson_(Tensor &tensor, const Tensor &lambda, Generator &gen) const {
    dist_inplace_runner<DistributionOpT::poisson>(tensor, gen, lambda);
}
void BackendCUDA::exponential_(Tensor &tensor, const Tensor &lambda, Generator &gen) const {
    dist_inplace_runner<DistributionOpT::exponential>(tensor, gen, lambda);
}
void BackendCUDA::normal_(Tensor &tensor, const Tensor &mu, const Tensor &std, Generator &gen) const {
    dist_inplace_runner<DistributionOpT::normal>(tensor, gen, mu, std);
}
void BackendCUDA::cauchy_(Tensor &tensor, const Tensor &loc, const Tensor &scale, Generator &gen) const {
    dist_inplace_runner<DistributionOpT::cauchy>(tensor, gen, loc, scale);
}
void BackendCUDA::lognormal_(Tensor &tensor, const Tensor &mu, const Tensor &std, Generator &gen) const {
    dist_inplace_runner<DistributionOpT::lognormal>(tensor, gen, mu, std);
}
void BackendCUDA::weibull_(Tensor &tensor, const Tensor &scale, const Tensor &shape, Generator &gen) const {
    dist_inplace_runner<DistributionOpT::weibull>(tensor, gen, scale, shape);
}

// ------------------------------------------------
// Binary Operators
// ------------------------------------------------
auto BackendCUDA::eq(const Tensor &lhs, const Tensor &rhs) const -> Tensor {
    return binary_runner<BinaryOpT::equal>(lhs, rhs);
}
auto BackendCUDA::ne(const Tensor &lhs, const Tensor &rhs) const -> Tensor {
    return binary_runner<BinaryOpT::not_equal>(lhs, rhs);
}
auto BackendCUDA::lt(const Tensor &lhs, const Tensor &rhs) const -> Tensor {
    return binary_runner<BinaryOpT::less_than>(lhs, rhs);
}
auto BackendCUDA::le(const Tensor &lhs, const Tensor &rhs) const -> Tensor {
    return binary_runner<BinaryOpT::less_than_eq>(lhs, rhs);
}
auto BackendCUDA::gt(const Tensor &lhs, const Tensor &rhs) const -> Tensor {
    return binary_runner<BinaryOpT::greater_than>(lhs, rhs);
}
auto BackendCUDA::ge(const Tensor &lhs, const Tensor &rhs) const -> Tensor {
    return binary_runner<BinaryOpT::greater_than_eq>(lhs, rhs);
}
auto BackendCUDA::logical_or(const Tensor &lhs, const Tensor &rhs) const -> Tensor {
    return binary_runner<BinaryOpT::logical_or>(lhs, rhs);
}
auto BackendCUDA::logical_and(const Tensor &lhs, const Tensor &rhs) const -> Tensor {
    return binary_runner<BinaryOpT::logical_and>(lhs, rhs);
}
auto BackendCUDA::bitwise_or(const Tensor &lhs, const Tensor &rhs) const -> Tensor {
    return binary_runner<BinaryOpT::bitwise_or>(lhs, rhs);
}
auto BackendCUDA::bitwise_and(const Tensor &lhs, const Tensor &rhs) const -> Tensor {
    return binary_runner<BinaryOpT::bitwise_and>(lhs, rhs);
}
auto BackendCUDA::bitwise_xor(const Tensor &lhs, const Tensor &rhs) const -> Tensor {
    return binary_runner<BinaryOpT::bitwise_xor>(lhs, rhs);
}
auto BackendCUDA::bitwise_left_shift(const Tensor &lhs, const Tensor &rhs) const -> Tensor {
    return binary_runner<BinaryOpT::bitwise_left_shift>(lhs, rhs);
}
auto BackendCUDA::bitwise_right_shift(const Tensor &lhs, const Tensor &rhs) const -> Tensor {
    return binary_runner<BinaryOpT::bitwise_right_shift>(lhs, rhs);
}
auto BackendCUDA::modulo(const Tensor &lhs, const Tensor &rhs) const -> Tensor {
    return binary_runner<BinaryOpT::modulo>(lhs, rhs);
}
auto BackendCUDA::add(const Tensor &lhs, const Tensor &rhs) const -> Tensor {
    return binary_runner<BinaryOpT::add>(lhs, rhs);
}
auto BackendCUDA::sub(const Tensor &lhs, const Tensor &rhs) const -> Tensor {
    return binary_runner<BinaryOpT::subtract>(lhs, rhs);
}
auto BackendCUDA::mul(const Tensor &lhs, const Tensor &rhs) const -> Tensor {
    return binary_runner<BinaryOpT::multiply>(lhs, rhs);
}
auto BackendCUDA::div(const Tensor &lhs, const Tensor &rhs) const -> Tensor {
    return binary_runner<BinaryOpT::divide>(lhs, rhs);
}
auto BackendCUDA::maximum(const Tensor &lhs, const Tensor &rhs) const -> Tensor {
    return binary_runner<BinaryOpT::maximum>(lhs, rhs);
}
auto BackendCUDA::minimum(const Tensor &lhs, const Tensor &rhs) const -> Tensor {
    return binary_runner<BinaryOpT::minimum>(lhs, rhs);
}
auto BackendCUDA::pow(const Tensor &lhs, const Tensor &rhs) const -> Tensor {
    return binary_runner<BinaryOpT::pow>(lhs, rhs);
}
auto BackendCUDA::batched_matmul(const Tensor &lhs, const Tensor &rhs) const -> Tensor {
    return batched_matmul_runner(lhs, rhs);
}

void BackendCUDA::add_inplace(Tensor &lhs, const Tensor &rhs) const {
    binary_inplace_runner<BinaryOpT::add>(lhs, rhs);
}
void BackendCUDA::sub_inplace(Tensor &lhs, const Tensor &rhs) const {
    binary_inplace_runner<BinaryOpT::subtract>(lhs, rhs);
}
void BackendCUDA::mul_inplace(Tensor &lhs, const Tensor &rhs) const {
    binary_inplace_runner<BinaryOpT::multiply>(lhs, rhs);
}
void BackendCUDA::div_inplace(Tensor &lhs, const Tensor &rhs) const {
    binary_inplace_runner<BinaryOpT::divide>(lhs, rhs);
}

// ------------------------------------------------
// Reduction operations
// ------------------------------------------------
auto BackendCUDA::min(const Tensor &tensor) const -> Tensor {
    return reduce_all_runner<ReduceOpT::min>(tensor);
}
auto BackendCUDA::min(const Tensor &tensor, int dim) const -> Tensor {
    return reduce_dim_runner<ReduceOpT::min>(tensor, dim);
}

auto BackendCUDA::argmin(const Tensor &tensor) const -> Tensor {
    return reduce_all_runner<ReduceOpT::argmin>(tensor);
}
auto BackendCUDA::argmin(const Tensor &tensor, int dim) const -> Tensor {
    return reduce_dim_runner<ReduceOpT::argmin>(tensor, dim);
}

auto BackendCUDA::max(const Tensor &tensor) const -> Tensor {
    return reduce_all_runner<ReduceOpT::max>(tensor);
}
auto BackendCUDA::max(const Tensor &tensor, int dim) const -> Tensor {
    return reduce_dim_runner<ReduceOpT::max>(tensor, dim);
}

auto BackendCUDA::argmax(const Tensor &tensor) const -> Tensor {
    return reduce_all_runner<ReduceOpT::argmax>(tensor);
}
auto BackendCUDA::argmax(const Tensor &tensor, int dim) const -> Tensor {
    return reduce_dim_runner<ReduceOpT::argmax>(tensor, dim);
}

auto BackendCUDA::sum(const Tensor &tensor) const -> Tensor {
    return reduce_all_runner<ReduceOpT::sum>(tensor);
}
auto BackendCUDA::sum(const Tensor &tensor, int dim) const -> Tensor {
    return reduce_dim_runner<ReduceOpT::sum>(tensor, dim);
}

auto BackendCUDA::all(const Tensor &tensor) const -> bool {
    return reduce_all_runner<ReduceOpT::all>(tensor).item<uint8_t>();
}
auto BackendCUDA::all(const Tensor &tensor, int dim) const -> Tensor {
    return reduce_dim_runner<ReduceOpT::all>(tensor, dim);
}

auto BackendCUDA::any(const Tensor &tensor) const -> bool {
    return reduce_all_runner<ReduceOpT::any>(tensor).item<uint8_t>();
}
auto BackendCUDA::any(const Tensor &tensor, int dim) const -> Tensor {
    return reduce_dim_runner<ReduceOpT::any>(tensor, dim);
}

// ------------------------------------------------
// Unary operations
// ------------------------------------------------
auto BackendCUDA::identity(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::identity>(tensor);
}
void BackendCUDA::identity_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::identity>(tensor);
}

auto BackendCUDA::negate(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::negate>(tensor);
}
void BackendCUDA::negate_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::negate>(tensor);
}

auto BackendCUDA::logical_not(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::logical_not>(tensor);
}
void BackendCUDA::logical_not_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::logical_not>(tensor);
}

auto BackendCUDA::abs(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::abs>(tensor);
}
void BackendCUDA::abs_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::abs>(tensor);
}

auto BackendCUDA::sign(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::sign>(tensor);
}
void BackendCUDA::sign_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::sign>(tensor);
}

auto BackendCUDA::log(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::log>(tensor);
}
void BackendCUDA::log_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::log>(tensor);
}

auto BackendCUDA::log10(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::log10>(tensor);
}
void BackendCUDA::log10_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::log10>(tensor);
}

auto BackendCUDA::log2(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::log2>(tensor);
}
void BackendCUDA::log2_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::log2>(tensor);
}

auto BackendCUDA::log1p(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::log1p>(tensor);
}
void BackendCUDA::log1p_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::log1p>(tensor);
}

auto BackendCUDA::exp(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::exp>(tensor);
}
void BackendCUDA::exp_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::exp>(tensor);
}

auto BackendCUDA::exp2(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::exp2>(tensor);
}
void BackendCUDA::exp2_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::exp2>(tensor);
}

auto BackendCUDA::expm1(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::expm1>(tensor);
}
void BackendCUDA::expm1_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::expm1>(tensor);
}

auto BackendCUDA::sqrt(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::sqrt>(tensor);
}
void BackendCUDA::sqrt_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::sqrt>(tensor);
}

auto BackendCUDA::sin(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::sin>(tensor);
}
void BackendCUDA::sin_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::sin>(tensor);
}

auto BackendCUDA::cos(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::cos>(tensor);
}
void BackendCUDA::cos_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::cos>(tensor);
}

auto BackendCUDA::tan(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::tan>(tensor);
}
void BackendCUDA::tan_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::tan>(tensor);
}

auto BackendCUDA::asin(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::asin>(tensor);
}
void BackendCUDA::asin_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::asin>(tensor);
}

auto BackendCUDA::acos(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::acos>(tensor);
}
void BackendCUDA::acos_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::acos>(tensor);
}

auto BackendCUDA::atan(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::atan>(tensor);
}
void BackendCUDA::atan_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::atan>(tensor);
}

auto BackendCUDA::sinh(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::sinh>(tensor);
}
void BackendCUDA::sinh_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::sinh>(tensor);
}

auto BackendCUDA::cosh(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::cosh>(tensor);
}
void BackendCUDA::cosh_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::cosh>(tensor);
}

auto BackendCUDA::tanh(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::tanh>(tensor);
}
void BackendCUDA::tanh_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::tanh>(tensor);
}

auto BackendCUDA::asinh(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::asinh>(tensor);
}
void BackendCUDA::asinh_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::asinh>(tensor);
}

auto BackendCUDA::acosh(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::acosh>(tensor);
}
void BackendCUDA::acosh_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::acosh>(tensor);
}

auto BackendCUDA::atanh(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::atanh>(tensor);
}
void BackendCUDA::atanh_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::atanh>(tensor);
}

auto BackendCUDA::erf(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::erf>(tensor);
}
void BackendCUDA::erf_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::erf>(tensor);
}

auto BackendCUDA::erfc(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::erfc>(tensor);
}
void BackendCUDA::erfc_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::erfc>(tensor);
}

auto BackendCUDA::tgamma(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::tgamma>(tensor);
}
void BackendCUDA::tgamma_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::tgamma>(tensor);
}

auto BackendCUDA::lgamma(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::lgamma>(tensor);
}
void BackendCUDA::lgamma_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::lgamma>(tensor);
}

auto BackendCUDA::digamma(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::digamma>(tensor);
}
void BackendCUDA::digamma_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::digamma>(tensor);
}

auto BackendCUDA::ceil(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::ceil>(tensor);
}
void BackendCUDA::ceil_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::ceil>(tensor);
}

auto BackendCUDA::floor(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::floor>(tensor);
}
void BackendCUDA::floor_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::floor>(tensor);
}

auto BackendCUDA::round(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::round>(tensor);
}
void BackendCUDA::round_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::round>(tensor);
}

auto BackendCUDA::isinf(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::isinf>(tensor);
}
auto BackendCUDA::isnan(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::isnan>(tensor);
}
auto BackendCUDA::isfinite(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::isfinite>(tensor);
}

// ------------------------------------------------
// Activation functions
// ------------------------------------------------
auto BackendCUDA::sigmoid(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::sigmoid>(tensor);
}
void BackendCUDA::sigmoid_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::sigmoid>(tensor);
}

auto BackendCUDA::log_sigmoid(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::log_sigmoid>(tensor);
}
void BackendCUDA::log_sigmoid_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::log_sigmoid>(tensor);
}

auto BackendCUDA::hardsigmoid(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::hardsigmoid>(tensor);
}
void BackendCUDA::hardsigmoid_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::hardsigmoid>(tensor);
}

auto BackendCUDA::softplus(const Tensor &tensor, double beta, double threshold) const -> Tensor {
    return unary_runner<UnaryOpT::softplus>(tensor, beta, threshold);
}
void BackendCUDA::softplus_(Tensor &tensor, double beta, double threshold) const {
    unary_runner_inplace<UnaryOpT::softplus>(tensor, beta, threshold);
}

auto BackendCUDA::relu(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::relu>(tensor);
}
void BackendCUDA::relu_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::relu>(tensor);
}

auto BackendCUDA::relu6(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::relu6>(tensor);
}
void BackendCUDA::relu6_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::relu6>(tensor);
}

auto BackendCUDA::leaky_relu(const Tensor &tensor, double negative_slope) const -> Tensor {
    return unary_runner<UnaryOpT::leaky_relu>(tensor, negative_slope);
}
void BackendCUDA::leaky_relu_(Tensor &tensor, double negative_slope) const {
    unary_runner_inplace<UnaryOpT::leaky_relu>(tensor, negative_slope);
}

auto BackendCUDA::elu(const Tensor &tensor, double alpha) const -> Tensor {
    return unary_runner<UnaryOpT::elu>(tensor, alpha);
}
void BackendCUDA::elu_(Tensor &tensor, double alpha) const {
    unary_runner_inplace<UnaryOpT::elu>(tensor, alpha);
}

auto BackendCUDA::selu(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::selu>(tensor);
}
void BackendCUDA::selu_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::selu>(tensor);
}

auto BackendCUDA::silu(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::silu>(tensor);
}
void BackendCUDA::silu_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::silu>(tensor);
}

auto BackendCUDA::hardtanh(const Tensor &tensor, double min, double max) const -> Tensor {
    return unary_runner<UnaryOpT::hardtanh>(tensor, min, max);
}
void BackendCUDA::hardtanh_(Tensor &tensor, double min, double max) const {
    unary_runner_inplace<UnaryOpT::hardtanh>(tensor, min, max);
}

auto BackendCUDA::softsign(const Tensor &tensor) const -> Tensor {
    return unary_runner<UnaryOpT::softsign>(tensor);
}
void BackendCUDA::softsign_(Tensor &tensor) const {
    unary_runner_inplace<UnaryOpT::softsign>(tensor);
}

auto BackendCUDA::softmax(const Tensor &tensor, int dim) const -> Tensor {
    auto result = tensor.clone();
    softmax_(result, dim);
    return result;
}
void BackendCUDA::softmax_(Tensor &tensor, int dim) const {
    const auto m = tensor.max(dim, true).expand(tensor.shape());
    tensor -= m;
    tensor.exp_();
    const auto denom = tensor.sum(dim, true).expand(tensor.shape());
    tensor /= denom;
}

auto BackendCUDA::log_softmax(const Tensor &tensor, int dim) const -> Tensor {
    auto result = tensor.clone();
    log_softmax_(result, dim);
    return result;
}
void BackendCUDA::log_softmax_(Tensor &tensor, int dim) const {
    const auto m = tensor.max(dim, true).expand(tensor.shape());
    const auto logsumexp = log(exp(tensor - m).sum(dim, true).expand(tensor.shape()));
    tensor -= m;
    tensor -= logsumexp;
}

// ------------------------------------------------
// Util/misc
// ------------------------------------------------
auto BackendCUDA::where(const Tensor &cond, const Tensor &lhs, const Tensor &rhs) const -> Tensor {
    return where_runner(cond, lhs, rhs);
}

void BackendCUDA::clamp_(Tensor &tensor, const Tensor &min, const Tensor &max) const {
    clamp_inplace_runner(tensor, min, max);
}

auto BackendCUDA::conv2d(
    const Tensor &input,
    const Tensor &weight,
    const std::optional<Tensor> &bias,
    int stride,
    int padding
) const -> Tensor {
    return batched_conv2d_forward_runner(input, weight, bias, stride, padding);
}
auto BackendCUDA::conv2d_backward(
    const Tensor &grad_output,
    const Tensor &input,
    const Tensor &weight,
    const std::optional<Tensor> &bias,
    int stride,
    int padding
) const -> std::tuple<Tensor, Tensor, std::optional<Tensor>> {
    return batched_conv2d_backward_runner(grad_output, input, weight, bias, stride, padding);
}

auto BackendCUDA::max_pool2d(const Tensor &input, int kernel_size, int stride, int padding) const -> Tensor {
    return batched_pool2d_forward_runner<ReduceOpT::max>(input, kernel_size, stride, padding);
}
auto BackendCUDA::max_pool2d_backward(
    const Tensor &grad_output,
    const Tensor &input,
    const Tensor &result,
    int kernel_size,
    int stride,
    int padding
) const -> Tensor {
    return batched_pool2d_backward_runner<ReduceOpT::max>(grad_output, input, result, kernel_size, stride, padding);
}

auto BackendCUDA::min_pool2d(const Tensor &input, int kernel_size, int stride, int padding) const -> Tensor {
    return batched_pool2d_forward_runner<ReduceOpT::min>(input, kernel_size, stride, padding);
}

auto BackendCUDA::min_pool2d_backward(
    const Tensor &grad_output,
    const Tensor &input,
    const Tensor &result,
    int kernel_size,
    int stride,
    int padding
) const -> Tensor {
    return batched_pool2d_backward_runner<ReduceOpT::min>(grad_output, input, result, kernel_size, stride, padding);
}

auto BackendCUDA::avg_pool2d(const Tensor &input, int kernel_size, int stride, int padding) const -> Tensor {
    return batched_pool2d_forward_runner<ReduceOpT::mean>(input, kernel_size, stride, padding);
}
auto BackendCUDA::avg_pool2d_backward(
    const Tensor &grad_output,
    const Tensor &input,
    const Tensor &result,
    int kernel_size,
    int stride,
    int padding
) const -> Tensor {
    return batched_pool2d_backward_runner<ReduceOpT::mean>(grad_output, input, result, kernel_size, stride, padding);
}

auto BackendCUDA::current_memory_allocated(int device_id) const -> uint64_t {
    CHECK_DEVICE(device_id);
    return StorageCUDA::current_bytes_allocated[device_id];
}
auto BackendCUDA::total_memory_allocated([[maybe_unused]] int device_id) const -> uint64_t {
    CHECK_DEVICE(device_id);
    return StorageCUDA::total_bytes_allocated[device_id];
}

auto BackendCUDA::get_device_count() const -> int {
    return cuda::get_device_count();
}

}    // namespace tinytensor::cuda
