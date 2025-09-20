// backend_base.h
// Interface for all backends

#ifndef TINYTENSOR_BACKEND_BASE_H_
#define TINYTENSOR_BACKEND_BASE_H_

#include <tt/random.h>
#include <tt/scalar.h>
#include <tt/tensor.h>

#include "storage_base.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <tuple>
#include <vector>

namespace tinytensor {

class BackendBase {
public:
    BackendBase() = default;
    virtual ~BackendBase() = default;

    // Does not copy, but we can move construct
    BackendBase(const BackendBase &) = delete;
    BackendBase &operator=(const BackendBase &) = delete;
    BackendBase(BackendBase &&) = default;
    BackendBase &operator=(BackendBase &&) = default;

    // Tensor construction
    using StoragePtr = std::unique_ptr<StorageBase>;
    // No rvalue for bool required since we never steal the data, we transform to uint8_t
    [[nodiscard]] virtual auto from_vec(const std::vector<bool> &data, int device_id) const -> StoragePtr = 0;
    [[nodiscard]] virtual auto from_vec(const std::vector<uint8_t> &data, int device_id) const -> StoragePtr = 0;
    [[nodiscard]] virtual auto from_vec(std::vector<uint8_t> &&data, int device_id) const -> StoragePtr = 0;
    [[nodiscard]] virtual auto from_vec(const std::vector<int16_t> &data, int device_id) const -> StoragePtr = 0;
    [[nodiscard]] virtual auto from_vec(std::vector<int16_t> &&data, int device_id) const -> StoragePtr = 0;
    [[nodiscard]] virtual auto from_vec(const std::vector<int32_t> &data, int device_id) const -> StoragePtr = 0;
    [[nodiscard]] virtual auto from_vec(std::vector<int32_t> &&data, int device_id) const -> StoragePtr = 0;
    [[nodiscard]] virtual auto from_vec(const std::vector<int64_t> &data, int device_id) const -> StoragePtr = 0;
    [[nodiscard]] virtual auto from_vec(std::vector<int64_t> &&data, int device_id) const -> StoragePtr = 0;
    [[nodiscard]] virtual auto from_vec(const std::vector<float> &data, int device_id) const -> StoragePtr = 0;
    [[nodiscard]] virtual auto from_vec(std::vector<float> &&data, int device_id) const -> StoragePtr = 0;
    [[nodiscard]] virtual auto from_vec(const std::vector<double> &data, int device_id) const -> StoragePtr = 0;
    [[nodiscard]] virtual auto from_vec(std::vector<double> &&data, int device_id) const -> StoragePtr = 0;
    [[nodiscard]] virtual auto from_scalar(Scalar scalar, int device_id) const -> StoragePtr = 0;
    [[nodiscard]] virtual auto full(const Scalar &value, std::size_t N, int device_id) const -> StoragePtr = 0;
    [[nodiscard]] virtual auto arange(std::size_t N, ScalarType dtype, int device_id) const -> StoragePtr = 0;

    // Conversion to vec
    virtual void to_vec(const Tensor &tensor, std::vector<uint8_t> &data_out) const = 0;
    virtual void to_vec(const Tensor &tensor, std::vector<int16_t> &data_out) const = 0;
    virtual void to_vec(const Tensor &tensor, std::vector<int32_t> &data_out) const = 0;
    virtual void to_vec(const Tensor &tensor, std::vector<int64_t> &data_out) const = 0;
    virtual void to_vec(const Tensor &tensor, std::vector<float> &data_out) const = 0;
    virtual void to_vec(const Tensor &tensor, std::vector<double> &data_out) const = 0;

    // Getters
    [[nodiscard]] virtual auto item(const Tensor &tensor) const -> Scalar = 0;
    [[nodiscard]] virtual auto data_ptr(const Tensor &tensor) const -> uintptr_t = 0;

    // Indexing
    [[nodiscard]] virtual auto index_mask(const Tensor &input, const Tensor &mask, int N_mask) const -> Tensor = 0;
    [[nodiscard]] virtual auto index_indices(const Tensor &input, const Tensor &indices) const -> Tensor = 0;
    virtual void index_put_mask(Tensor &input, const Tensor &values, const Tensor &mask) const = 0;
    virtual void index_put_indices(Tensor &input, const Tensor &values, const Tensor &indices) const = 0;
    [[nodiscard]] virtual auto gather(const Tensor &input, const Tensor &indices, int idx) const -> Tensor = 0;

    // Casts
    [[nodiscard]] virtual auto to(const Tensor &tensor, ScalarType dtype) const -> Tensor = 0;

    virtual auto print(std::ostream &os, const Tensor &tensor) const -> std::ostream & = 0;

    virtual void assign(Tensor &lhs, const Tensor &rhs) const = 0;

    // ------------------------------------------------
    // Tensor Creation - Distributions
    // ------------------------------------------------
    [[nodiscard]] virtual auto uniform_int(const Tensor &low, const Tensor &high, Generator &gen) const -> Tensor = 0;
    [[nodiscard]] virtual auto uniform_real(const Tensor &low, const Tensor &high, Generator &gen) const -> Tensor = 0;
    [[nodiscard]] virtual auto bernoulli(const Tensor &p, Generator &gen) const -> Tensor = 0;
    [[nodiscard]] virtual auto binomial(const Tensor &p, const Tensor &num_draws, Generator &gen) const -> Tensor = 0;
    [[nodiscard]] virtual auto geometric(const Tensor &p, Generator &gen) const -> Tensor = 0;
    [[nodiscard]] virtual auto poisson(const Tensor &lambda, Generator &gen) const -> Tensor = 0;
    [[nodiscard]] virtual auto exponential(const Tensor &lambda, Generator &gen) const -> Tensor = 0;
    [[nodiscard]] virtual auto normal(const Tensor &mu, const Tensor &std, Generator &gen) const -> Tensor = 0;
    [[nodiscard]] virtual auto cauchy(const Tensor &loc, const Tensor &scale, Generator &gen) const -> Tensor = 0;
    [[nodiscard]] virtual auto lognormal(const Tensor &mu, const Tensor &std, Generator &gen) const -> Tensor = 0;
    [[nodiscard]] virtual auto weibull(const Tensor &scale, const Tensor &shape, Generator &gen) const -> Tensor = 0;
    virtual void uniform_int_(Tensor &tensor, const Tensor &low, const Tensor &high, Generator &gen) const = 0;
    virtual void uniform_real_(Tensor &tensor, const Tensor &low, const Tensor &high, Generator &gen) const = 0;
    virtual void bernoulli_(Tensor &tensor, const Tensor &p, Generator &gen) const = 0;
    virtual void binomial_(Tensor &tensor, const Tensor &p, const Tensor &num_draws, Generator &gen) const = 0;
    virtual void geometric_(Tensor &tensor, const Tensor &p, Generator &gen) const = 0;
    virtual void poisson_(Tensor &tensor, const Tensor &lambda, Generator &gen) const = 0;
    virtual void exponential_(Tensor &tensor, const Tensor &lambda, Generator &gen) const = 0;
    virtual void normal_(Tensor &tensor, const Tensor &mu, const Tensor &std, Generator &gen) const = 0;
    virtual void cauchy_(Tensor &tensor, const Tensor &loc, const Tensor &scale, Generator &gen) const = 0;
    virtual void lognormal_(Tensor &tensor, const Tensor &mu, const Tensor &std, Generator &gen) const = 0;
    virtual void weibull_(Tensor &tensor, const Tensor &scale, const Tensor &shape, Generator &gen) const = 0;

    // ------------------------------------------------
    // Binary Operators
    // ------------------------------------------------
    [[nodiscard]] virtual auto eq(const Tensor &lhs, const Tensor &rhs) const -> Tensor = 0;
    [[nodiscard]] virtual auto ne(const Tensor &lhs, const Tensor &rhs) const -> Tensor = 0;
    [[nodiscard]] virtual auto lt(const Tensor &lhs, const Tensor &rhs) const -> Tensor = 0;
    [[nodiscard]] virtual auto le(const Tensor &lhs, const Tensor &rhs) const -> Tensor = 0;
    [[nodiscard]] virtual auto gt(const Tensor &lhs, const Tensor &rhs) const -> Tensor = 0;
    [[nodiscard]] virtual auto ge(const Tensor &lhs, const Tensor &rhs) const -> Tensor = 0;
    [[nodiscard]] virtual auto logical_or(const Tensor &lhs, const Tensor &rhs) const -> Tensor = 0;
    [[nodiscard]] virtual auto logical_and(const Tensor &lhs, const Tensor &rhs) const -> Tensor = 0;
    [[nodiscard]] virtual auto bitwise_or(const Tensor &lhs, const Tensor &rhs) const -> Tensor = 0;
    [[nodiscard]] virtual auto bitwise_and(const Tensor &lhs, const Tensor &rhs) const -> Tensor = 0;
    [[nodiscard]] virtual auto bitwise_xor(const Tensor &lhs, const Tensor &rhs) const -> Tensor = 0;
    [[nodiscard]] virtual auto bitwise_left_shift(const Tensor &lhs, const Tensor &rhs) const -> Tensor = 0;
    [[nodiscard]] virtual auto bitwise_right_shift(const Tensor &lhs, const Tensor &rhs) const -> Tensor = 0;
    [[nodiscard]] virtual auto modulo(const Tensor &lhs, const Tensor &rhs) const -> Tensor = 0;
    [[nodiscard]] virtual auto add(const Tensor &lhs, const Tensor &rhs) const -> Tensor = 0;
    [[nodiscard]] virtual auto sub(const Tensor &lhs, const Tensor &rhs) const -> Tensor = 0;
    [[nodiscard]] virtual auto mul(const Tensor &lhs, const Tensor &rhs) const -> Tensor = 0;
    [[nodiscard]] virtual auto div(const Tensor &lhs, const Tensor &rhs) const -> Tensor = 0;
    [[nodiscard]] virtual auto maximum(const Tensor &lhs, const Tensor &rhs) const -> Tensor = 0;
    [[nodiscard]] virtual auto minimum(const Tensor &lhs, const Tensor &rhs) const -> Tensor = 0;
    [[nodiscard]] virtual auto pow(const Tensor &lhs, const Tensor &rhs) const -> Tensor = 0;
    [[nodiscard]] virtual auto batched_matmul(const Tensor &lhs, const Tensor &rhs) const -> Tensor = 0;

    virtual void add_inplace(Tensor &lhs, const Tensor &rhs) const = 0;
    virtual void sub_inplace(Tensor &lhs, const Tensor &rhs) const = 0;
    virtual void mul_inplace(Tensor &lhs, const Tensor &rhs) const = 0;
    virtual void div_inplace(Tensor &lhs, const Tensor &rhs) const = 0;

    // ------------------------------------------------
    // Reduction operations
    // ------------------------------------------------
    [[nodiscard]] virtual auto min(const Tensor &tensor) const -> Tensor = 0;
    [[nodiscard]] virtual auto min(const Tensor &tensor, int dim) const -> Tensor = 0;
    [[nodiscard]] virtual auto argmin(const Tensor &tensor) const -> Tensor = 0;
    [[nodiscard]] virtual auto argmin(const Tensor &tensor, int dim) const -> Tensor = 0;
    [[nodiscard]] virtual auto max(const Tensor &tensor) const -> Tensor = 0;
    [[nodiscard]] virtual auto max(const Tensor &tensor, int dim) const -> Tensor = 0;
    [[nodiscard]] virtual auto argmax(const Tensor &tensor) const -> Tensor = 0;
    [[nodiscard]] virtual auto argmax(const Tensor &tensor, int dim) const -> Tensor = 0;
    [[nodiscard]] virtual auto sum(const Tensor &tensor) const -> Tensor = 0;
    [[nodiscard]] virtual auto sum(const Tensor &tensor, int dim) const -> Tensor = 0;
    [[nodiscard]] virtual auto all(const Tensor &tensor) const -> bool = 0;
    [[nodiscard]] virtual auto all(const Tensor &tensor, int dim) const -> Tensor = 0;
    [[nodiscard]] virtual auto any(const Tensor &tensor) const -> bool = 0;
    [[nodiscard]] virtual auto any(const Tensor &tensor, int dim) const -> Tensor = 0;

    // ------------------------------------------------
    // Unary operations
    // ------------------------------------------------
    [[nodiscard]] virtual auto identity(const Tensor &tensor) const -> Tensor = 0;
    [[nodiscard]] virtual auto negate(const Tensor &tensor) const -> Tensor = 0;
    [[nodiscard]] virtual auto logical_not(const Tensor &tensor) const -> Tensor = 0;
    [[nodiscard]] virtual auto abs(const Tensor &tensor) const -> Tensor = 0;
    [[nodiscard]] virtual auto sign(const Tensor &tensor) const -> Tensor = 0;
    [[nodiscard]] virtual auto log(const Tensor &tensor) const -> Tensor = 0;
    [[nodiscard]] virtual auto log10(const Tensor &tensor) const -> Tensor = 0;
    [[nodiscard]] virtual auto log2(const Tensor &tensor) const -> Tensor = 0;
    [[nodiscard]] virtual auto log1p(const Tensor &tensor) const -> Tensor = 0;
    [[nodiscard]] virtual auto exp(const Tensor &tensor) const -> Tensor = 0;
    [[nodiscard]] virtual auto exp2(const Tensor &tensor) const -> Tensor = 0;
    [[nodiscard]] virtual auto expm1(const Tensor &tensor) const -> Tensor = 0;
    [[nodiscard]] virtual auto sqrt(const Tensor &tensor) const -> Tensor = 0;
    [[nodiscard]] virtual auto sin(const Tensor &tensor) const -> Tensor = 0;
    [[nodiscard]] virtual auto cos(const Tensor &tensor) const -> Tensor = 0;
    [[nodiscard]] virtual auto tan(const Tensor &tensor) const -> Tensor = 0;
    [[nodiscard]] virtual auto asin(const Tensor &tensor) const -> Tensor = 0;
    [[nodiscard]] virtual auto acos(const Tensor &tensor) const -> Tensor = 0;
    [[nodiscard]] virtual auto atan(const Tensor &tensor) const -> Tensor = 0;
    [[nodiscard]] virtual auto sinh(const Tensor &tensor) const -> Tensor = 0;
    [[nodiscard]] virtual auto cosh(const Tensor &tensor) const -> Tensor = 0;
    [[nodiscard]] virtual auto tanh(const Tensor &tensor) const -> Tensor = 0;
    [[nodiscard]] virtual auto asinh(const Tensor &tensor) const -> Tensor = 0;
    [[nodiscard]] virtual auto acosh(const Tensor &tensor) const -> Tensor = 0;
    [[nodiscard]] virtual auto atanh(const Tensor &tensor) const -> Tensor = 0;
    [[nodiscard]] virtual auto erf(const Tensor &tensor) const -> Tensor = 0;
    [[nodiscard]] virtual auto erfc(const Tensor &tensor) const -> Tensor = 0;
    [[nodiscard]] virtual auto tgamma(const Tensor &tensor) const -> Tensor = 0;
    [[nodiscard]] virtual auto lgamma(const Tensor &tensor) const -> Tensor = 0;
    [[nodiscard]] virtual auto digamma(const Tensor &tensor) const -> Tensor = 0;
    [[nodiscard]] virtual auto ceil(const Tensor &tensor) const -> Tensor = 0;
    [[nodiscard]] virtual auto floor(const Tensor &tensor) const -> Tensor = 0;
    [[nodiscard]] virtual auto round(const Tensor &tensor) const -> Tensor = 0;
    [[nodiscard]] virtual auto isinf(const Tensor &tensor) const -> Tensor = 0;
    [[nodiscard]] virtual auto isnan(const Tensor &tensor) const -> Tensor = 0;
    [[nodiscard]] virtual auto isfinite(const Tensor &tensor) const -> Tensor = 0;

    virtual void identity_(Tensor &tensor) const = 0;
    virtual void negate_(Tensor &tensor) const = 0;
    virtual void logical_not_(Tensor &tensor) const = 0;
    virtual void abs_(Tensor &tensor) const = 0;
    virtual void sign_(Tensor &tensor) const = 0;
    virtual void log_(Tensor &tensor) const = 0;
    virtual void log10_(Tensor &tensor) const = 0;
    virtual void log2_(Tensor &tensor) const = 0;
    virtual void log1p_(Tensor &tensor) const = 0;
    virtual void exp_(Tensor &tensor) const = 0;
    virtual void exp2_(Tensor &tensor) const = 0;
    virtual void expm1_(Tensor &tensor) const = 0;
    virtual void sqrt_(Tensor &tensor) const = 0;
    virtual void sin_(Tensor &tensor) const = 0;
    virtual void cos_(Tensor &tensor) const = 0;
    virtual void tan_(Tensor &tensor) const = 0;
    virtual void asin_(Tensor &tensor) const = 0;
    virtual void acos_(Tensor &tensor) const = 0;
    virtual void atan_(Tensor &tensor) const = 0;
    virtual void sinh_(Tensor &tensor) const = 0;
    virtual void cosh_(Tensor &tensor) const = 0;
    virtual void tanh_(Tensor &tensor) const = 0;
    virtual void asinh_(Tensor &tensor) const = 0;
    virtual void acosh_(Tensor &tensor) const = 0;
    virtual void atanh_(Tensor &tensor) const = 0;
    virtual void erf_(Tensor &tensor) const = 0;
    virtual void erfc_(Tensor &tensor) const = 0;
    virtual void tgamma_(Tensor &tensor) const = 0;
    virtual void lgamma_(Tensor &tensor) const = 0;
    virtual void digamma_(Tensor &tensor) const = 0;
    virtual void ceil_(Tensor &tensor) const = 0;
    virtual void floor_(Tensor &tensor) const = 0;
    virtual void round_(Tensor &tensor) const = 0;

    // ------------------------------------------------
    // Activation functions
    // ------------------------------------------------
    [[nodiscard]] virtual auto sigmoid(const Tensor &tensor) const -> Tensor = 0;
    [[nodiscard]] virtual auto log_sigmoid(const Tensor &tensor) const -> Tensor = 0;
    [[nodiscard]] virtual auto hardsigmoid(const Tensor &tensor) const -> Tensor = 0;
    [[nodiscard]] virtual auto softplus(const Tensor &tensor, double beta, double threshold) const -> Tensor = 0;
    [[nodiscard]] virtual auto relu(const Tensor &tensor) const -> Tensor = 0;
    [[nodiscard]] virtual auto relu6(const Tensor &tensor) const -> Tensor = 0;
    [[nodiscard]] virtual auto leaky_relu(const Tensor &tensor, double negative_slope) const -> Tensor = 0;
    [[nodiscard]] virtual auto elu(const Tensor &tensor, double alpha) const -> Tensor = 0;
    [[nodiscard]] virtual auto selu(const Tensor &tensor) const -> Tensor = 0;
    [[nodiscard]] virtual auto silu(const Tensor &tensor) const -> Tensor = 0;
    [[nodiscard]] virtual auto hardtanh(const Tensor &tensor, double min, double max) const -> Tensor = 0;
    [[nodiscard]] virtual auto softsign(const Tensor &tensor) const -> Tensor = 0;
    [[nodiscard]] virtual auto softmax(const Tensor &tensor, int dim) const -> Tensor = 0;
    [[nodiscard]] virtual auto log_softmax(const Tensor &tensor, int dim) const -> Tensor = 0;

    virtual void sigmoid_(Tensor &tensor) const = 0;
    virtual void log_sigmoid_(Tensor &tensor) const = 0;
    virtual void hardsigmoid_(Tensor &tensor) const = 0;
    virtual void softplus_(Tensor &tensor, double beta, double threshold) const = 0;
    virtual void relu_(Tensor &tensor) const = 0;
    virtual void relu6_(Tensor &tensor) const = 0;
    virtual void leaky_relu_(Tensor &tensor, double negative_slope) const = 0;
    virtual void elu_(Tensor &tensor, double alpha) const = 0;
    virtual void selu_(Tensor &tensor) const = 0;
    virtual void silu_(Tensor &tensor) const = 0;
    virtual void hardtanh_(Tensor &tensor, double min, double max) const = 0;
    virtual void softsign_(Tensor &tensor) const = 0;
    virtual void softmax_(Tensor &tensor, int dim) const = 0;
    virtual void log_softmax_(Tensor &tensor, int dim) const = 0;

    // ------------------------------------------------
    // Util/misc
    // ------------------------------------------------

    [[nodiscard]] virtual auto where(const Tensor &cond, const Tensor &lhs, const Tensor &rhs) const -> Tensor = 0;
    virtual void clamp_(Tensor &tensor, const Tensor &min, const Tensor &max) const = 0;

    [[nodiscard]] virtual auto conv2d(
        const Tensor &input,
        const Tensor &weight,
        const std::optional<Tensor> &bias,
        int stride,
        int padding
    ) const -> Tensor = 0;

    [[nodiscard]] virtual auto conv2d_backward(
        const Tensor &grad_output,
        const Tensor &input,
        const Tensor &weight,
        const std::optional<Tensor> &bias,
        int stride,
        int padding
    ) const -> std::tuple<Tensor, Tensor, std::optional<Tensor>> = 0;

    [[nodiscard]] virtual auto max_pool2d(const Tensor &input, int kernel_size, int stride, int padding) const
        -> Tensor = 0;
    [[nodiscard]] virtual auto max_pool2d_backward(
        const Tensor &grad_output,
        const Tensor &input,
        const Tensor &result,
        int kernel_size,
        int stride,
        int padding
    ) const -> Tensor = 0;
    [[nodiscard]] virtual auto min_pool2d(const Tensor &input, int kernel_size, int stride, int padding) const
        -> Tensor = 0;
    [[nodiscard]] virtual auto min_pool2d_backward(
        const Tensor &grad_output,
        const Tensor &input,
        const Tensor &result,
        int kernel_size,
        int stride,
        int padding
    ) const -> Tensor = 0;
    [[nodiscard]] virtual auto avg_pool2d(const Tensor &input, int kernel_size, int stride, int padding) const
        -> Tensor = 0;
    [[nodiscard]] virtual auto avg_pool2d_backward(
        const Tensor &grad_output,
        const Tensor &input,
        const Tensor &result,
        int kernel_size,
        int stride,
        int padding
    ) const -> Tensor = 0;

    [[nodiscard]] virtual auto current_memory_allocated(int device_id) const -> uint64_t = 0;
    [[nodiscard]] virtual auto total_memory_allocated(int device_id) const -> uint64_t = 0;
};

}    // namespace tinytensor

#endif    // TINYTENSOR_BACKEND_BASE_H_
