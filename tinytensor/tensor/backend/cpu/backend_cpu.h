// backend_cpu.h
// CPU backend

#ifndef TINYTENSOR_BACKEND_CPU_H_
#define TINYTENSOR_BACKEND_CPU_H_

#include <tt/random.h>
#include <tt/scalar.h>
#include <tt/tensor.h>

#include "tensor/backend_base.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <ostream>
#include <tuple>
#include <vector>

namespace tinytensor::cpu {

class BackendCPU : public BackendBase {
    using kU8CType = to_ctype_t<kU8>;
    using kI16CType = to_ctype_t<kI16>;
    using kI32CType = to_ctype_t<kI32>;
    using kI64CType = to_ctype_t<kI64>;
    using kF32CType = to_ctype_t<kF32>;
    using kF64CType = to_ctype_t<kF64>;

public:
    // Tensor construction
    [[nodiscard]] auto from_vec(const std::vector<bool> &data, int device_id) const -> StoragePtr override;
    [[nodiscard]] auto from_vec(const std::vector<kU8CType> &data, int device_id) const -> StoragePtr override;
    [[nodiscard]] auto from_vec(std::vector<kU8CType> &&data, int device_id) const -> StoragePtr override;
    [[nodiscard]] auto from_vec(const std::vector<kI16CType> &data, int device_id) const -> StoragePtr override;
    [[nodiscard]] auto from_vec(std::vector<kI16CType> &&data, int device_id) const -> StoragePtr override;
    [[nodiscard]] auto from_vec(const std::vector<kI32CType> &data, int device_id) const -> StoragePtr override;
    [[nodiscard]] auto from_vec(std::vector<kI32CType> &&data, int device_id) const -> StoragePtr override;
    [[nodiscard]] auto from_vec(const std::vector<kI64CType> &data, int device_id) const -> StoragePtr override;
    [[nodiscard]] auto from_vec(std::vector<kI64CType> &&data, int device_id) const -> StoragePtr override;
    [[nodiscard]] auto from_vec(const std::vector<kF32CType> &data, int device_id) const -> StoragePtr override;
    [[nodiscard]] auto from_vec(std::vector<kF32CType> &&data, int device_id) const -> StoragePtr override;
    [[nodiscard]] auto from_vec(const std::vector<kF64CType> &data, int device_id) const -> StoragePtr override;
    [[nodiscard]] auto from_vec(std::vector<kF64CType> &&data, int device_id) const -> StoragePtr override;
    [[nodiscard]] auto from_scalar(const Scalar scalar, int device_id) const -> StoragePtr override;
    [[nodiscard]] auto full(const Scalar &value, std::size_t N, int device_id) const -> StoragePtr override;
    [[nodiscard]] auto arange(std::size_t N, ScalarType dtype, int device_id) const -> StoragePtr override;

    // Conversion to vec
    void to_vec(const Tensor &tensor, std::vector<kU8CType> &data_out) const override;
    void to_vec(const Tensor &tensor, std::vector<kI16CType> &data_out) const override;
    void to_vec(const Tensor &tensor, std::vector<kI32CType> &data_out) const override;
    void to_vec(const Tensor &tensor, std::vector<kI64CType> &data_out) const override;
    void to_vec(const Tensor &tensor, std::vector<kF32CType> &data_out) const override;
    void to_vec(const Tensor &tensor, std::vector<kF64CType> &data_out) const override;

    // Getters
    [[nodiscard]] auto item(const Tensor &tensor) const -> Scalar override;
    [[nodiscard]] auto data_ptr(const Tensor &tensor) const -> uintptr_t override;

    // Indexing
    [[nodiscard]] auto index_mask(const Tensor &input, const Tensor &mask, int N_mask) const -> Tensor override;
    [[nodiscard]] auto index_indices(const Tensor &input, const Tensor &indices) const -> Tensor override;
    void index_put_mask(Tensor &input, const Tensor &values, const Tensor &mask) const override;
    void index_put_indices(Tensor &input, const Tensor &values, const Tensor &indices) const override;
    [[nodiscard]] auto gather(const Tensor &input, const Tensor &indices, int idx) const -> Tensor override;

    // Casts
    [[nodiscard]] auto to(const Tensor &tensor, ScalarType dtype) const -> Tensor override;

    auto print(std::ostream &os, const Tensor &tensor) const -> std::ostream & override;

    void assign(Tensor &lhs, const Tensor &rhs) const override;

    // ------------------------------------------------
    // Tensor Creation - Distributions
    // ------------------------------------------------
    [[nodiscard]] auto uniform_int(const Tensor &low, const Tensor &high, Generator &gen) const -> Tensor override;
    [[nodiscard]] auto uniform_real(const Tensor &low, const Tensor &high, Generator &gen) const -> Tensor override;
    [[nodiscard]] auto bernoulli(const Tensor &p, Generator &gen) const -> Tensor override;
    [[nodiscard]] auto binomial(const Tensor &p, const Tensor &num_draws, Generator &gen) const -> Tensor override;
    [[nodiscard]] auto geometric(const Tensor &p, Generator &gen) const -> Tensor override;
    [[nodiscard]] auto poisson(const Tensor &lambda, Generator &gen) const -> Tensor override;
    [[nodiscard]] auto exponential(const Tensor &lambda, Generator &gen) const -> Tensor override;
    [[nodiscard]] auto normal(const Tensor &mu, const Tensor &std, Generator &gen) const -> Tensor override;
    [[nodiscard]] auto cauchy(const Tensor &loc, const Tensor &scale, Generator &gen) const -> Tensor override;
    [[nodiscard]] auto lognormal(const Tensor &mu, const Tensor &std, Generator &gen) const -> Tensor override;
    [[nodiscard]] auto weibull(const Tensor &scale, const Tensor &shape, Generator &gen) const -> Tensor override;

    void uniform_int_(Tensor &tensor, const Tensor &low, const Tensor &high, Generator &gen) const override;
    void uniform_real_(Tensor &tensor, const Tensor &low, const Tensor &high, Generator &gen) const override;
    void bernoulli_(Tensor &tensor, const Tensor &p, Generator &gen) const override;
    void binomial_(Tensor &tensor, const Tensor &p, const Tensor &num_draws, Generator &gen) const override;
    void geometric_(Tensor &tensor, const Tensor &p, Generator &gen) const override;
    void poisson_(Tensor &tensor, const Tensor &lambda, Generator &gen) const override;
    void exponential_(Tensor &tensor, const Tensor &lambda, Generator &gen) const override;
    void normal_(Tensor &tensor, const Tensor &mu, const Tensor &std, Generator &gen) const override;
    void cauchy_(Tensor &tensor, const Tensor &loc, const Tensor &scale, Generator &gen) const override;
    void lognormal_(Tensor &tensor, const Tensor &mu, const Tensor &std, Generator &gen) const override;
    void weibull_(Tensor &tensor, const Tensor &scale, const Tensor &shape, Generator &gen) const override;

    // ------------------------------------------------
    // Binary Operators
    // ------------------------------------------------
    [[nodiscard]] auto eq(const Tensor &lhs, const Tensor &rhs) const -> Tensor override;
    [[nodiscard]] auto ne(const Tensor &lhs, const Tensor &rhs) const -> Tensor override;
    [[nodiscard]] auto lt(const Tensor &lhs, const Tensor &rhs) const -> Tensor override;
    [[nodiscard]] auto le(const Tensor &lhs, const Tensor &rhs) const -> Tensor override;
    [[nodiscard]] auto gt(const Tensor &lhs, const Tensor &rhs) const -> Tensor override;
    [[nodiscard]] auto ge(const Tensor &lhs, const Tensor &rhs) const -> Tensor override;
    [[nodiscard]] auto logical_or(const Tensor &lhs, const Tensor &rhs) const -> Tensor override;
    [[nodiscard]] auto logical_and(const Tensor &lhs, const Tensor &rhs) const -> Tensor override;
    [[nodiscard]] auto bitwise_or(const Tensor &lhs, const Tensor &rhs) const -> Tensor override;
    [[nodiscard]] auto bitwise_and(const Tensor &lhs, const Tensor &rhs) const -> Tensor override;
    [[nodiscard]] auto bitwise_xor(const Tensor &lhs, const Tensor &rhs) const -> Tensor override;
    [[nodiscard]] auto bitwise_left_shift(const Tensor &lhs, const Tensor &rhs) const -> Tensor override;
    [[nodiscard]] auto bitwise_right_shift(const Tensor &lhs, const Tensor &rhs) const -> Tensor override;
    [[nodiscard]] auto modulo(const Tensor &lhs, const Tensor &rhs) const -> Tensor override;
    [[nodiscard]] auto add(const Tensor &lhs, const Tensor &rhs) const -> Tensor override;
    [[nodiscard]] auto sub(const Tensor &lhs, const Tensor &rhs) const -> Tensor override;
    [[nodiscard]] auto mul(const Tensor &lhs, const Tensor &rhs) const -> Tensor override;
    [[nodiscard]] auto div(const Tensor &lhs, const Tensor &rhs) const -> Tensor override;
    [[nodiscard]] auto maximum(const Tensor &lhs, const Tensor &rhs) const -> Tensor override;
    [[nodiscard]] auto minimum(const Tensor &lhs, const Tensor &rhs) const -> Tensor override;
    [[nodiscard]] auto pow(const Tensor &lhs, const Tensor &rhs) const -> Tensor override;
    [[nodiscard]] auto batched_matmul(const Tensor &lhs, const Tensor &rhs) const -> Tensor override;

    void add_inplace(Tensor &lhs, const Tensor &rhs) const override;
    void sub_inplace(Tensor &lhs, const Tensor &rhs) const override;
    void mul_inplace(Tensor &lhs, const Tensor &rhs) const override;
    void div_inplace(Tensor &lhs, const Tensor &rhs) const override;

    // ------------------------------------------------
    // Reduction operations
    // ------------------------------------------------
    [[nodiscard]] auto min(const Tensor &tensor) const -> Tensor override;
    [[nodiscard]] auto min(const Tensor &tensor, int dim) const -> Tensor override;
    [[nodiscard]] auto argmin(const Tensor &tensor) const -> Tensor override;
    [[nodiscard]] auto argmin(const Tensor &tensor, int dim) const -> Tensor override;
    [[nodiscard]] auto max(const Tensor &tensor) const -> Tensor override;
    [[nodiscard]] auto max(const Tensor &tensor, int dim) const -> Tensor override;
    [[nodiscard]] auto argmax(const Tensor &tensor) const -> Tensor override;
    [[nodiscard]] auto argmax(const Tensor &tensor, int dim) const -> Tensor override;
    [[nodiscard]] auto sum(const Tensor &tensor) const -> Tensor override;
    [[nodiscard]] auto sum(const Tensor &tensor, int dim) const -> Tensor override;
    [[nodiscard]] auto all(const Tensor &input) const -> bool override;
    [[nodiscard]] auto all(const Tensor &input, int dim) const -> Tensor override;
    [[nodiscard]] auto any(const Tensor &input) const -> bool override;
    [[nodiscard]] auto any(const Tensor &input, int dim) const -> Tensor override;

    // ------------------------------------------------
    // Unary operations
    // ------------------------------------------------
    [[nodiscard]] auto identity(const Tensor &tensor) const -> Tensor override;
    [[nodiscard]] auto negate(const Tensor &tensor) const -> Tensor override;
    [[nodiscard]] auto logical_not(const Tensor &tensor) const -> Tensor override;
    [[nodiscard]] auto abs(const Tensor &tensor) const -> Tensor override;
    [[nodiscard]] auto sign(const Tensor &tensor) const -> Tensor override;
    [[nodiscard]] auto log(const Tensor &tensor) const -> Tensor override;
    [[nodiscard]] auto log10(const Tensor &tensor) const -> Tensor override;
    [[nodiscard]] auto log2(const Tensor &tensor) const -> Tensor override;
    [[nodiscard]] auto log1p(const Tensor &tensor) const -> Tensor override;
    [[nodiscard]] auto exp(const Tensor &tensor) const -> Tensor override;
    [[nodiscard]] auto exp2(const Tensor &tensor) const -> Tensor override;
    [[nodiscard]] auto expm1(const Tensor &tensor) const -> Tensor override;
    [[nodiscard]] auto sqrt(const Tensor &tensor) const -> Tensor override;
    [[nodiscard]] auto sin(const Tensor &tensor) const -> Tensor override;
    [[nodiscard]] auto cos(const Tensor &tensor) const -> Tensor override;
    [[nodiscard]] auto tan(const Tensor &tensor) const -> Tensor override;
    [[nodiscard]] auto asin(const Tensor &tensor) const -> Tensor override;
    [[nodiscard]] auto acos(const Tensor &tensor) const -> Tensor override;
    [[nodiscard]] auto atan(const Tensor &tensor) const -> Tensor override;
    [[nodiscard]] auto sinh(const Tensor &tensor) const -> Tensor override;
    [[nodiscard]] auto cosh(const Tensor &tensor) const -> Tensor override;
    [[nodiscard]] auto tanh(const Tensor &tensor) const -> Tensor override;
    [[nodiscard]] auto asinh(const Tensor &tensor) const -> Tensor override;
    [[nodiscard]] auto acosh(const Tensor &tensor) const -> Tensor override;
    [[nodiscard]] auto atanh(const Tensor &tensor) const -> Tensor override;
    [[nodiscard]] auto erf(const Tensor &tensor) const -> Tensor override;
    [[nodiscard]] auto erfc(const Tensor &tensor) const -> Tensor override;
    [[nodiscard]] auto tgamma(const Tensor &tensor) const -> Tensor override;
    [[nodiscard]] auto lgamma(const Tensor &tensor) const -> Tensor override;
    [[nodiscard]] auto digamma(const Tensor &tensor) const -> Tensor override;
    [[nodiscard]] auto ceil(const Tensor &tensor) const -> Tensor override;
    [[nodiscard]] auto floor(const Tensor &tensor) const -> Tensor override;
    [[nodiscard]] auto round(const Tensor &tensor) const -> Tensor override;
    [[nodiscard]] auto isinf(const Tensor &tensor) const -> Tensor override;
    [[nodiscard]] auto isnan(const Tensor &tensor) const -> Tensor override;
    [[nodiscard]] auto isfinite(const Tensor &tensor) const -> Tensor override;

    void identity_(Tensor &tensor) const override;
    void negate_(Tensor &tensor) const override;
    void logical_not_(Tensor &tensor) const override;
    void abs_(Tensor &tensor) const override;
    void sign_(Tensor &tensor) const override;
    void log_(Tensor &tensor) const override;
    void log10_(Tensor &tensor) const override;
    void log2_(Tensor &tensor) const override;
    void log1p_(Tensor &tensor) const override;
    void exp_(Tensor &tensor) const override;
    void exp2_(Tensor &tensor) const override;
    void expm1_(Tensor &tensor) const override;
    void sqrt_(Tensor &tensor) const override;
    void sin_(Tensor &tensor) const override;
    void cos_(Tensor &tensor) const override;
    void tan_(Tensor &tensor) const override;
    void asin_(Tensor &tensor) const override;
    void acos_(Tensor &tensor) const override;
    void atan_(Tensor &tensor) const override;
    void sinh_(Tensor &tensor) const override;
    void cosh_(Tensor &tensor) const override;
    void tanh_(Tensor &tensor) const override;
    void asinh_(Tensor &tensor) const override;
    void acosh_(Tensor &tensor) const override;
    void atanh_(Tensor &tensor) const override;
    void erf_(Tensor &tensor) const override;
    void erfc_(Tensor &tensor) const override;
    void tgamma_(Tensor &tensor) const override;
    void lgamma_(Tensor &tensor) const override;
    void digamma_(Tensor &tensor) const override;
    void ceil_(Tensor &tensor) const override;
    void floor_(Tensor &tensor) const override;
    void round_(Tensor &tensor) const override;

    // ------------------------------------------------
    // Activation functions
    // ------------------------------------------------
    [[nodiscard]] auto sigmoid(const Tensor &tensor) const -> Tensor override;
    [[nodiscard]] auto log_sigmoid(const Tensor &tensor) const -> Tensor override;
    [[nodiscard]] auto hardsigmoid(const Tensor &tensor) const -> Tensor override;
    [[nodiscard]] auto softplus(const Tensor &tensor, double beta, double threshold) const -> Tensor override;
    [[nodiscard]] auto relu(const Tensor &tensor) const -> Tensor override;
    [[nodiscard]] auto relu6(const Tensor &tensor) const -> Tensor override;
    [[nodiscard]] auto leaky_relu(const Tensor &tensor, double negative_slope) const -> Tensor override;
    [[nodiscard]] auto elu(const Tensor &tensor, double alpha) const -> Tensor override;
    [[nodiscard]] auto selu(const Tensor &tensor) const -> Tensor override;
    [[nodiscard]] auto silu(const Tensor &tensor) const -> Tensor override;
    [[nodiscard]] auto hardtanh(const Tensor &tensor, double min, double max) const -> Tensor override;
    [[nodiscard]] auto softsign(const Tensor &tensor) const -> Tensor override;
    [[nodiscard]] auto softmax(const Tensor &tensor, int dim) const -> Tensor override;
    [[nodiscard]] auto log_softmax(const Tensor &tensor, int dim) const -> Tensor override;

    void sigmoid_(Tensor &tensor) const override;
    void log_sigmoid_(Tensor &tensor) const override;
    void hardsigmoid_(Tensor &tensor) const override;
    void softplus_(Tensor &tensor, double beta, double threshold) const override;
    void relu_(Tensor &tensor) const override;
    void relu6_(Tensor &tensor) const override;
    void leaky_relu_(Tensor &tensor, double negative_slope) const override;
    void elu_(Tensor &tensor, double alpha) const override;
    void selu_(Tensor &tensor) const override;
    void silu_(Tensor &tensor) const override;
    void hardtanh_(Tensor &tensor, double min, double max) const override;
    void softsign_(Tensor &tensor) const override;
    void softmax_(Tensor &tensor, int dim) const override;
    void log_softmax_(Tensor &tensor, int dim) const override;

    // ------------------------------------------------
    // Util/misc
    // ------------------------------------------------
    [[nodiscard]] auto where(const Tensor &cond, const Tensor &lhs, const Tensor &rhs) const -> Tensor override;
    void clamp_(Tensor &tensor, const Tensor &min, const Tensor &max) const override;

    [[nodiscard]] auto conv2d(
        const Tensor &input,
        const Tensor &weight,
        const std::optional<Tensor> &bias,
        int stride,
        int padding
    ) const -> Tensor override;
    [[nodiscard]] auto conv2d_backward(
        const Tensor &grad_output,
        const Tensor &input,
        const Tensor &weight,
        const std::optional<Tensor> &bias,
        int stride,
        int padding
    ) const -> std::tuple<Tensor, Tensor, std::optional<Tensor>> override;

    [[nodiscard]] auto max_pool2d(const Tensor &input, int kernel_size, int stride, int padding) const
        -> Tensor override;
    [[nodiscard]] auto max_pool2d_backward(
        const Tensor &grad_output,
        const Tensor &input,
        const Tensor &result,
        int kernel_size,
        int stride,
        int padding
    ) const -> Tensor override;
    [[nodiscard]] auto min_pool2d(const Tensor &input, int kernel_size, int stride, int padding) const
        -> Tensor override;
    [[nodiscard]] auto min_pool2d_backward(
        const Tensor &grad_output,
        const Tensor &input,
        const Tensor &result,
        int kernel_size,
        int stride,
        int padding
    ) const -> Tensor override;
    [[nodiscard]] auto avg_pool2d(const Tensor &input, int kernel_size, int stride, int padding) const
        -> Tensor override;
    [[nodiscard]] auto avg_pool2d_backward(
        const Tensor &grad_output,
        const Tensor &input,
        const Tensor &result,
        int kernel_size,
        int stride,
        int padding
    ) const -> Tensor override;

    [[nodiscard]] auto current_memory_allocated(int device_id) const -> uint64_t override;
    [[nodiscard]] auto total_memory_allocated(int device_id) const -> uint64_t override;
};

}    // namespace tinytensor::cpu

#endif    // TINYTENSOR_BACKEND_BASE_H_
