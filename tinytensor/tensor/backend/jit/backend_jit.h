#pragma once
#include <tensor/backend_base.h>
#include <tt/exception.h>

namespace tinytensor {

class BackendJIT : public BackendBase {
public:
    BackendJIT() = default;

    [[nodiscard]] auto relu(const Tensor &tensor) const -> Tensor override;
    [[nodiscard]] auto add(const Tensor &lhs, const Tensor &rhs) const -> Tensor override;
    
    [[noreturn]] void not_impl(const char* name) const {
        std::string msg = std::string("JIT Backend: Function not implemented: ") + name;
        TT_ERROR(msg.c_str());
    }

    // Since not_impl is [[noreturn]], don't need to return a dummy value.
    // this fixes the "Cannot convert braced-init-list" error for Tensor.
#define STUB_RET(ret_type, name, ...) \
    [[nodiscard]] auto name(__VA_ARGS__) const -> ret_type override { not_impl(#name); }

#define STUB_VOID(name, ...) \
    void name(__VA_ARGS__) const override { not_impl(#name); }

    // STUBBED INTERFACE (Satisfies Abstract Class)

    // Tensor Construction
    STUB_RET(StoragePtr, from_vec, const std::vector<bool> &data, int device_id)
    STUB_RET(StoragePtr, from_vec, const std::vector<uint8_t> &data, int device_id)
    STUB_RET(StoragePtr, from_vec, std::vector<uint8_t> &&data, int device_id)
    STUB_RET(StoragePtr, from_vec, const std::vector<int16_t> &data, int device_id)
    STUB_RET(StoragePtr, from_vec, std::vector<int16_t> &&data, int device_id)
    STUB_RET(StoragePtr, from_vec, const std::vector<int32_t> &data, int device_id)
    STUB_RET(StoragePtr, from_vec, std::vector<int32_t> &&data, int device_id)
    STUB_RET(StoragePtr, from_vec, const std::vector<int64_t> &data, int device_id)
    STUB_RET(StoragePtr, from_vec, std::vector<int64_t> &&data, int device_id)
    STUB_RET(StoragePtr, from_vec, const std::vector<float> &data, int device_id)
    STUB_RET(StoragePtr, from_vec, std::vector<float> &&data, int device_id)
    STUB_RET(StoragePtr, from_vec, const std::vector<double> &data, int device_id)
    STUB_RET(StoragePtr, from_vec, std::vector<double> &&data, int device_id)
    STUB_RET(StoragePtr, from_scalar, Scalar scalar, int device_id)
    STUB_RET(StoragePtr, full, const Scalar &value, std::size_t N, int device_id)
    STUB_RET(StoragePtr, arange, std::size_t N, ScalarType dtype, int device_id)

    // Conversions
    STUB_VOID(to_vec, const Tensor &tensor, std::vector<uint8_t> &data_out)
    STUB_VOID(to_vec, const Tensor &tensor, std::vector<int16_t> &data_out)
    STUB_VOID(to_vec, const Tensor &tensor, std::vector<int32_t> &data_out)
    STUB_VOID(to_vec, const Tensor &tensor, std::vector<int64_t> &data_out)
    STUB_VOID(to_vec, const Tensor &tensor, std::vector<float> &data_out)
    STUB_VOID(to_vec, const Tensor &tensor, std::vector<double> &data_out)

    // Getters
    STUB_RET(Scalar, item, const Tensor &tensor)
    [[nodiscard]] auto data_ptr(const Tensor &tensor) const -> uintptr_t override { return 0; }

    // Indexing
    STUB_RET(Tensor, index_mask, const Tensor &input, const Tensor &mask, int N_mask)
    STUB_RET(Tensor, index_indices, const Tensor &input, const Tensor &indices)
    STUB_VOID(index_put_mask, Tensor &input, const Tensor &values, const Tensor &mask)
    STUB_VOID(index_put_indices, Tensor &input, const Tensor &values, const Tensor &indices)
    STUB_RET(Tensor, gather, const Tensor &input, const Tensor &indices, int idx)

    // Casts
    STUB_RET(Tensor, to, const Tensor &tensor, ScalarType dtype)
    auto print(std::ostream &os, const Tensor &tensor) const -> std::ostream & override { return os << "JIT Tensor"; }
    STUB_VOID(assign, Tensor &lhs, const Tensor &rhs)

    // Distributions
    STUB_RET(Tensor, uniform_int, const Tensor &low, const Tensor &high, Generator &gen)
    STUB_RET(Tensor, uniform_real, const Tensor &low, const Tensor &high, Generator &gen)
    STUB_RET(Tensor, bernoulli, const Tensor &p, Generator &gen)
    STUB_RET(Tensor, binomial, const Tensor &p, const Tensor &num_draws, Generator &gen)
    STUB_RET(Tensor, geometric, const Tensor &p, Generator &gen)
    STUB_RET(Tensor, poisson, const Tensor &lambda, Generator &gen)
    STUB_RET(Tensor, exponential, const Tensor &lambda, Generator &gen)
    STUB_RET(Tensor, normal, const Tensor &mu, const Tensor &std, Generator &gen)
    STUB_RET(Tensor, cauchy, const Tensor &loc, const Tensor &scale, Generator &gen)
    STUB_RET(Tensor, lognormal, const Tensor &mu, const Tensor &std, Generator &gen)
    STUB_RET(Tensor, weibull, const Tensor &scale, const Tensor &shape, Generator &gen)

    STUB_VOID(uniform_int_, Tensor &tensor, const Tensor &low, const Tensor &high, Generator &gen)
    STUB_VOID(uniform_real_, Tensor &tensor, const Tensor &low, const Tensor &high, Generator &gen)
    STUB_VOID(bernoulli_, Tensor &tensor, const Tensor &p, Generator &gen)
    STUB_VOID(binomial_, Tensor &tensor, const Tensor &p, const Tensor &num_draws, Generator &gen)
    STUB_VOID(geometric_, Tensor &tensor, const Tensor &p, Generator &gen)
    STUB_VOID(poisson_, Tensor &tensor, const Tensor &lambda, Generator &gen)
    STUB_VOID(exponential_, Tensor &tensor, const Tensor &lambda, Generator &gen)
    STUB_VOID(normal_, Tensor &tensor, const Tensor &mu, const Tensor &std, Generator &gen)
    STUB_VOID(cauchy_, Tensor &tensor, const Tensor &loc, const Tensor &scale, Generator &gen)
    STUB_VOID(lognormal_, Tensor &tensor, const Tensor &mu, const Tensor &std, Generator &gen)
    STUB_VOID(weibull_, Tensor &tensor, const Tensor &scale, const Tensor &shape, Generator &gen)

    // Binary Ops
    STUB_RET(Tensor, eq, const Tensor &lhs, const Tensor &rhs)
    STUB_RET(Tensor, ne, const Tensor &lhs, const Tensor &rhs)
    STUB_RET(Tensor, lt, const Tensor &lhs, const Tensor &rhs)
    STUB_RET(Tensor, le, const Tensor &lhs, const Tensor &rhs)
    STUB_RET(Tensor, gt, const Tensor &lhs, const Tensor &rhs)
    STUB_RET(Tensor, ge, const Tensor &lhs, const Tensor &rhs)
    STUB_RET(Tensor, logical_or, const Tensor &lhs, const Tensor &rhs)
    STUB_RET(Tensor, logical_and, const Tensor &lhs, const Tensor &rhs)
    STUB_RET(Tensor, bitwise_or, const Tensor &lhs, const Tensor &rhs)
    STUB_RET(Tensor, bitwise_and, const Tensor &lhs, const Tensor &rhs)
    STUB_RET(Tensor, bitwise_xor, const Tensor &lhs, const Tensor &rhs)
    STUB_RET(Tensor, bitwise_left_shift, const Tensor &lhs, const Tensor &rhs)
    STUB_RET(Tensor, bitwise_right_shift, const Tensor &lhs, const Tensor &rhs)
    STUB_RET(Tensor, modulo, const Tensor &lhs, const Tensor &rhs)
    STUB_RET(Tensor, sub, const Tensor &lhs, const Tensor &rhs)
    STUB_RET(Tensor, mul, const Tensor &lhs, const Tensor &rhs)
    STUB_RET(Tensor, div, const Tensor &lhs, const Tensor &rhs)
    STUB_RET(Tensor, maximum, const Tensor &lhs, const Tensor &rhs)
    STUB_RET(Tensor, minimum, const Tensor &lhs, const Tensor &rhs)
    STUB_RET(Tensor, pow, const Tensor &lhs, const Tensor &rhs)
    STUB_RET(Tensor, batched_matmul, const Tensor &lhs, const Tensor &rhs)

    STUB_VOID(add_inplace, Tensor &lhs, const Tensor &rhs)
    STUB_VOID(sub_inplace, Tensor &lhs, const Tensor &rhs)
    STUB_VOID(mul_inplace, Tensor &lhs, const Tensor &rhs)
    STUB_VOID(div_inplace, Tensor &lhs, const Tensor &rhs)

    // Reductions
    STUB_RET(Tensor, min, const Tensor &tensor)
    STUB_RET(Tensor, min, const Tensor &tensor, int dim)
    STUB_RET(Tensor, argmin, const Tensor &tensor)
    STUB_RET(Tensor, argmin, const Tensor &tensor, int dim)
    STUB_RET(Tensor, max, const Tensor &tensor)
    STUB_RET(Tensor, max, const Tensor &tensor, int dim)
    STUB_RET(Tensor, argmax, const Tensor &tensor)
    STUB_RET(Tensor, argmax, const Tensor &tensor, int dim)
    STUB_RET(Tensor, sum, const Tensor &tensor)
    STUB_RET(Tensor, sum, const Tensor &tensor, int dim)
    STUB_RET(bool, all, const Tensor &tensor)
    STUB_RET(Tensor, all, const Tensor &tensor, int dim)
    STUB_RET(bool, any, const Tensor &tensor)
    STUB_RET(Tensor, any, const Tensor &tensor, int dim)

    // Unary Ops
    STUB_RET(Tensor, identity, const Tensor &tensor)
    STUB_RET(Tensor, negate, const Tensor &tensor)
    STUB_RET(Tensor, logical_not, const Tensor &tensor)
    STUB_RET(Tensor, abs, const Tensor &tensor)
    STUB_RET(Tensor, sign, const Tensor &tensor)
    STUB_RET(Tensor, log, const Tensor &tensor)
    STUB_RET(Tensor, log10, const Tensor &tensor)
    STUB_RET(Tensor, log2, const Tensor &tensor)
    STUB_RET(Tensor, log1p, const Tensor &tensor)
    STUB_RET(Tensor, exp, const Tensor &tensor)
    STUB_RET(Tensor, exp2, const Tensor &tensor)
    STUB_RET(Tensor, expm1, const Tensor &tensor)
    STUB_RET(Tensor, sqrt, const Tensor &tensor)
    STUB_RET(Tensor, sin, const Tensor &tensor)
    STUB_RET(Tensor, cos, const Tensor &tensor)
    STUB_RET(Tensor, tan, const Tensor &tensor)
    STUB_RET(Tensor, asin, const Tensor &tensor)
    STUB_RET(Tensor, acos, const Tensor &tensor)
    STUB_RET(Tensor, atan, const Tensor &tensor)
    STUB_RET(Tensor, sinh, const Tensor &tensor)
    STUB_RET(Tensor, cosh, const Tensor &tensor)
    STUB_RET(Tensor, tanh, const Tensor &tensor)
    STUB_RET(Tensor, asinh, const Tensor &tensor)
    STUB_RET(Tensor, acosh, const Tensor &tensor)
    STUB_RET(Tensor, atanh, const Tensor &tensor)
    STUB_RET(Tensor, erf, const Tensor &tensor)
    STUB_RET(Tensor, erfc, const Tensor &tensor)
    STUB_RET(Tensor, tgamma, const Tensor &tensor)
    STUB_RET(Tensor, lgamma, const Tensor &tensor)
    STUB_RET(Tensor, digamma, const Tensor &tensor)
    STUB_RET(Tensor, ceil, const Tensor &tensor)
    STUB_RET(Tensor, floor, const Tensor &tensor)
    STUB_RET(Tensor, round, const Tensor &tensor)
    STUB_RET(Tensor, isinf, const Tensor &tensor)
    STUB_RET(Tensor, isnan, const Tensor &tensor)
    STUB_RET(Tensor, isfinite, const Tensor &tensor)

    STUB_VOID(identity_, Tensor &tensor)
    STUB_VOID(negate_, Tensor &tensor)
    STUB_VOID(logical_not_, Tensor &tensor)
    STUB_VOID(abs_, Tensor &tensor)
    STUB_VOID(sign_, Tensor &tensor)
    STUB_VOID(log_, Tensor &tensor)
    STUB_VOID(log10_, Tensor &tensor)
    STUB_VOID(log2_, Tensor &tensor)
    STUB_VOID(log1p_, Tensor &tensor)
    STUB_VOID(exp_, Tensor &tensor)
    STUB_VOID(exp2_, Tensor &tensor)
    STUB_VOID(expm1_, Tensor &tensor)
    STUB_VOID(sqrt_, Tensor &tensor)
    STUB_VOID(sin_, Tensor &tensor)
    STUB_VOID(cos_, Tensor &tensor)
    STUB_VOID(tan_, Tensor &tensor)
    STUB_VOID(asin_, Tensor &tensor)
    STUB_VOID(acos_, Tensor &tensor)
    STUB_VOID(atan_, Tensor &tensor)
    STUB_VOID(sinh_, Tensor &tensor)
    STUB_VOID(cosh_, Tensor &tensor)
    STUB_VOID(tanh_, Tensor &tensor)
    STUB_VOID(asinh_, Tensor &tensor)
    STUB_VOID(acosh_, Tensor &tensor)
    STUB_VOID(atanh_, Tensor &tensor)
    STUB_VOID(erf_, Tensor &tensor)
    STUB_VOID(erfc_, Tensor &tensor)
    STUB_VOID(tgamma_, Tensor &tensor)
    STUB_VOID(lgamma_, Tensor &tensor)
    STUB_VOID(digamma_, Tensor &tensor)
    STUB_VOID(ceil_, Tensor &tensor)
    STUB_VOID(floor_, Tensor &tensor)
    STUB_VOID(round_, Tensor &tensor)

    // Activations
    STUB_RET(Tensor, sigmoid, const Tensor &tensor)
    STUB_RET(Tensor, log_sigmoid, const Tensor &tensor)
    STUB_RET(Tensor, hardsigmoid, const Tensor &tensor)
    STUB_RET(Tensor, softplus, const Tensor &tensor, double beta, double threshold)
    STUB_RET(Tensor, relu6, const Tensor &tensor)
    STUB_RET(Tensor, leaky_relu, const Tensor &tensor, double negative_slope)
    STUB_RET(Tensor, elu, const Tensor &tensor, double alpha)
    STUB_RET(Tensor, selu, const Tensor &tensor)
    STUB_RET(Tensor, silu, const Tensor &tensor)
    STUB_RET(Tensor, hardtanh, const Tensor &tensor, double min, double max)
    STUB_RET(Tensor, softsign, const Tensor &tensor)
    STUB_RET(Tensor, softmax, const Tensor &tensor, int dim)
    STUB_RET(Tensor, log_softmax, const Tensor &tensor, int dim)

    STUB_VOID(sigmoid_, Tensor &tensor)
    STUB_VOID(log_sigmoid_, Tensor &tensor)
    STUB_VOID(hardsigmoid_, Tensor &tensor)
    STUB_VOID(softplus_, Tensor &tensor, double beta, double threshold)
    STUB_VOID(relu_, Tensor &tensor)
    STUB_VOID(relu6_, Tensor &tensor)
    STUB_VOID(leaky_relu_, Tensor &tensor, double negative_slope)
    STUB_VOID(elu_, Tensor &tensor, double alpha)
    STUB_VOID(selu_, Tensor &tensor)
    STUB_VOID(silu_, Tensor &tensor)
    STUB_VOID(hardtanh_, Tensor &tensor, double min, double max)
    STUB_VOID(softsign_, Tensor &tensor)
    STUB_VOID(softmax_, Tensor &tensor, int dim)
    STUB_VOID(log_softmax_, Tensor &tensor, int dim)

    // Utils
    STUB_RET(Tensor, where, const Tensor &cond, const Tensor &lhs, const Tensor &rhs)
    STUB_VOID(clamp_, Tensor &tensor, const Tensor &min, const Tensor &max)
    STUB_RET(Tensor, conv2d, const Tensor &input, const Tensor &weight, const std::optional<Tensor> &bias, int stride, int padding)

    // complex return type needs manual stub
    auto conv2d_backward(const Tensor &grad_output, const Tensor &input, const Tensor &weight, const std::optional<Tensor> &bias, int stride, int padding) const -> std::tuple<Tensor, Tensor, std::optional<Tensor>> override { not_impl("conv2d_backward"); }

    STUB_RET(Tensor, max_pool2d, const Tensor &input, int kernel_size, int stride, int padding)
    STUB_RET(Tensor, max_pool2d_backward, const Tensor &grad_output, const Tensor &input, const Tensor &result, int kernel_size, int stride, int padding)
    STUB_RET(Tensor, min_pool2d, const Tensor &input, int kernel_size, int stride, int padding)
    STUB_RET(Tensor, min_pool2d_backward, const Tensor &grad_output, const Tensor &input, const Tensor &result, int kernel_size, int stride, int padding)
    STUB_RET(Tensor, avg_pool2d, const Tensor &input, int kernel_size, int stride, int padding)
    STUB_RET(Tensor, avg_pool2d_backward, const Tensor &grad_output, const Tensor &input, const Tensor &result, int kernel_size, int stride, int padding)

    STUB_RET(uint64_t, current_memory_allocated, int device_id)
    STUB_RET(uint64_t, total_memory_allocated, int device_id)

#undef STUB_RET
#undef STUB_VOID
};

} // namespace tinytensor