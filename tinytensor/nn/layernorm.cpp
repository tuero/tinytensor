// layernorm.cpp
// LayerNorm layer module

#include <tt/autograd.h>
#include <tt/device.h>
#include <tt/exception.h>
#include <tt/nn/init.h>
#include <tt/nn/layernorm.h>
#include <tt/scalar.h>
#include <tt/shape.h>
#include <tt/tensor.h>

#include <cmath>
#include <format>
#include <memory>
#include <ostream>
#include <ranges>
#include <vector>

namespace tinytensor::nn {

LayerNorm::LayerNorm(const Shape &normalized_shape, const LayerNormOptions &options, ScalarType dtype, Device device)
    : gamma(std::make_shared<Tensor>(ones(normalized_shape, dtype, device, options.affine))),
      beta(std::make_shared<Tensor>(zeros(normalized_shape, dtype, device, options.bias))),
      normalized_shape_(normalized_shape),
      options_(options) {
    int N = normalized_shape_.ndim();
    for (int i : std::views::iota(0, N) | std::views::transform([](int idx) -> int { return -idx - 1; })) {
        normalized_dims_.push_back(i);
    }
    register_param(gamma);
    register_param(beta);
}

auto LayerNorm::forward(const Tensor &x) -> Tensor {
    int N = normalized_shape_.ndim();
    if (x.dim() < N) {
        TT_EXCEPTION(
            std::format(
                "Expected input to have number of dimensions at least as large as normalized_shape {}, given shape {}",
                normalized_shape_,
                x.shape()
            )
        );
    }
    // Check last N dims of input if it matches expected shape
    for (const auto &i : normalized_dims_) {
        if (x.size(i) != normalized_shape_[i]) {
            TT_EXCEPTION(
                std::format(
                    "Expected last {:d} dimensions of input to match normalized shape {}, given shape {}",
                    N,
                    normalized_shape_,
                    x.shape()
                )
            );
        }
    }

    Tensor mean = x.mean(normalized_dims_, true).expand(x.shape());
    Tensor var = x.var(normalized_dims_, true, 0).expand(x.shape());
    Tensor x_hat = (x - mean) / sqrt(var + options_.eps);
    return gamma->expand(x.shape()) * x_hat + beta->expand(x.shape());
}

void LayerNorm::pretty_print(std::ostream &os) const {
    os << std::format(
        "LayerNorm(normalized_shape={:s}, eps={:f}, affine={}, bias={}, dtype={}, device={})",
        normalized_shape_,
        options_.eps,
        options_.affine,
        options_.bias,
        gamma->dtype(),
        gamma->device()
    );
}

}    // namespace tinytensor::nn
