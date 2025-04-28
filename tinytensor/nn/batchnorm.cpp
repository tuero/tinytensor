// batchnorm.cpp
// BatchNorm layer module

#include <tt/autograd.h>
#include <tt/device.h>
#include <tt/exception.h>
#include <tt/nn/batchnorm.h>
#include <tt/nn/init.h>
#include <tt/scalar.h>
#include <tt/tensor.h>

#include <cmath>
#include <format>
#include <memory>
#include <ostream>

namespace tinytensor::nn {

BatchNorm1d::BatchNorm1d(int num_features, const BatchNormOptions &options, ScalarType dtype, Device device)
    : gamma(std::make_shared<Tensor>(ones({1, num_features}, dtype, device, options.affine))),
      beta(std::make_shared<Tensor>(zeros({1, num_features}, dtype, device, options.affine))),
      moving_mean(std::make_shared<Tensor>(zeros({1, num_features}, dtype, device, false))),
      moving_var(std::make_shared<Tensor>(ones({1, num_features}, dtype, device, false))),
      num_features_(num_features),
      options_(options) {
    if (num_features <= 0) {
        TT_EXCEPTION(std::format("Expected num_features > 0, given num_features={:d}", num_features));
    }
    register_param(gamma);
    register_param(beta);
    register_param(moving_mean);
    register_param(moving_var);
}

auto BatchNorm1d::forward(const Tensor &x) -> Tensor {
    if (x.dim() != 2 && x.dim() != 3) {
        TT_EXCEPTION(std::format("Expected input to have 2 or 3 dimensions, given shape {}", x.shape()));
    }
    if (x.size(1) != num_features_) {
        TT_EXCEPTION(std::format("Expected size of input at dim 1 to be {}, given shape {}", num_features_, x.shape()));
    }
    if (is_train_ || !options_.track_running_stats) {
        Tensor mean = x.dim() == 2 ? x.mean(0, true) : x.mean({0, 2}, true);
        Tensor var_biased = (x.dim() == 2 ? x.var(0, true, 0) : x.var({0, 2}, true, 0)).expand(x.shape());
        Tensor var_unbiased = x.dim() == 2 ? x.var(0, true, 1) : x.var({0, 2}, true, 1);
        Tensor x_hat = (x - mean.expand(x.shape())) / sqrt(var_biased + options_.eps);
        if (options_.track_running_stats) {
            moving_mean->mul_(1.0 - options_.momentum).add_(options_.momentum * mean.reshape(moving_mean->shape()));
            moving_var->mul_(1.0 - options_.momentum)
                .add_(options_.momentum * var_unbiased.reshape(moving_var->shape()));
        }
        Tensor g = x.dim() == 2 ? *gamma : gamma->unsqueeze(-1);
        Tensor b = x.dim() == 2 ? *beta : beta->unsqueeze(-1);
        return g.expand(x.shape()) * x_hat + b.expand(x.shape());
    } else {
        Tensor m = x.dim() == 2 ? *moving_mean : moving_mean->unsqueeze(-1);
        Tensor v = x.dim() == 2 ? *moving_var : moving_var->unsqueeze(-1);
        Tensor x_hat = (x - m.expand(x.shape())) / sqrt(v.expand(x.shape()) + options_.eps);
        Tensor g = x.dim() == 2 ? *gamma : gamma->unsqueeze(-1);
        Tensor b = x.dim() == 2 ? *beta : beta->unsqueeze(-1);
        Tensor result = g.expand(x.shape()) * x_hat + b.expand(x.shape());
        return result;
    }
}

void BatchNorm1d::pretty_print(std::ostream &os) const {
    os << std::format(
        "BatchNorm1d(num_features={:d}, eps={:f}, momentum={:f}, affine={}, track_running_stats={}, dtype={}, "
        "device={})",
        num_features_,
        options_.eps,
        options_.momentum,
        options_.affine,
        options_.track_running_stats,
        gamma->dtype(),
        gamma->device()
    );
}

// -----------------------------------------------

BatchNorm2d::BatchNorm2d(int num_features, const BatchNormOptions &options, ScalarType dtype, Device device)
    : gamma(std::make_shared<Tensor>(ones({1, num_features, 1, 1}, dtype, device, options.affine))),
      beta(std::make_shared<Tensor>(zeros({1, num_features, 1, 1}, dtype, device, options.affine))),
      moving_mean(std::make_shared<Tensor>(zeros({1, num_features, 1, 1}, dtype, device, false))),
      moving_var(std::make_shared<Tensor>(ones({1, num_features, 1, 1}, dtype, device, false))),
      num_features_(num_features),
      options_(options) {
    if (num_features <= 0) {
        TT_EXCEPTION(std::format("Expected num_features > 0, given num_features={:d}", num_features));
    }
    register_param(gamma);
    register_param(beta);
    register_param(moving_mean);
    register_param(moving_var);
}

auto BatchNorm2d::forward(const Tensor &x) -> Tensor {
    if (x.dim() != 4) {
        TT_EXCEPTION(std::format("Expected input to have 4 dimensions, given shape {}", x.shape()));
    }
    if (x.size(1) != num_features_) {
        TT_EXCEPTION(std::format("Expected size of input at dim 1 to be {}, given shape {}", num_features_, x.shape()));
    }
    if (is_train_ || !options_.track_running_stats) {
        Tensor mean = x.mean({0, 2, 3}, true);
        Tensor var_biased = x.var({0, 2, 3}, true, 0).expand(x.shape());
        Tensor var_unbiased = x.var({0, 2, 3}, true, 1);
        Tensor x_hat = (x - mean.expand(x.shape())) / sqrt(var_biased + options_.eps);
        if (options_.track_running_stats) {
            moving_mean->mul_(1.0 - options_.momentum).add_(options_.momentum * mean);
            moving_var->mul_(1.0 - options_.momentum).add_(options_.momentum * var_unbiased);
        }
        return gamma->expand(x.shape()) * x_hat + beta->expand(x.shape());
    } else {
        Tensor x_hat = (x - moving_mean->expand(x.shape())) / sqrt(moving_var->expand(x.shape()) + options_.eps);
        return gamma->expand(x.shape()) * x_hat + beta->expand(x.shape());
    }
}

void BatchNorm2d::pretty_print(std::ostream &os) const {
    os << std::format(
        "BatchNorm2d(num_features={:d}, eps={:f}, momentum={:f}, affine={}, track_running_stats={}, dtype={}, "
        "device={})",
        num_features_,
        options_.eps,
        options_.momentum,
        options_.affine,
        options_.track_running_stats,
        gamma->dtype(),
        gamma->device()
    );
}

}    // namespace tinytensor::nn
