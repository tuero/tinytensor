// intancenorm.cpp
// InstanceNorm layer module

#include <tt/autograd.h>
#include <tt/device.h>
#include <tt/exception.h>
#include <tt/nn/init.h>
#include <tt/nn/instancenorm.h>
#include <tt/scalar.h>
#include <tt/tensor.h>

#include <cmath>
#include <format>
#include <memory>
#include <ostream>

namespace tinytensor::nn {

InstanceNorm1d::InstanceNorm1d(int num_features, const InstanceNormOptions &options, ScalarType dtype, Device device)
    : gamma(std::make_shared<Tensor>(ones({1, num_features, 1}, dtype, device, options.affine))),
      beta(std::make_shared<Tensor>(zeros({1, num_features, 1}, dtype, device, options.affine))),
      moving_mean(std::make_shared<Tensor>(zeros({1, num_features, 1}, dtype, device, false))),
      moving_var(std::make_shared<Tensor>(ones({1, num_features, 1}, dtype, device, false))),
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

auto InstanceNorm1d::forward(const Tensor &x) -> Tensor {
    if (x.dim() != 3) {
        TT_EXCEPTION(std::format("Expected input to have 3 dimensions, given shape {}", x.shape()));
    }
    if (x.size(1) != num_features_) {
        TT_EXCEPTION(std::format("Expected size of input at dim 1 to be {}, given shape {}", num_features_, x.shape()));
    }
    // Instance norm takes stats over the inner dims (stats per channel)
    // stats are then averaged over the batch
    // https://discuss.pytorch.org/t/understanding-instance-normalization-2d-with-running-mean-and-running-var/144139/2?u=tuero
    if (is_train_ || !options_.track_running_stats) {
        Tensor mean = x.mean(-1, true);
        Tensor var_biased = x.var(-1, true, 0).expand(x.shape());
        Tensor var_unbiased = x.var(-1, true, 1);
        Tensor x_hat = (x - mean.expand(x.shape())) / sqrt(var_biased + options_.eps);
        if (options_.track_running_stats) {
            moving_mean->mul_(1.0 - options_.momentum).add_(options_.momentum * mean.mean(0, true));
            moving_var->mul_(1.0 - options_.momentum).add_(options_.momentum * var_unbiased.mean(0, true));
        }
        return gamma->expand(x.shape()) * x_hat + beta->expand(x.shape());
    } else {
        Tensor x_hat = (x - moving_mean->expand(x.shape())) / sqrt(moving_var->expand(x.shape()) + options_.eps);
        return gamma->expand(x.shape()) * x_hat + beta->expand(x.shape());
    }
}

void InstanceNorm1d::pretty_print(std::ostream &os) const {
    os << std::format(
        "InstanceNorm1d(num_features={:d}, eps={:f}, momentum={:f}, affine={}, track_running_stats={}, dtype={}, "
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

InstanceNorm2d::InstanceNorm2d(int num_features, const InstanceNormOptions &options, ScalarType dtype, Device device)
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

auto InstanceNorm2d::forward(const Tensor &x) -> Tensor {
    if (x.dim() != 4) {
        TT_EXCEPTION(std::format("Expected input to have 4 dimensions, given shape {}", x.shape()));
    }
    if (x.size(1) != num_features_) {
        TT_EXCEPTION(std::format("Expected size of input at dim 1 to be {}, given shape {}", num_features_, x.shape()));
    }
    // Instance norm takes stats over the inner dims (stats per channel)
    // stats are then averaged over the batch
    // https://discuss.pytorch.org/t/understanding-instance-normalization-2d-with-running-mean-and-running-var/144139/2?u=tuero
    if (is_train_ || !options_.track_running_stats) {
        Tensor mean = x.mean({2, 3}, true);
        Tensor var_biased = x.var({2, 3}, true, 0).expand(x.shape());
        Tensor var_unbiased = x.var({2, 3}, true, 1);
        Tensor x_hat = (x - mean.expand(x.shape())) / sqrt(var_biased + options_.eps);
        if (options_.track_running_stats) {
            moving_mean->mul_(1.0 - options_.momentum).add_(options_.momentum * mean.mean(0, true));
            moving_var->mul_(1.0 - options_.momentum).add_(options_.momentum * var_unbiased.mean(0, true));
        }
        return gamma->expand(x.shape()) * x_hat + beta->expand(x.shape());
    } else {
        Tensor x_hat = (x - moving_mean->expand(x.shape())) / sqrt(moving_var->expand(x.shape()) + options_.eps);
        return gamma->expand(x.shape()) * x_hat + beta->expand(x.shape());
    }
}

void InstanceNorm2d::pretty_print(std::ostream &os) const {
    os << std::format(
        "InstanceNorm2d(num_features={:d}, eps={:f}, momentum={:f}, affine={}, track_running_stats={}, dtype={}, "
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
