// init.cpp
// Initialization methods

#include <tt/exception.h>
#include <tt/grad_mode.h>
#include <tt/nn/init.h>
#include <tt/tensor.h>
#include <tt/util.h>

#include <cmath>
#include <format>
#include <numbers>
#include <tuple>

namespace tinytensor::nn {

namespace {
void uniform_real_no_grad(Tensor tensor, double low, double high) {
    autograd::NoGradGuard guard;
    tensor.uniform_real_(low, high);
}

void normal_no_grad(Tensor tensor, double mu, double std) {
    autograd::NoGradGuard guard;
    tensor.normal_(mu, std);
}
}    // namespace

// Assumes dim(0) is the size of the input dimension, dim(1) is the size of the output dimension
auto calc_fan_in_out(const Tensor &tensor) -> std::tuple<double, double> {
    auto dimensions = tensor.dim();
    if (dimensions < 2) {
        TT_EXCEPTION(std::format(
            "Unable to calculcate fan in and fan out for tensors with dimensions fewer than 2. Given tensor with {:d} "
            "dimensions",
            dimensions
        ));
    }
    auto input_features = tensor.size(0);
    auto output_features = tensor.size(0);
    int receptive_field_size = 1;
    for (auto i = 2; i < dimensions; ++i) {
        receptive_field_size *= tensor.size(i);
    }
    return {
        static_cast<double>(input_features * receptive_field_size),
        static_cast<double>(output_features * receptive_field_size)
    };
}

auto calc_gain(GainActivation gain_activation, const GainActivationParams &params) -> double {
    constexpr auto negative_slope_param_str = "negative_slope";
    switch (gain_activation) {
    case GainActivation::linear:
        return 1;
    case GainActivation::conv:
        return 1;
    case GainActivation::sigmoid:
        return 1;
    case GainActivation::tanh:
        return 5.0 / 3.0;
    case GainActivation::relu:
        return std::numbers::sqrt2;
    case GainActivation::leaky_relu:
        if (!params.contains(negative_slope_param_str)) {
            TT_EXCEPTION("leaky_relu gain activation function requires 'negative_slope' as an additional param");
        }
        return std::sqrt(2.0 / (1 + std::pow(params.at(negative_slope_param_str), 2)));
    case GainActivation::selu:
        return 3.0 / 4.0;
    }
    unreachable();
}

void uniform_(Tensor tensor, double low, double high) {
    uniform_real_no_grad(tensor, low, high);
}

void normal_(Tensor tensor, double mu, double std) {
    normal_no_grad(tensor, mu, std);
}

void constant_(Tensor tensor, double value) {
    autograd::NoGradGuard guard;
    tensor = value;
}

// U(-a, a) where a = gain * sqrt(6 / (fan_in + fan_out))
void xavier_uniform_(Tensor tensor, double gain) {
    const auto [fan_in, fan_out] = calc_fan_in_out(tensor);
    double a = gain * std::sqrt(6.0 / (fan_in + fan_out));
    uniform_real_no_grad(tensor, -a, a);
}

// N(0, std^2) where std = gain * sqrt(2 / (fan_in + fan_out))
void xavier_normal_(Tensor tensor, double gain) {
    const auto [fan_in, fan_out] = calc_fan_in_out(tensor);
    double std = gain * std::sqrt(2.0 / (fan_in + fan_out));
    normal_no_grad(tensor, 0, std);
}

// U(-bound, boun) where bound = sqrt(3) * (gain / sqrt(fan_mode))
void kaiming_uniform_(Tensor tensor, double gain, FanMode fan_mode) {
    const auto [fan_in, fan_out] = calc_fan_in_out(tensor);
    auto fan = fan_mode == FanMode::fan_in ? fan_in : fan_out;
    double std = gain / std::sqrt(fan);
    double bound = std::numbers::sqrt3 * std;
    uniform_real_no_grad(tensor, -bound, bound);
}

// N(-bound, boun) where std = gain / sqrt(fan_mode)
void kaiming_normal_(Tensor tensor, double gain, FanMode fan_mode) {
    const auto [fan_in, fan_out] = calc_fan_in_out(tensor);
    auto fan = fan_mode == FanMode::fan_in ? fan_in : fan_out;
    double std = gain / std::sqrt(fan);
    normal_no_grad(tensor, 0, std);
}

}    // namespace tinytensor::nn
