// conv2d.cpp
// Conv2d layer module

#include <tt/device.h>
#include <tt/exception.h>
#include <tt/grad_mode.h>
#include <tt/nn/conv2d.h>
#include <tt/nn/init.h>
#include <tt/scalar.h>
#include <tt/tensor.h>

#include <cmath>
#include <format>
#include <memory>
#include <optional>
#include <ostream>

namespace tinytensor::nn {

Conv2d::Conv2d(
    int in_channels,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    bool _bias,
    ScalarType dtype,
    Device device
)
    : weight(
          std::make_shared<Tensor>(zeros({out_channels, in_channels, kernel_size, kernel_size}, dtype, device, true))
      ),
      in_channels_(in_channels),
      out_channels_(out_channels),
      kernel_size_(kernel_size),
      stride_(stride),
      padding_(padding),
      has_bias_(_bias) {
    if (in_channels <= 0) {
        TT_EXCEPTION(std::format("Expected in_channels > 0, given in_channels={:d}", in_channels));
    }
    if (out_channels <= 0) {
        TT_EXCEPTION(std::format("Expected out_channels > 0, given out_channels={:d}", out_channels));
    }
    if (kernel_size <= 0) {
        TT_EXCEPTION(std::format("Expected kernel_size > 0, given kernel_size={:d}", kernel_size));
    }
    if (stride <= 0) {
        TT_EXCEPTION(std::format("Expected stride > 0, given stride={:d}", stride));
    }
    if (padding < 0) {
        TT_EXCEPTION(std::format("Expected padding >= 0, given padding={:d}", padding));
    }
    // Weight initialized by Kaiming uniform
    autograd::NoGradGuard guard;
    nn::kaiming_uniform_(weight->permute({1, 0, 2, 3}));
    register_param(weight);
    if (_bias) {
        bias = std::make_shared<Tensor>(zeros({out_channels}, dtype, device, true));
        const auto [fan_in, _] = calc_fan_in_out(*weight);
        double bound = 1.0 / std::sqrt(fan_in);
        nn::uniform_(*bias.value(), -bound, bound);
        register_param(bias.value());
    }
}

auto Conv2d::forward(const Tensor &input) const -> Tensor {
    // Redundant check but pretty error msg specific for this rathern then throwing inside conv2d
    if (input.dim() == 4 && input.size(1) != weight->size(1)) {
        TT_EXCEPTION(
            std::format("Input channels {:d} does not match layer input channels {:d}", input.size(1), weight->size(1))
        );
    }
    return conv2d(
        input,
        *weight,
        bias.has_value() ? std::make_optional<Tensor>(**bias) : std::nullopt,
        stride_,
        padding_
    );
}

void Conv2d::pretty_print(std::ostream &os) const {
    os << std::format(
        "Conv2d(in_channels={:d}, out_channels={:d}, kernel_size={:d}, stride={:d}, padding={:d}, bias={:}, dtype={}, "
        "device={})",
        in_channels_,
        out_channels_,
        kernel_size_,
        stride_,
        padding_,
        has_bias_,
        weight->dtype(),
        weight->device()
    );
}

}    // namespace tinytensor::nn
