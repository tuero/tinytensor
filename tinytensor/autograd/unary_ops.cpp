// unary_ops.cpp
// Unary operations with autograd support

#include "autograd/unary_ops.h"

#include <tt/autograd.h>
#include <tt/device.h>
#include <tt/exception.h>
#include <tt/grad_mode.h>
#include <tt/scalar.h>
#include <tt/tensor.h>

#include "tensor/backend/common/dispatch.h"
#include "tensor/backend_register.h"

#include <format>
#include <numbers>

namespace tinytensor::autograd {

// NOLINTNEXTLINE(*-macro-usage)
#define CHECK_VERSION(tensor, version)                                                                               \
    if (tensor.version_count() != version) {                                                                         \
        TT_EXCEPTION(std::format(                                                                                    \
            "Inplace operation on tensor required for autograd detected. Tensor with version {:d} saved in forward " \
            "pass, but has version {:d} in backward pass",                                                           \
            version,                                                                                                 \
            tensor.version_count()                                                                                   \
        ));                                                                                                          \
    }

// Element-wise clone forward and backward
auto TensorClone::forward(
    [[maybe_unused]] AutogradStorage &storage,
    [[maybe_unused]] bool is_grad_required,
    const Tensor &tensor
) -> Tensor {
    return get_backend(tensor.device())->identity(tensor);
}
auto TensorClone::backward([[maybe_unused]] const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    return {grad_output};
}

// Element-wise to type forward and backward
// Gradient is to convert back to input type
auto TensorToScalarType::forward(
    [[maybe_unused]] AutogradStorage &storage,
    [[maybe_unused]] bool is_grad_required,
    const Tensor &tensor,
    ScalarType dtype
) -> Tensor {
    if (is_grad_required) {
        storage["dtype"] = tensor.dtype();
    }
    // If we are casting to bool, clamp to 0-1 since its using uint8_t under the hood
    auto result = (dtype == kBool) ? tensor.clamp(ClampOptions().min(0).max(1)) : tensor;
    return get_backend(result.device())->to(result, dtype);
}
auto TensorToScalarType::backward([[maybe_unused]] const AutogradStorage &storage, const Tensor &grad_output)
    -> GradList {
    const auto &dtype = std::get<ScalarType>(storage.at("dtype"));
    return {grad_output.to(dtype)};
}

// Element-wise to device forward and backward
// Gradient is to convert back to input device
auto TensorToDevice::forward(
    [[maybe_unused]] AutogradStorage &storage,
    [[maybe_unused]] bool is_grad_required,
    const Tensor &tensor,
    Device device
) -> Tensor {
    if (is_grad_required) {
        storage["device"] = tensor.device();
    }
    // bool is a special case
    if (tensor.dtype() == kBool) {
        return {tensor.to_vec<bool>(), tensor.shape(), device};
    }
    return DISPATCH_ALL_TYPES(tensor.dtype(), "Tensor::to", [&]() -> Tensor {
        return {tensor.to_vec<scalar_t>(), tensor.shape(), device};
    });
}
auto TensorToDevice::backward([[maybe_unused]] const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &device = std::get<Device>(storage.at("device"));
    return {grad_output.to(device)};
}

// Element-wise identity forward and backward
// d/dx x = 1
auto TensorIdentity::forward(
    [[maybe_unused]] AutogradStorage &storage,
    [[maybe_unused]] bool is_grad_required,
    const Tensor &tensor
) -> Tensor {
    return get_backend(tensor.device())->identity(tensor);
}
auto TensorIdentity::backward([[maybe_unused]] const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    return {grad_output};
}

// Element-wise abs forward and backward
// d/dx abs(x) = sign(x)
auto TensorAbs::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
    }
    return get_backend(tensor.device())->abs(tensor);
}
auto TensorAbs::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    return {grad_output * sign(input)};
}

// Element-wise sign forward and backward
// d/dx sign(x) = 0
auto TensorSign::forward(
    [[maybe_unused]] AutogradStorage &storage,
    [[maybe_unused]] bool is_grad_required,
    const Tensor &tensor
) -> Tensor {
    return get_backend(tensor.device())->sign(tensor);
}
auto TensorSign::backward([[maybe_unused]] const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    return {zeros_like(grad_output)};
}

// Element-wise negate forward and backward
// d/dx -x = -1
auto TensorNegate::forward(
    [[maybe_unused]] AutogradStorage &storage,
    [[maybe_unused]] bool is_grad_required,
    const Tensor &tensor
) -> Tensor {
    return get_backend(tensor.device())->negate(tensor);
}
auto TensorNegate::backward([[maybe_unused]] const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    return {-grad_output};
}

// Element-wise log forward and backward
// d/dx log(x) = 1/x
auto TensorLog::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
    }
    return get_backend(tensor.device())->log(tensor);
}
auto TensorLog::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    return {grad_output / input};
}

// Element-wise log2 forward and backward
// d/dx log_2(x) = 1/(log(2) * x)
auto TensorLog2::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
    }
    return get_backend(tensor.device())->log2(tensor);
}
auto TensorLog2::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    return {grad_output / (std::numbers::ln2 * input)};
}

// Element-wise log10 forward and backward
// d/dx log_10(x) = 1/(log(10) * x)
auto TensorLog10::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
    }
    return get_backend(tensor.device())->log10(tensor);
}
auto TensorLog10::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    return {grad_output / (std::numbers::ln10 * input)};
}

// Element-wise log1p forward and backward
// d/dx log(1 + x) = 1/(1+x)
auto TensorLog1p::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
    }
    return get_backend(tensor.device())->log1p(tensor);
}
auto TensorLog1p::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    return {grad_output / (1.0 + input)};
}

// Element-wise exp forward and backward
// d/dx exp(x) = exp(x)
auto TensorExp::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    auto result = get_backend(tensor.device())->exp(tensor);
    if (is_grad_required) {
        storage["result"] = make_versioned_tensor(result);
    }
    return result;
}
auto TensorExp::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[result, version] = std::get<VersionedTensor>(storage.at("result"));
    CHECK_VERSION(result, version);
    return {grad_output * result};
}

// Element-wise exp2 forward and backward
// d/dx 2^x = 2^x * log(2)
auto TensorExp2::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    auto result = get_backend(tensor.device())->exp2(tensor);
    if (is_grad_required) {
        storage["result"] = make_versioned_tensor(result);
    }
    return result;
}
auto TensorExp2::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[result, version] = std::get<VersionedTensor>(storage.at("result"));
    CHECK_VERSION(result, version);
    return {grad_output * (std::numbers::ln2 * result)};
}

// Element-wise exp1m forward and backward
// d/dx exp(x) - 1 = exp(x)
auto TensorExpm1::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
    }
    return get_backend(tensor.device())->expm1(tensor);
}
auto TensorExpm1::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    return {grad_output * exp(input)};
}

// Element-wise sqrt forward and backward
// d/dx sqrt(x) = 1/(2 * sqrt(x))
auto TensorSqrt::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    auto result = get_backend(tensor.device())->sqrt(tensor);
    if (is_grad_required) {
        storage["result"] = make_versioned_tensor(result);
    }
    return result;
}
auto TensorSqrt::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[result, version] = std::get<VersionedTensor>(storage.at("result"));
    CHECK_VERSION(result, version);
    return {grad_output / (2 * result)};
}

// Element-wise sin forward and backward
// d/dx sin(x) = cos(x)
auto TensorSin::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
    }
    return get_backend(tensor.device())->sin(tensor);
}
auto TensorSin::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    return {grad_output * cos(input)};
}

// Element-wise cos forward and backward
// d/dx cos(x) = -sin(x)
auto TensorCos::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
    }
    return get_backend(tensor.device())->cos(tensor);
}
auto TensorCos::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    return {grad_output * (-sin(input))};
}

// Element-wise tan forward and backward
// d/dx tan(x) = sec^2(x) = 1 / cos^2(x)
auto TensorTan::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
    }
    return get_backend(tensor.device())->tan(tensor);
}
auto TensorTan::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    return {grad_output / pow(cos(input), 2)};
}

// Element-wise asin forward and backward
// d/dx asin(x) = 1 / sqrt(1 - x^2)
auto TensorASin::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
    }
    return get_backend(tensor.device())->asin(tensor);
}
auto TensorASin::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    return {grad_output / sqrt(1 - pow(input, 2))};
}

// Element-wise acos forward and backward
// d/dx acos(x) = - 1 / sqrt(1 - x^2)
auto TensorACos::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
    }
    return get_backend(tensor.device())->acos(tensor);
}
auto TensorACos::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    return {-grad_output / sqrt(1 - pow(input, 2))};
}

// Element-wise atan forward and backward
// d/dx atan(x) = 1 / (x^2 + 1)
auto TensorATan::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
    }
    return get_backend(tensor.device())->atan(tensor);
}
auto TensorATan::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    return {grad_output / (pow(input, 2) + 1)};
}

// Element-wise sinh forward and backward
// d/dx sinh(x) = cosh(x)
auto TensorSinh::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
    }
    return get_backend(tensor.device())->sinh(tensor);
}
auto TensorSinh::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    return {grad_output * cosh(input)};
}

// Element-wise cosh forward and backward
// d/dx cosh(x) = sinh(x)
auto TensorCosh::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
    }
    return get_backend(tensor.device())->cosh(tensor);
}
auto TensorCosh::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    return {grad_output * sinh(input)};
}

// Element-wise tanh forward and backward
// d/dx tanh(x) = sech^2(x) = 1 / cosh^2(x)
auto TensorTanh::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
    }
    return get_backend(tensor.device())->tanh(tensor);
}
auto TensorTanh::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    return {grad_output / pow(cosh(input), 2)};
}

// Element-wise asinh forward and backward
// d/dx asinh(x) = 1 / sqrt(x^2 + 1)
auto TensorASinh::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
    }
    return get_backend(tensor.device())->asinh(tensor);
}
auto TensorASinh::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    return {grad_output / sqrt(pow(input, 2) + 1)};
}

// Element-wise acosh forward and backward
// d/dx acosh(x) = 1 / (sqrt(x-1) * sqrt(x+1))
// Since acosh(x) is defined on x>=1, negative inputs have a NaN grad
// Use grad defined on positive end:
// d/dx acosh(x) = 1 / sqrt(x^2-1)
auto TensorACosh::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
    }
    return get_backend(tensor.device())->acosh(tensor);
}
auto TensorACosh::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    return {grad_output / sqrt(pow(input, 2) - 1)};
}

// Element-wise atanh forward and backward
// d/dx atanh(x) = 1 / (1 - x^2)
auto TensorATanh::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
    }
    return get_backend(tensor.device())->atanh(tensor);
}
auto TensorATanh::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    return {grad_output / (1 - pow(input, 2))};
}

// Element-wise erf forward and backward
// d/dx erf(x) = 2 / sqrt(pi) * exp(-x^2)
// https://en.wikipedia.org/wiki/Error_function
auto TensorErf::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
    }
    return get_backend(tensor.device())->erf(tensor);
}
auto TensorErf::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    return {grad_output * 2 * std::numbers::inv_sqrtpi * exp(-pow(input, 2))};
}

// Element-wise erf forward and backward
// d/dx erfc(x) = -2 / sqrt(pi) * exp(-x^2)
// https://en.wikipedia.org/wiki/Error_function
auto TensorErfc::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
    }
    return get_backend(tensor.device())->erfc(tensor);
}
auto TensorErfc::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    return {-grad_output * 2 * std::numbers::inv_sqrtpi * exp(-pow(input, 2))};
}

// Element-wise gamma forward and backward
// d/dx gamma(x) = gamma(x) * digamma(x)
// https://en.wikipedia.org/wiki/Gamma_function
auto TensorTGamma::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
    }
    return get_backend(tensor.device())->tgamma(tensor);
}
auto TensorTGamma::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    return {grad_output * tgamma(input) * digamma(input)};
}

// Element-wise log-gamma forward and backward
// d/dx lgamma(x) = digamma(x)
// https://en.wikipedia.org/wiki/Gamma_function
auto TensorLGamma::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
    }
    return get_backend(tensor.device())->lgamma(tensor);
}
auto TensorLGamma::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    return {grad_output * digamma(input)};
}

// Element-wise sigmoid forward and backward
// d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
auto TensorSigmoid::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    auto result = get_backend(tensor.device())->sigmoid(tensor);
    if (is_grad_required) {
        storage["result"] = make_versioned_tensor(result);
    }
    return result;
}
auto TensorSigmoid::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[result, version] = std::get<VersionedTensor>(storage.at("result"));
    CHECK_VERSION(result, version);
    return {grad_output * result * (1 - result)};
}

// Element-wise log-sigmoid forward and backward
// d/dx log(sigmoid(x)) = 1 / (exp(x) + 1)
auto TensorLogSigmoid::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
    }
    return get_backend(tensor.device())->log_sigmoid(tensor);
}
auto TensorLogSigmoid::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    return {grad_output / (exp(input) + 1)};
}

// Element-wise hard-sigmoid forward and backward
//                       0 if x <= -3 or x >= 3
// d/dx hardsigmoid(x) = 1/6 else
auto TensorHardSigmoid::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
    }
    return get_backend(tensor.device())->hardsigmoid(tensor);
}
auto TensorHardSigmoid::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    const auto mask = (input > -3) && (input < 3);
    return {grad_output * where(mask, Scalar(1.0 / 6, input.dtype()), Scalar(0, input.dtype()))};
}

// Element-wise softplus forward and backward
//                                      sigmoid(beta * x) if x < threshold
// d/dx softplus(x, beta, threshold) =  1 else
auto TensorSoftplus::forward(
    AutogradStorage &storage,
    bool is_grad_required,
    const Tensor &tensor,
    double beta,
    double threshold
) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
        storage["beta"] = beta;
        storage["threshold"] = threshold;
    }
    return get_backend(tensor.device())->softplus(tensor, beta, threshold);
}
auto TensorSoftplus::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    const auto &beta = std::get<double>(storage.at("beta"));
    const auto &threshold = std::get<double>(storage.at("threshold"));
    const auto mask = (input < threshold);
    return {grad_output * where(mask, sigmoid(beta * input), ones_like(input))};
}

// Element-wise relu forward and backward
//                0 if x < 0
// d/dx relu(x) = 1 else
auto TensorRelu::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
    }
    return get_backend(tensor.device())->relu(tensor);
}
auto TensorRelu::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    const auto mask = (input < 0);
    return {grad_output * where(mask, Scalar(0, input.dtype()), Scalar(1, input.dtype()))};
}

// Element-wise relu6 forward and backward
//                 0 if x < 0 or x > 6
// d/dx relu6(x) = 1 else
auto TensorRelu6::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
    }
    return get_backend(tensor.device())->relu6(tensor);
}
auto TensorRelu6::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    const auto mask = (input < 0) || (input > 6);
    return {grad_output * where(mask, Scalar(0, input.dtype()), Scalar(1, input.dtype()))};
}

// Element-wise leaky_relu forward and backward
//                      negative_slope if x < 0
// d/dx leaky_relu(x) = 1 else
auto TensorLeakyRelu::forward(
    AutogradStorage &storage,
    bool is_grad_required,
    const Tensor &tensor,
    double negative_slope
) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
        storage["negative_slope"] = negative_slope;
    }
    return get_backend(tensor.device())->leaky_relu(tensor, negative_slope);
}
auto TensorLeakyRelu::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    const auto &negative_slope = std::get<double>(storage.at("negative_slope"));
    const auto mask = (input < 0);
    return {grad_output * where(mask, Scalar(negative_slope, input.dtype()), Scalar(1, input.dtype()))};
}

// Element-wise elu forward and backward
//                      alpha * exp(x)
// d/dx elu(x, alpha) = 1 else
auto TensorElu::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor, double alpha) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
        storage["alpha"] = alpha;
    }
    return get_backend(tensor.device())->elu(tensor, alpha);
}
auto TensorElu::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    const auto &alpha = std::get<double>(storage.at("alpha"));
    const auto mask = (input < 0);
    return {grad_output * where(mask, alpha * exp(input), Scalar(1, input.dtype()))};
}

// Element-wise selu forward and backward
//                scale * alpha * exp(x)
// d/dx selu(x) = scale else
auto TensorSelu::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
    }
    return get_backend(tensor.device())->selu(tensor);
}
auto TensorSelu::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    constexpr double alpha = 1.67326324235437728;
    constexpr double scale = 1.05070098735548049;
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    const auto mask = (input < 0);
    return {grad_output * where(mask, scale * alpha * exp(input), Scalar(scale, input.dtype()))};
}

// Element-wise silu forward and backward
// d/dx silu(x) = sigmoid(x) + x * sigmoid(x) * (1-sigmoid(x))
auto TensorSilu::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
    }
    return get_backend(tensor.device())->silu(tensor);
}
auto TensorSilu::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    const auto s = sigmoid(input);
    return {grad_output * (s + input * s * (1 - s))};
}

// Element-wise hardtanh forward and backward
//                              0 if x < min
// d/dx hardtanh(x, min, max) = 1 if min <= x <= max
//                              0 if x > max
auto TensorHardtanh::forward(
    AutogradStorage &storage,
    bool is_grad_required,
    const Tensor &tensor,
    double min,
    double max
) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
        storage["min"] = min;
        storage["max"] = max;
    }
    return get_backend(tensor.device())->hardtanh(tensor, min, max);
}
auto TensorHardtanh::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    const auto &min = std::get<double>(storage.at("min"));
    const auto &max = std::get<double>(storage.at("max"));
    const auto mask = (input < min) || (input > max);
    return {grad_output * where(mask, Scalar(0, input.dtype()), Scalar(1, input.dtype()))};
}

// Element-wise softsign forward and backward
// d/dx softsign(x) = 1 / (1 + |x|)^2
auto TensorSoftsign::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
    }
    return get_backend(tensor.device())->softsign(tensor);
}
auto TensorSoftsign::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    return {grad_output / pow(1 + abs(input), 2)};
}

// Element-wise softmax forward and backward
// d/dx softmax(x, dim) = each non-selected dim needs to compute Jacobian (NxN d_output / d_xi)
//                        then multiply by grad_output [g_i](1xN) * [J]{NxN}
//                        If you fully expand this product out, you get each element of input grad is
//                        [g_i * o_i - \sum_j g_j o_i o_j]_i
auto TensorSoftmax::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor, int dim) -> Tensor {
    auto result = get_backend(tensor.device())->softmax(tensor, dim);
    if (is_grad_required) {
        storage["result"] = make_versioned_tensor(result);
        storage["dim"] = dim;
    }
    return result;
}
auto TensorSoftmax::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[result, version] = std::get<VersionedTensor>(storage.at("result"));
    CHECK_VERSION(result, version);
    const auto &dim = std::get<int>(storage.at("dim"));
    Tensor _grad_output = grad_output * result;
    Tensor sum_grad = sum(_grad_output, dim, true);
    return {_grad_output - result * sum_grad.expand(result.shape())};
}

// Element-wise log-softmax forward and backward
// https://math.stackexchange.com/questions/4258008/derivative-of-the-log-softmax-function
auto TensorLogSoftmax::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor, int dim)
    -> Tensor {
    auto result = get_backend(tensor.device())->log_softmax(tensor, dim);
    if (is_grad_required) {
        storage["result"] = make_versioned_tensor(result);
        storage["dim"] = dim;
    }
    return result;
}
auto TensorLogSoftmax::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[result, version] = std::get<VersionedTensor>(storage.at("result"));
    CHECK_VERSION(result, version);
    const auto &dim = std::get<int>(storage.at("dim"));
    return {grad_output - exp(result) * grad_output.sum(dim, true).expand(result.shape())};
}

}    // namespace tinytensor::autograd
