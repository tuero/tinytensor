// conv_ops.cpp
// Conv2d operations with autograd support

#include "autograd/conv_ops.h"

#include <tt/autograd.h>
#include <tt/exception.h>
#include <tt/grad_mode.h>
#include <tt/index.h>
#include <tt/scalar.h>
#include <tt/shape.h>
#include <tt/tensor.h>

#include "tensor/backend_register.h"

#include <format>
#include <optional>
#include <ranges>
#include <utility>

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

// Conv2d forward and backward
auto TensorConv2d::forward(
    AutogradStorage &storage,
    bool is_grad_required,
    const Tensor &tensor,
    const Tensor &weight,
    const std::optional<Tensor> &bias,
    int stride,
    int padding
) -> Tensor {
    if (is_grad_required) {
        storage["tensor"] = make_versioned_tensor(tensor);
        storage["weight"] = make_versioned_tensor(weight);
        if (bias) {
            storage["bias"] = make_versioned_tensor(*bias);
        }
        storage["stride"] = stride;
        storage["padding"] = padding;
    }
    return get_backend(tensor.device())->conv2d(tensor, weight, bias, stride, padding);
}
auto TensorConv2d::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[tensor, tensor_version] = std::get<VersionedTensor>(storage.at("tensor"));
    const auto &[weight, weight_version] = std::get<VersionedTensor>(storage.at("weight"));
    std::optional<Tensor> bias = std::nullopt;
    const auto &stride = std::get<int>(storage.at("stride"));
    const auto &padding = std::get<int>(storage.at("padding"));

    // Check all versions are valid
    CHECK_VERSION(tensor, tensor_version);
    CHECK_VERSION(weight, weight_version);
    if (storage.contains("bias")) {
        const auto &[_bias, bias_version] = std::get<VersionedTensor>(storage.at("bias"));
        CHECK_VERSION(_bias, bias_version);
        bias = _bias;
    }

    // Backward
    const auto &[grad_input, grad_weight, grad_bias] =
        get_backend(tensor.device())->conv2d_backward(grad_output, tensor, weight, bias, stride, padding);
    GradList grad_inputs = {std::move(grad_input), std::move(grad_weight)};

    if (bias) {
        grad_inputs.emplace_back(std::move(*grad_bias));
    }
    return grad_inputs;
}

// MaxPool2d forward and backward
auto TensorMaxPool2d::forward(
    AutogradStorage &storage,
    bool is_grad_required,
    const Tensor &tensor,
    int kernel_size,
    int stride,
    int padding
) -> Tensor {
    const auto result = get_backend(tensor.device())->max_pool2d(tensor, kernel_size, stride, padding);
    if (is_grad_required) {
        storage["tensor"] = make_versioned_tensor(tensor);
        storage["result"] = make_versioned_tensor(result);
        storage["kernel_size"] = kernel_size;
        storage["stride"] = stride;
        storage["padding"] = padding;
    }
    return result;
}
auto TensorMaxPool2d::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[tensor, tensor_version] = std::get<VersionedTensor>(storage.at("tensor"));
    const auto &[result, result_version] = std::get<VersionedTensor>(storage.at("result"));
    const auto &kernel_size = std::get<int>(storage.at("kernel_size"));
    const auto &stride = std::get<int>(storage.at("stride"));
    const auto &padding = std::get<int>(storage.at("padding"));

    // Check all versions are valid
    CHECK_VERSION(tensor, tensor_version);
    CHECK_VERSION(result, result_version);

    auto grad_input =
        get_backend(tensor.device())->max_pool2d_backward(grad_output, tensor, result, kernel_size, stride, padding);
    return {std::move(grad_input)};
}

// MinPool2d forward and backward
auto TensorMinPool2d::forward(
    AutogradStorage &storage,
    bool is_grad_required,
    const Tensor &tensor,
    int kernel_size,
    int stride,
    int padding
) -> Tensor {
    const auto result = get_backend(tensor.device())->min_pool2d(tensor, kernel_size, stride, padding);
    if (is_grad_required) {
        storage["tensor"] = make_versioned_tensor(tensor);
        storage["result"] = make_versioned_tensor(result);
        storage["kernel_size"] = kernel_size;
        storage["stride"] = stride;
        storage["padding"] = padding;
    }
    return result;
}
auto TensorMinPool2d::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[tensor, tensor_version] = std::get<VersionedTensor>(storage.at("tensor"));
    const auto &[result, result_version] = std::get<VersionedTensor>(storage.at("result"));
    const auto &kernel_size = std::get<int>(storage.at("kernel_size"));
    const auto &stride = std::get<int>(storage.at("stride"));
    const auto &padding = std::get<int>(storage.at("padding"));

    // Check all versions are valid
    CHECK_VERSION(tensor, tensor_version);
    CHECK_VERSION(result, result_version);

    auto grad_input =
        get_backend(tensor.device())->min_pool2d_backward(grad_output, tensor, result, kernel_size, stride, padding);
    return {std::move(grad_input)};
}

// AvgPool2d forward and backward
auto TensorAvgPool2d::forward(
    AutogradStorage &storage,
    bool is_grad_required,
    const Tensor &tensor,
    int kernel_size,
    int stride,
    int padding
) -> Tensor {
    const auto result = get_backend(tensor.device())->avg_pool2d(tensor, kernel_size, stride, padding);
    if (is_grad_required) {
        storage["tensor"] = make_versioned_tensor(tensor);
        storage["result"] = make_versioned_tensor(result);
        storage["kernel_size"] = kernel_size;
        storage["stride"] = stride;
        storage["padding"] = padding;
    }
    return result;
}
auto TensorAvgPool2d::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[tensor, tensor_version] = std::get<VersionedTensor>(storage.at("tensor"));
    const auto &[result, result_version] = std::get<VersionedTensor>(storage.at("result"));
    const auto &kernel_size = std::get<int>(storage.at("kernel_size"));
    const auto &stride = std::get<int>(storage.at("stride"));
    const auto &padding = std::get<int>(storage.at("padding"));

    // Check all versions are valid
    CHECK_VERSION(tensor, tensor_version);
    CHECK_VERSION(result, result_version);

    auto grad_input =
        get_backend(tensor.device())->avg_pool2d_backward(grad_output, tensor, result, kernel_size, stride, padding);
    return {std::move(grad_input)};
}

}    // namespace tinytensor::autograd
