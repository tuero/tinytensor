// shape_ops.cpp
// Shape operations with autograd support

#include "autograd/shape_ops.h"

#include <tt/autograd.h>
#include <tt/grad_mode.h>
#include <tt/index.h>
#include <tt/scalar.h>
#include <tt/shape.h>
#include <tt/tensor.h>

#include <cstddef>
#include <memory>
#include <ranges>
#include <vector>

namespace tinytensor::autograd {

// Broadcast forward and backward
// Gradient is summed along dims which need to be reduced
// - New dims which were prepended
// - dims which do not match i.e. from expansions
auto TensorBroadcast::forward(
    [[maybe_unused]] AutogradStorage &storage,
    [[maybe_unused]] bool is_grad_required,
    const Tensor &tensor,
    const Shape &shape
) -> Tensor {
    if (is_grad_required) {
        storage["input_shape"] = tensor.shape();
    }
    const auto input_shape = tensor.shape();
    const auto input_stride = tensor.stride();
    // const auto result_shape = shape;

    // Set resulting shape
    // If number of dimensions not equal, prepend 1s until same length
    // For each resulting dim, set max of the sizes
    Shape shape_for_stride(std::vector<int>(static_cast<std::size_t>(shape.ndim()), 1));
    for (int i = input_shape.ndim() - 1; i >= 0; --i) {
        shape_for_stride[i + shape.ndim() - input_shape.ndim()] = input_shape[i];
    }

    // Set resulting stride
    // Dims which get expanded during broadcast have stride set to 0,
    Shape _result_stride(std::vector<int>(static_cast<std::size_t>(shape.ndim()), 0));
    const auto size_diff = shape.size() - tensor.dim();
    for (const auto i : std::views::iota(0, tensor.dim()) | std::views::reverse) {
        if (input_shape[i] == shape[i + size_diff]) {
            _result_stride[i + size_diff] = input_stride[i];
        }
    }

    // Result is a shallow copy, just with a new view of the same underlying data
    Tensor result = tensor;
    get_ctx(result) = std::make_shared<SharedGrad>();
    Shape &result_shape = get_shape(result);
    Shape &result_stride = get_stride(result);
    result_shape = shape;
    result_stride = _result_stride;
    return result;
}
auto TensorBroadcast::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &shape_input = std::get<Shape>(storage.at("input_shape"));
    const auto &shape_grad = grad_output.shape();

    // Find reduction dims
    std::vector<int> reduce_dims;
    int leading_dims = shape_grad.size() - shape_input.size();
    // New dims which were preprended
    for (int i = 0; i < leading_dims; ++i) {
        reduce_dims.push_back(i);
    }
    // Dims which do not match in size i.e. from expansions
    for (int i = leading_dims; i < shape_grad.size(); ++i) {
        if (shape_input[i - leading_dims] == 1 && shape_grad[i] != 1) {
            reduce_dims.push_back(i);
        }
    }
    // Reduce dims by sum using chain rule
    Tensor grad_input = grad_output;
    for (const auto dim : std::views::reverse(reduce_dims)) {
        grad_input = grad_input.sum(dim);
    }
    return {grad_input.reshape(shape_input)};
}

// Reshape forward and backward
// Gradient is reshape back to original shape
auto TensorReshape::forward(
    [[maybe_unused]] AutogradStorage &storage,
    [[maybe_unused]] bool is_grad_required,
    const Tensor &tensor,
    const Shape &shape
) -> Tensor {
    if (is_grad_required) {
        storage["input_shape"] = tensor.shape();
    }
    // view returned if contiguous, then set shape/stride
    Tensor result = tensor;
    get_ctx(result) = std::make_shared<SharedGrad>();
    Shape &result_shape = get_shape(result);
    Shape &result_stride = get_stride(result);

    result_shape = shape;
    result_stride = shape.to_stride();
    return result;
}
auto TensorReshape::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &shape = std::get<Shape>(storage.at("input_shape"));
    return {grad_output.reshape(shape)};
}

// Permute forward and backward
// Gradient is inverse permute
auto TensorPermute::forward(
    [[maybe_unused]] AutogradStorage &storage,
    [[maybe_unused]] bool is_grad_required,
    const Tensor &tensor,
    const std::vector<int> &dims
) -> Tensor {
    if (is_grad_required) {
        storage["dims"] = dims;
    }
    // Result is a shallow copy, just with a new view of the same underlying data
    Tensor result = tensor;
    get_ctx(result) = std::make_shared<SharedGrad>();
    Shape &result_shape = get_shape(result);
    Shape &result_stride = get_stride(result);
    for (int i = 0; i < tensor.dim(); ++i) {
        result_shape[i] = tensor.shape()[dims[static_cast<std::size_t>(i)]];
        result_stride[i] = tensor.stride()[dims[static_cast<std::size_t>(i)]];
    }
    return result;
}
auto TensorPermute::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &dims = std::get<std::vector<int>>(storage.at("dims"));

    // Compute inverse permutation
    auto inverse_dims = dims;
    for (int i = 0; i < static_cast<int>(dims.size()); ++i) {
        inverse_dims[static_cast<std::size_t>(dims[static_cast<std::size_t>(i)])] = i;
    }
    return {grad_output.permute(inverse_dims)};
}

}    // namespace tinytensor::autograd
