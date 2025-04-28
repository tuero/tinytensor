// conv_ops.h
// Conv2d operations with autograd support

#ifndef TINYTENSOR_AUTOGRAD_CONV_OPS_H_
#define TINYTENSOR_AUTOGRAD_CONV_OPS_H_

#include <tt/autograd.h>
#include <tt/scalar.h>
#include <tt/tensor.h>

#include <optional>
#include <string>

namespace tinytensor::autograd {

struct TensorConv2d : public TensorFunction<TensorConv2d> {
    static constexpr std::string name = "Conv2d";
    static auto forward(
        AutogradStorage &storage,
        bool is_grad_required,
        const Tensor &tensor,
        const Tensor &weight,
        const std::optional<Tensor> &bias,
        int stride,
        int padding
    ) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorMaxPool2d : public TensorFunction<TensorMaxPool2d> {
    static constexpr std::string name = "MaxPool2d";
    static auto forward(
        AutogradStorage &storage,
        bool is_grad_required,
        const Tensor &tensor,
        int kernel_size,
        int stride,
        int padding
    ) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorMinPool2d : public TensorFunction<TensorMinPool2d> {
    static constexpr std::string name = "MinPool2d";
    static auto forward(
        AutogradStorage &storage,
        bool is_grad_required,
        const Tensor &tensor,
        int kernel_size,
        int stride,
        int padding
    ) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorAvgPool2d : public TensorFunction<TensorAvgPool2d> {
    static constexpr std::string name = "AvgPool2d";
    static auto forward(
        AutogradStorage &storage,
        bool is_grad_required,
        const Tensor &tensor,
        int kernel_size,
        int stride,
        int padding
    ) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

}    // namespace tinytensor::autograd

#endif    // TINYTENSOR_AUTOGRAD_CONV_OPS_H_
