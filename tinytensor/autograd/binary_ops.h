// binary_ops.h
// Binary operations with autograd support

#ifndef TINYTENSOR_AUTOGRAD_BINARY_OPS_H_
#define TINYTENSOR_AUTOGRAD_BINARY_OPS_H_

#include <tt/autograd.h>
#include <tt/tensor.h>

#include <string>

namespace tinytensor::autograd {

struct TensorAdd : public TensorFunction<TensorAdd> {
    static constexpr std::string name = "Add";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &lhs, const Tensor &rhs)
        -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorSub : public TensorFunction<TensorSub> {
    static constexpr std::string name = "Sub";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &lhs, const Tensor &rhs)
        -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorMul : public TensorFunction<TensorMul> {
    static constexpr std::string name = "Mul";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &lhs, const Tensor &rhs)
        -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorDiv : public TensorFunction<TensorDiv> {
    static constexpr std::string name = "Div";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &lhs, const Tensor &rhs)
        -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorMaximum : public TensorFunction<TensorMaximum> {
    static constexpr std::string name = "Max";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &lhs, const Tensor &rhs)
        -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorMinimum : public TensorFunction<TensorMinimum> {
    static constexpr std::string name = "Min";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &lhs, const Tensor &rhs)
        -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorPow : public TensorFunction<TensorPow> {
    static constexpr std::string name = "Pow";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &lhs, const Tensor &rhs)
        -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorBatchedMatmul : public TensorFunction<TensorBatchedMatmul> {
    static constexpr std::string name = "Matmul";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &lhs, const Tensor &rhs)
        -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

}    // namespace tinytensor::autograd

#endif    // TINYTENSOR_AUTOGRAD_BINARY_OPS_H_
