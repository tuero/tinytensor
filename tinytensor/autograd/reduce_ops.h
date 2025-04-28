// reduce_ops.h
// Reduction operations with autograd support

#ifndef TINYTENSOR_AUTOGRAD_REDUCE_OPS_H_
#define TINYTENSOR_AUTOGRAD_REDUCE_OPS_H_

#include <tt/autograd.h>
#include <tt/scalar.h>
#include <tt/shape.h>
#include <tt/tensor.h>

#include <string>

namespace tinytensor::autograd {

struct TensorMinAll : public TensorFunction<TensorMinAll> {
    static constexpr std::string name = "MinAll";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorMinDim : public TensorFunction<TensorMinDim> {
    static constexpr std::string name = "MinDim";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor, int dim, bool keep_dim)
        -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorMaxAll : public TensorFunction<TensorMaxAll> {
    static constexpr std::string name = "MaxAll";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorMaxDim : public TensorFunction<TensorMaxDim> {
    static constexpr std::string name = "MaxDim";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor, int dim, bool keep_dim)
        -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorSumAll : public TensorFunction<TensorSumAll> {
    static constexpr std::string name = "SumAll";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorSumDim : public TensorFunction<TensorSumDim> {
    static constexpr std::string name = "SumDim";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor, int dim, bool keep_dim)
        -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorMeanAll : public TensorFunction<TensorMeanAll> {
    static constexpr std::string name = "MeanAll";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorMeanDim : public TensorFunction<TensorMeanDim> {
    static constexpr std::string name = "MeanAll";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor, int dim, bool keep_dim)
        -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

}    // namespace tinytensor::autograd

#endif    // TINYTENSOR_AUTOGRAD_REDUCE_OPS_H_
