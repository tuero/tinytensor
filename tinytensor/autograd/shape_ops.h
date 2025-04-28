// shape_ops.h
// Shape operations with autograd support

#ifndef TINYTENSOR_AUTOGRAD_SHAPE_OPS_H_
#define TINYTENSOR_AUTOGRAD_SHAPE_OPS_H_

#include <tt/autograd.h>
#include <tt/scalar.h>
#include <tt/shape.h>
#include <tt/tensor.h>

#include <string>
#include <vector>

namespace tinytensor::autograd {

struct TensorBroadcast : public TensorFunction<TensorBroadcast> {
    static constexpr std::string name = "Reshape";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor, const Shape &shape)
        -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorReshape : public TensorFunction<TensorReshape> {
    static constexpr std::string name = "Reshape";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor, const Shape &shape)
        -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorPermute : public TensorFunction<TensorPermute> {
    static constexpr std::string name = "Permute";
    static auto
        forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor, const std::vector<int> &dims)
            -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

}    // namespace tinytensor::autograd

#endif    // TINYTENSOR_AUTOGRAD_SHAPE_OPS_H_
