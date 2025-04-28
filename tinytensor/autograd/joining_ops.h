// joining_ops.h
// Joining operations with autograd support

#ifndef TINYTENSOR_AUTOGRAD_JOIN_OPS_H_
#define TINYTENSOR_AUTOGRAD_JOIN_OPS_H_

#include <tt/autograd.h>
#include <tt/scalar.h>
#include <tt/tensor.h>

#include <string>

namespace tinytensor::autograd {

struct TensorCat : public TensorFunction<TensorCat> {
    static constexpr std::string name = "Cat";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const TensorList &tensors, int dim) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

}    // namespace tinytensor::autograd

#endif    // TINYTENSOR_AUTOGRAD_JOIN_OPS_H_
