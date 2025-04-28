// misc_ops.h
// Misc operations with autograd support

#ifndef TINYTENSOR_AUTOGRAD_MISC_OPS_H_
#define TINYTENSOR_AUTOGRAD_MISC_OPS_H_

#include <tt/autograd.h>
#include <tt/scalar.h>
#include <tt/shape.h>
#include <tt/tensor.h>

#include <string>

namespace tinytensor::autograd {

struct TensorWhere : public TensorFunction<TensorWhere> {
    static constexpr std::string name = "Where";
    static auto forward(
        AutogradStorage &storage,
        bool is_grad_required,
        const Tensor &cond,
        const Tensor &lhs,
        const Tensor &rhs
    ) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

}    // namespace tinytensor::autograd

#endif    // TINYTENSOR_AUTOGRAD_MISC_OPS_H_
