// clamp_ops.h
// Clamp operations with autograd support

#ifndef TINYTENSOR_AUTOGRAD_CLAMP_OPS_H_
#define TINYTENSOR_AUTOGRAD_CLAMP_OPS_H_

#include <tt/autograd.h>
#include <tt/scalar.h>
#include <tt/shape.h>
#include <tt/tensor.h>

#include <string>

namespace tinytensor::autograd {

struct TensorClamp : public TensorFunction<TensorClamp> {
    static constexpr std::string name = "Clamp";
    static auto forward(
        AutogradStorage &storage,
        bool is_grad_required,
        const Tensor &tensor,
        const Tensor &min,
        const Tensor &max
    ) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

}    // namespace tinytensor::autograd

#endif    // TINYTENSOR_AUTOGRAD_CLAMP_OPS_H_
