// backward.cpp
// Backward pass functionality

#include "autograd/backward.h"

#include <tt/autograd.h>
#include <tt/exception.h>
#include <tt/grad_mode.h>
#include <tt/scalar.h>
#include <tt/tensor.h>

#include "autograd/dag.h"

#include <cassert>
#include <format>
#include <optional>
#include <ranges>

namespace tinytensor::autograd {

void calc_grad_input(Tensor &tensor, bool retain_graph) {
    if (tensor.ctx_->grad_func) {
        if (!tensor.ctx_->grad) {
            TT_ERROR("Gradient was not propogated to this tensor");
        }
        GradList grad_inputs = tensor.ctx_->grad_func(tensor.ctx_->storage, *tensor.ctx_->grad);
        if (grad_inputs.size() > tensor.ctx_->parents.size()) {
            // Better messaging if we think graph not retained vs an incorrect implementation of forward/backward
            if (tensor.ctx_->parents.empty()) {
                TT_EXCEPTION(
                    std::format(
                        "Backward function {:s} returned {:d} gradients but has no saved parents.\n\tIf you are "
                        "wanting to "
                        "perform .backward() twice, try keeping the computation graph by using .backard(true)",
                        tensor.ctx_->grad_func_name,
                        grad_inputs.size(),
                        tensor.ctx_->parents.size()
                    )
                );
            } else {
                TT_ERROR(
                    std::format(
                        "Backward function {:s} returned {:d} gradients, but only only takes {:d} inputs",
                        tensor.ctx_->grad_func_name,
                        grad_inputs.size(),
                        tensor.ctx_->parents.size()
                    )
                );
            }
        }
        for (int i : std::views::iota(0, grad_inputs.size())) {
            if (grad_inputs[i]) {
                tensor.ctx_->parents[i].add_grad(grad_inputs[i].value());
            }
        }
        // Remove reference to outstanding tensors in computation graph if we do not need anymore
        if (!retain_graph) {
            tensor.ctx_->parents.clear();
            tensor.ctx_->storage.clear();
        }
    }
}

void backward(Tensor &tensor, const Tensor &grad, bool retain_graph) {
    assert(is_float_dtype(tensor.dtype()));
    assert(tensor.device() == grad.device());
    assert(tensor.shape() == grad.shape());
    assert(tensor.dtype() == grad.dtype());

    // Set initial grad to 1
    tensor.ctx_->grad = grad;

    // Build DAG
    TensorList dag = autograd::build_dag(tensor);

    // Dont recursively set grads
    // Walk backward along DAG and calc the grad
    const autograd::NoGradGuard guard;
    for (auto &t : std::ranges::reverse_view(dag)) {
        calc_grad_input(t, retain_graph);
        t.apply_grad_hook();
    }
}

}    // namespace tinytensor::autograd
