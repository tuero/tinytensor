// misc_ops.cpp
// Misc operations with autograd support

#include "autograd/misc_ops.h"

#include <tt/autograd.h>
#include <tt/exception.h>
#include <tt/grad_mode.h>
#include <tt/scalar.h>
#include <tt/shape.h>
#include <tt/tensor.h>

#include "tensor/backend_register.h"

#include <format>
#include <optional>

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

// Where forward and backward
// Element-wise gradient flows back to lhs of cond true, rhs if cond false
auto TensorWhere::forward(
    AutogradStorage &storage,
    bool is_grad_required,
    const Tensor &cond,
    const Tensor &lhs,
    const Tensor &rhs
) -> Tensor {
    if (is_grad_required) {
        storage["cond"] = make_versioned_tensor(cond);
        storage["lhs"] = make_versioned_tensor(lhs);
        storage["rhs"] = make_versioned_tensor(rhs);
    }
    return get_backend(cond.device())->where(cond, lhs, rhs);
}
auto TensorWhere::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[cond, cond_version] = std::get<VersionedTensor>(storage.at("cond"));
    const auto &[lhs, lhs_version] = std::get<VersionedTensor>(storage.at("lhs"));
    const auto &[rhs, rhs_version] = std::get<VersionedTensor>(storage.at("rhs"));

    CHECK_VERSION(cond, cond_version);
    CHECK_VERSION(lhs, lhs_version);
    CHECK_VERSION(rhs, rhs_version);

    return {std::nullopt, where(cond, grad_output, 0), where(!cond, grad_output, 0)};
}

}    // namespace tinytensor::autograd
