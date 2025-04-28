// binary_ops.cpp
// Binary operations with autograd support

#include "autograd/binary_ops.h"

#include <tt/autograd.h>
#include <tt/exception.h>
#include <tt/grad_mode.h>
#include <tt/tensor.h>

#include "tensor/backend_register.h"

#include <format>

namespace tinytensor::autograd {

// NOLINTNEXTLINE(*-macro-usage)
#define CHECK_VERSION(tensor, version)                                                                           \
    if (tensor.version_count() != version) {                                                                     \
        TT_EXCEPTION(                                                                                            \
            std::format(                                                                                         \
                "Inplace operation on tensor required for autograd detected. Tensor with version {:d} saved in " \
                "forward "                                                                                       \
                "pass, but has version {:d} in backward pass",                                                   \
                version,                                                                                         \
                tensor.version_count()                                                                           \
            )                                                                                                    \
        );                                                                                                       \
    }

// Element-wise addition forward and backward
// d(x+y)/dx = 1
// d(x+y)/dy = 1
auto TensorAdd::forward(
    [[maybe_unused]] AutogradStorage &storage,
    [[maybe_unused]] bool is_grad_required,
    const Tensor &lhs,
    const Tensor &rhs
) -> Tensor {
    return get_backend(lhs.device())->add(lhs, rhs);
}
auto TensorAdd::backward([[maybe_unused]] const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    return {grad_output, grad_output};
}

// Element-wise subtraction forward and backward
// d(x-y)/dx = 1
// d(x-y)/dy = -1
auto TensorSub::forward(
    [[maybe_unused]] AutogradStorage &storage,
    [[maybe_unused]] bool is_grad_required,
    const Tensor &lhs,
    const Tensor &rhs
) -> Tensor {
    return get_backend(lhs.device())->sub(lhs, rhs);
}
auto TensorSub::backward([[maybe_unused]] const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    return {grad_output, -grad_output};
}

// Element-wise multiplication forward and backward
// d(x*y)/dx = y
// d(x*y)/dy = x
auto TensorMul::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &lhs, const Tensor &rhs)
    -> Tensor {
    if (is_grad_required) {
        storage["lhs"] = make_versioned_tensor(lhs);
        storage["rhs"] = make_versioned_tensor(rhs);
    }
    return get_backend(lhs.device())->mul(lhs, rhs);
}
auto TensorMul::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[lhs, version_lhs] = std::get<VersionedTensor>(storage.at("lhs"));
    CHECK_VERSION(lhs, version_lhs);
    const auto &[rhs, version_rhs] = std::get<VersionedTensor>(storage.at("rhs"));
    CHECK_VERSION(rhs, version_rhs);
    return {grad_output * rhs, grad_output * lhs};
}

// Element-wise division forward and backward
// d(u/v)/du = 1/v
// d(u/v)/dv = -u/v^2
auto TensorDiv::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &lhs, const Tensor &rhs)
    -> Tensor {
    if (is_grad_required) {
        storage["lhs"] = make_versioned_tensor(lhs);
        storage["rhs"] = make_versioned_tensor(rhs);
    }
    return get_backend(lhs.device())->div(lhs, rhs);
}
auto TensorDiv::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    // (1/v, -u/v^2)
    const auto &[u, version_lhs] = std::get<VersionedTensor>(storage.at("lhs"));
    CHECK_VERSION(u, version_lhs);
    const auto &[v, version_rhs] = std::get<VersionedTensor>(storage.at("rhs"));
    CHECK_VERSION(v, version_rhs);
    const Tensor v_recip = 1.0 / v;
    const Tensor v_recip_grad = grad_output * v_recip;
    return {v_recip_grad, -v_recip_grad * v_recip * u};
}

// Element-wise max forward and backward
// d/dx max(x,y) = 1 if x else y
// d/dy max(x,y) = 1 if x else y
auto TensorMaximum::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &lhs, const Tensor &rhs)
    -> Tensor {
    if (is_grad_required) {
        storage["lhs"] = make_versioned_tensor(lhs);
        storage["rhs"] = make_versioned_tensor(rhs);
    }
    return get_backend(lhs.device())->maximum(lhs, rhs);
}
auto TensorMaximum::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[lhs, version_lhs] = std::get<VersionedTensor>(storage.at("lhs"));
    CHECK_VERSION(lhs, version_lhs);
    const auto &[rhs, version_rhs] = std::get<VersionedTensor>(storage.at("rhs"));
    CHECK_VERSION(rhs, version_rhs);
    const Tensor mask = lhs > rhs;
    return {grad_output * mask, grad_output * !mask};
}

// Element-wise min forward and backward
// d/dx min(x,y) = 1 if x else y
// d/dy min(x,y) = 1 if x else y
auto TensorMinimum::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &lhs, const Tensor &rhs)
    -> Tensor {
    if (is_grad_required) {
        storage["lhs"] = make_versioned_tensor(lhs);
        storage["rhs"] = make_versioned_tensor(rhs);
    }
    return get_backend(lhs.device())->minimum(lhs, rhs);
}
auto TensorMinimum::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[lhs, version_lhs] = std::get<VersionedTensor>(storage.at("lhs"));
    CHECK_VERSION(lhs, version_lhs);
    const auto &[rhs, version_rhs] = std::get<VersionedTensor>(storage.at("rhs"));
    CHECK_VERSION(rhs, version_rhs);
    const Tensor mask = lhs < rhs;
    return {grad_output * mask, grad_output * !mask};
}

// Element-wise power forward and backward
// d/dx x^y = y * x^(y-1)
// d/dy x^y = x^y * log(y)
auto TensorPow::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &lhs, const Tensor &rhs)
    -> Tensor {
    if (is_grad_required) {
        storage["lhs"] = make_versioned_tensor(lhs);
        storage["rhs"] = make_versioned_tensor(rhs);
    }
    return get_backend(lhs.device())->pow(lhs, rhs);
}
auto TensorPow::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[lhs, version_lhs] = std::get<VersionedTensor>(storage.at("lhs"));
    CHECK_VERSION(lhs, version_lhs);
    const auto &[rhs, version_rhs] = std::get<VersionedTensor>(storage.at("rhs"));
    CHECK_VERSION(rhs, version_rhs);
    return {grad_output * rhs * pow(lhs, rhs - 1), grad_output * pow(lhs, rhs) * log(lhs)};
}

// Element-wise batched matmul forward and backward
// X=(b, n, k) @ Y=(b, k, m) = (b, n, m)=C
// dC/dX = C=(b, n, m) @ Y^T=(b, m, k)
// dC/dY = X^T=(b, k, n) @ C=(b, n, m)
auto TensorBatchedMatmul::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &lhs, const Tensor &rhs)
    -> Tensor {
    if (is_grad_required) {
        storage["lhs"] = make_versioned_tensor(lhs);
        storage["rhs"] = make_versioned_tensor(rhs);
    }
    return get_backend(lhs.device())->batched_matmul(lhs, rhs);
}
auto TensorBatchedMatmul::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[lhs, version_lhs] = std::get<VersionedTensor>(storage.at("lhs"));
    CHECK_VERSION(lhs, version_lhs);
    const auto &[rhs, version_rhs] = std::get<VersionedTensor>(storage.at("rhs"));
    CHECK_VERSION(rhs, version_rhs);
    return {matmul(grad_output, rhs.permute({0, 2, 1})), matmul(lhs.permute({0, 2, 1}), grad_output)};
}

}    // namespace tinytensor::autograd
