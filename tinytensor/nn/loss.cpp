// loss.h
// Loss layer functions and modules

#include <tt/device.h>
#include <tt/exception.h>
#include <tt/nn/init.h>
#include <tt/nn/loss.h>
#include <tt/scalar.h>
#include <tt/tensor.h>
#include <tt/util.h>

#include <cmath>
#include <cstddef>
#include <format>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

namespace tinytensor::nn {

namespace {
constexpr auto reduction_mode_to_str(ReductionMode mode) -> std::string {
    switch (mode) {
    case ReductionMode::none:
        return "none";
    case ReductionMode::mean:
        return "mean";
    case ReductionMode::sum:
        return "sum";
    case ReductionMode::batch_mean:
        return "batch_mean";
    }
    unreachable();
}

auto handle_reduction(const Tensor &tensor, ReductionMode mode) -> Tensor {
    switch (mode) {
    case ReductionMode::none:
        return tensor;
    case ReductionMode::mean:
        return tensor.mean();
    case ReductionMode::sum:
        return tensor.sum();
    case ReductionMode::batch_mean:
        return tensor.sum() / tensor.size(0);
    }
    unreachable();
}
}    // namespace

// -------------------------------------------------

auto l1_loss(const Tensor &input, const Tensor &target, ReductionMode mode) -> Tensor {
    Tensor loss = abs(input - target);
    return handle_reduction(loss, mode);
}

L1Loss::L1Loss(ReductionMode mode)
    : mode_(mode) {}

void L1Loss::pretty_print(std::ostream &os) const {
    os << std::format("L1Loss(mode={:s})", reduction_mode_to_str(mode_));
}

// -------------------------------------------------

auto mse_loss(const Tensor &input, const Tensor &target, ReductionMode mode) -> Tensor {
    Tensor loss = pow(input - target, 2);
    return handle_reduction(loss, mode);
}

MSELoss::MSELoss(ReductionMode mode)
    : mode_(mode) {}

void MSELoss::pretty_print(std::ostream &os) const {
    os << std::format("MSELoss(mode={:s})", reduction_mode_to_str(mode_));
}

// -------------------------------------------------

auto cross_entropy_loss(
    const Tensor &input,
    const Tensor &target_indices,
    const std::optional<Tensor> &weight,
    ReductionMode mode
) -> Tensor {
    int reduce_dim = input.dim() == 1 ? 0 : 1;
    int num_classes = input.size(reduce_dim);
    Tensor w = weight.value_or(ones({num_classes}, TensorOptions().dtype(input.dtype()).device(input.device())));
    if (w.numel() != num_classes) {
        TT_EXCEPTION(
            std::format(
                "Given weight of shape {:s}, expected to have same size as number of classes in put of shape {:s} with "
                "{:d} classes",
                w.shape(),
                input.shape(),
                num_classes
            )
        );
    }
    if (!is_integral_dtype(target_indices.dtype())) {
        TT_EXCEPTION(std::format("Expected indices to be of integral dtype, given {:s}", target_indices.dtype()));
    }
    // Expand weight to match input shape
    for (int i = 0; i < input.dim() - 2; ++i) {
        w = w.unsqueeze(-1);
    }
    w = w.expand(input.shape());

    // one_hot puts class dim at inner most dim, need to permute it to 2nd dim
    Tensor target_one_hot = one_hot(target_indices, num_classes);
    if (input.dim() > 2) {
        std::vector<int> permute_dims(static_cast<std::size_t>(input.dim()), 0);
        permute_dims[1] = input.dim() - 1;
        for (int i = 2; i < input.dim(); ++i) {
            permute_dims[static_cast<std::size_t>(i)] = i - 1;
        }
        target_one_hot = target_one_hot.permute(permute_dims);
    }
    Tensor loss = sum(-target_one_hot * w * log_softmax(input, reduce_dim), reduce_dim);

    // Handle reduction manually
    // mean() is a normalized mean using the weights
    // see https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss
    // https://github.com/pytorch/pytorch/issues/40560
    switch (mode) {
    case ReductionMode::none:
        return loss;
    case ReductionMode::mean: {
        Tensor normalizer = (w.expand(target_one_hot.shape()) * target_one_hot).sum();
        return loss.sum() / normalizer;
    }
    case ReductionMode::sum:
        return loss.sum();
    default:
        TT_EXCEPTION(std::format("Reduction mode {:s} is unsupported for this loss", reduction_mode_to_str(mode)));
    }
    unreachable();
}

CrossEntropyLoss::CrossEntropyLoss(const std::optional<Tensor> &weight, ReductionMode mode)
    : mode_(mode) {
    if (weight) {
        weight_ = std::make_shared<Tensor>(weight.value());
        register_param(weight_.value());
    }
}

void CrossEntropyLoss::pretty_print(std::ostream &os) const {
    os << std::format("CrossEntropyLoss(mode={:s})", reduction_mode_to_str(mode_));
}

// -------------------------------------------------

auto nll_loss(
    const Tensor &input,
    const Tensor &target_indices,
    const std::optional<Tensor> &weight,
    ReductionMode mode
) -> Tensor {
    int reduce_dim = input.dim() == 1 ? 0 : 1;
    int num_classes = input.size(reduce_dim);
    Tensor w = weight.value_or(ones({num_classes}, TensorOptions().dtype(input.dtype()).device(input.device())));
    if (w.numel() != num_classes) {
        TT_EXCEPTION(
            std::format(
                "Given weight of shape {:s}, expected to have same size as number of classes in put of shape {:s} with "
                "{:d} classes",
                w.shape(),
                input.shape(),
                num_classes
            )
        );
    }
    if (!is_integral_dtype(target_indices.dtype())) {
        TT_EXCEPTION(std::format("Expected indices to be of integral dtype, given {:s}", target_indices.dtype()));
    }
    // Expand weight to match input shape
    for (int i = 0; i < input.dim() - 2; ++i) {
        w = w.unsqueeze(-1);
    }
    w = w.expand(input.shape());

    // one_hot puts class dim at inner most dim, need to permute it to 2nd dim
    Tensor target_one_hot = one_hot(target_indices, num_classes);
    if (input.dim() > 2) {
        std::vector<int> permute_dims(static_cast<std::size_t>(input.dim()), 0);
        permute_dims[1] = input.dim() - 1;
        for (int i = 2; i < input.dim(); ++i) {
            permute_dims[static_cast<std::size_t>(i)] = i - 1;
        }
        target_one_hot = target_one_hot.permute(permute_dims);
    }
    Tensor loss = sum(-target_one_hot * w * input, reduce_dim);

    // Handle reduction manually
    // mean() is a normalized mean using the weights
    // see https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss
    // https://github.com/pytorch/pytorch/issues/40560
    switch (mode) {
    case ReductionMode::none:
        return loss;
    case ReductionMode::mean: {
        Tensor normalizer = (w.expand(target_one_hot.shape()) * target_one_hot).sum();
        return loss.sum() / normalizer;
    }
    case ReductionMode::sum:
        return loss.sum();
    default:
        TT_EXCEPTION(std::format("Reduction mode {:s} is unsupported for this loss", reduction_mode_to_str(mode)));
    }
    unreachable();
}

NLLLoss::NLLLoss(const std::optional<Tensor> &weight, ReductionMode mode)
    : mode_(mode) {
    if (weight) {
        weight_ = std::make_shared<Tensor>(weight.value());
        register_param(weight_.value());
    }
}

void NLLLoss::pretty_print(std::ostream &os) const {
    os << std::format("NLLLoss(mode={:s})", reduction_mode_to_str(mode_));
}

// -------------------------------------------------

auto kld_loss(const Tensor &input, const Tensor &target, ReductionMode mode, bool log_target) -> Tensor {
    Tensor loss = log_target ? (exp(target) * (target - input)) : (target * (log(target) - input));
    return handle_reduction(loss, mode);
}

KLDivLoss::KLDivLoss(ReductionMode mode, bool log_target)
    : mode_(mode), log_target_(log_target) {}

void KLDivLoss::pretty_print(std::ostream &os) const {
    os << std::format("KLDivLoss(mode={:s})", reduction_mode_to_str(mode_));
}

// -------------------------------------------------

auto bce_loss(const Tensor &input, const Tensor &target, ReductionMode mode) -> Tensor {
    // clamp log(x) to -100 <-> clamp x to exp(-100)
    const double min_value = std::exp(-100);
    Tensor clamped_input = clamp(input, ClampOptions().min(min_value));
    Tensor loss = -(target * log(clamped_input) + (1 - target) * log(1 - clamped_input));
    return handle_reduction(loss, mode);
}

BCELoss::BCELoss(ReductionMode mode)
    : mode_(mode) {}

void BCELoss::pretty_print(std::ostream &os) const {
    os << std::format("BCELoss(mode={:s})", reduction_mode_to_str(mode_));
}

// -------------------------------------------------

auto bce_with_logits_loss(const Tensor &input, const Tensor &target, ReductionMode mode) -> Tensor {
    return bce_loss(sigmoid(input), target, mode);
}

BCEWithLogitsLoss::BCEWithLogitsLoss(ReductionMode mode)
    : mode_(mode) {}

void BCEWithLogitsLoss::pretty_print(std::ostream &os) const {
    os << std::format("BCEWithLogitsLoss(mode={:s})", reduction_mode_to_str(mode_));
}

// -------------------------------------------------

auto huber_loss(const Tensor &input, const Tensor &target, double delta, ReductionMode mode) -> Tensor {
    Tensor mask = abs(input - target) < delta;
    Tensor under = pow(input - target, 2) / 2;
    Tensor over = delta * (abs(input - target) - (0.5 * delta));
    Tensor loss = where(mask, under, over);
    return handle_reduction(loss, mode);
}

HuberLoss::HuberLoss(double delta, ReductionMode mode)
    : delta_(delta), mode_(mode) {}

void HuberLoss::pretty_print(std::ostream &os) const {
    os << std::format("HuberLoss(delta={:f}, mode={:s})", delta_, reduction_mode_to_str(mode_));
}

// -------------------------------------------------

auto smooth_l1_loss(const Tensor &input, const Tensor &target, double beta, ReductionMode mode) -> Tensor {
    Tensor mask = abs(input - target) < beta;
    Tensor under = pow(input - target, 2) / (2 * beta);
    Tensor over = (abs(input - target) - (0.5 * beta));
    Tensor loss = where(mask, under, over);
    return handle_reduction(loss, mode);
}

SmoothL1Loss::SmoothL1Loss(double beta, ReductionMode mode)
    : beta_(beta), mode_(mode) {}

void SmoothL1Loss::pretty_print(std::ostream &os) const {
    os << std::format("SmoothL1Loss(delta={:f}, mode={:s})", beta_, reduction_mode_to_str(mode_));
}

// -------------------------------------------------

auto soft_margin_loss(const Tensor &input, const Tensor &target, ReductionMode mode) -> Tensor {
    Tensor loss = log(1 + exp(-target * input));
    return handle_reduction(loss, mode);
}

SoftMarginLoss::SoftMarginLoss(ReductionMode mode)
    : mode_(mode) {}

void SoftMarginLoss::pretty_print(std::ostream &os) const {
    os << std::format("SoftMarginLoss(mode={:s})", reduction_mode_to_str(mode_));
}

// -------------------------------------------------

}    // namespace tinytensor::nn
