// loss.h
// Loss layer functions and modules

#ifndef TINYTENSOR_NN_LOSS_H_
#define TINYTENSOR_NN_LOSS_H_

#include <tt/export.h>
#include <tt/nn/module.h>
#include <tt/tensor.h>

#include <memory>
#include <optional>
#include <ostream>
#include <string>

namespace tinytensor::nn {

// Reduction modes used for the loss functions
enum class TINYTENSOR_EXPORT ReductionMode {
    none,
    mean,
    sum,
    batch_mean,
};

// -------------------------------------------------

/**
 * Compute the L1 loss between input and target
 * @note See https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html
 * @param input The input
 * @param target The target
 * @param mode The reduction mode
 * @return The loss result
 */
[[nodiscard]] TINYTENSOR_EXPORT auto
    l1_loss(const Tensor &input, const Tensor &target, ReductionMode mode = ReductionMode::mean) -> Tensor;

class TINYTENSOR_EXPORT L1Loss : public Module {
public:
    /**
     * Create a L1Loss module
     * @param mode The reduction mode
     */
    L1Loss(ReductionMode mode = ReductionMode::mean);

    /**
     * Compute the L1 loss between input and target
     * @param input The input
     * @param target The target
     * @return The loss result
     */
    [[nodiscard]] auto forward(const Tensor &input, const Tensor &target) const -> Tensor {
        return l1_loss(input, target, mode_);
    }

    /**
     * Compute the L1 loss between input and target
     * @param input The input
     * @param target The target
     * @return The loss result
     */
    [[nodiscard]] auto operator()(const Tensor &input, const Tensor &target) const -> Tensor {
        return forward(input, target);
    }

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "L1Loss";
    }

private:
    ReductionMode mode_;
};

// -------------------------------------------------

/**
 * Compute the MSE loss between input and target
 * @note See https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html
 * @param input The input
 * @param target The target
 * @param mode The reduction mode
 * @return The loss result
 */
[[nodiscard]] TINYTENSOR_EXPORT auto
    mse_loss(const Tensor &input, const Tensor &target, ReductionMode mode = ReductionMode::mean) -> Tensor;

class TINYTENSOR_EXPORT MSELoss : public Module {
public:
    /**
     * Create a MSELoss module
     * @param mode The reduction mode
     */
    MSELoss(ReductionMode mode = ReductionMode::mean);

    /**
     * Compute the MSE loss between input and target
     * @param input The input
     * @param target The target
     * @return The loss result
     */
    [[nodiscard]] auto forward(const Tensor &input, const Tensor &target) const -> Tensor {
        return mse_loss(input, target, mode_);
    }

    /**
     * Compute the MSE loss between input and target
     * @param input The input
     * @param target The target
     * @return The loss result
     */
    [[nodiscard]] auto operator()(const Tensor &input, const Tensor &target) const -> Tensor {
        return forward(input, target);
    }

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "MSELoss";
    }

private:
    ReductionMode mode_;
};

// -------------------------------------------------

/**
 * Compute the Cross Entropy loss between input and target
 * @note See https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
 * @param input The input, which should be of shape (C), (N,C), or (N,C,d_1,...,d_k) for k>=1
 * @param target_indices The target indices, which should be of shape (1), (N), or (N,d_1,...,d_k) for k>=1 following
 * the input shape options
 * @param weight An optional rescaling weight for each class, of shape (C)
 * @param mode The reduction mode
 * @return The loss result, of shape (1), (N), or (N,d_1,...,d_k) with k>=1 following the input shape options
 */
[[nodiscard]] TINYTENSOR_EXPORT auto cross_entropy_loss(
    const Tensor &input,
    const Tensor &target_indices,
    const std::optional<Tensor> &weight = std::nullopt,
    ReductionMode mode = ReductionMode::mean
) -> Tensor;

class TINYTENSOR_EXPORT CrossEntropyLoss : public Module {
public:
    /**
     * Create a CrossEntropyLoss module
     * @param weight An optional rescaling weight for each class, of shape (C)
     * @param mode The reduction mode
     */
    CrossEntropyLoss(const std::optional<Tensor> &weight = std::nullopt, ReductionMode mode = ReductionMode::mean);

    /**
     * Compute the Cross Entropy loss between input and target
     * @param input The input, which should be of shape (C), (N,C), or (N,C,d_1,...,d_k) for k>=1
     * @param target_indices The target indices, which should be of shape (1), (N), or (N,d_1,...,d_k) for k>=1
     * following the input shape options
     * @return The loss result, of shape (1), (N), or (N,d_1,...,d_k) with k>=1 following the input shape options
     */
    [[nodiscard]] auto forward(const Tensor &input, const Tensor &target_indices) const -> Tensor {
        return cross_entropy_loss(
            input,
            target_indices,
            weight_.has_value() ? std::make_optional<Tensor>(**weight_) : std::nullopt,
            mode_
        );
    }

    /**
     * Compute the Cross Entropy loss between input and target
     * @param input The input, which should be of shape (C), (N,C), or (N,C,d_1,...,d_k) for k>=1
     * @param target_indices The target indices, which should be of shape (1), (N), or (N,d_1,...,d_k) for k>=1
     * following the input shape options
     * @return The loss result, of shape (1), (N), or (N,d_1,...,d_k) with k>=1 following the input shape options
     */
    [[nodiscard]] auto operator()(const Tensor &input, const Tensor &target_indices) const -> Tensor {
        return forward(input, target_indices);
    }

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "CrossEntropyLoss";
    }

private:
    ReductionMode mode_;
    std::optional<std::shared_ptr<Tensor>> weight_;
};

// -------------------------------------------------

/**
 * Compute the Negative Log Likelihood loss between input and target
 * @note See https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html
 * @param input The input of log-probabilities, which should be of shape (C), (N,C), or (N,C,d_1,...,d_k) for k>=1
 * @param target_indices The target indices, which should be of shape (1), (N), or (N,d_1,...,d_k) for k>=1 following
 * the input shape options
 * @param weight An optional rescaling weight for each class, of shape (C)
 * @param mode The reduction mode
 * @return The loss result, of shape (1), (N), or (N,d_1,...,d_k) with k>=1 following the input shape options
 */
[[nodiscard]] TINYTENSOR_EXPORT auto nll_loss(
    const Tensor &input,
    const Tensor &target_indices,
    const std::optional<Tensor> &weight = std::nullopt,
    ReductionMode mode = ReductionMode::mean
) -> Tensor;

class TINYTENSOR_EXPORT NLLLoss : public Module {
public:
    /**
     * Create a NLLLoss module
     * @param weight An optional rescaling weight for each class, of shape (C)
     * @param mode The reduction mode
     */
    NLLLoss(const std::optional<Tensor> &weight = std::nullopt, ReductionMode mode = ReductionMode::mean);

    /**
     * Compute the Negative Log Likelihood loss between input and target
     * @param input The input of log-probabilities, which should be of shape (C), (N,C), or (N,C,d_1,...,d_k) for k>=1
     * @param target_indices The target indices, which should be of shape (1), (N), or (N,d_1,...,d_k) for k>=1
     * following the input shape options
     * @return The loss result, of shape (1), (N), or (N,d_1,...,d_k) with k>=1 following the input shape options
     */
    [[nodiscard]] auto forward(const Tensor &input, const Tensor &target_indices) const -> Tensor {
        return nll_loss(
            input,
            target_indices,
            weight_.has_value() ? std::make_optional<Tensor>(**weight_) : std::nullopt,
            mode_
        );
    }

    /**
     * Compute the Negative Log Likelihood loss between input and target
     * @param input The input of log-probabilities, which should be of shape (C), (N,C), or (N,C,d_1,...,d_k) for k>=1
     * @param target_indices The target indices, which should be of shape (1), (N), or (N,d_1,...,d_k) for k>=1
     * following the input shape options
     * @return The loss result, of shape (1), (N), or (N,d_1,...,d_k) with k>=1 following the input shape options
     */
    [[nodiscard]] auto operator()(const Tensor &input, const Tensor &target_indices) const -> Tensor {
        return forward(input, target_indices);
    }

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "NLLLoss";
    }

private:
    ReductionMode mode_;
    std::optional<std::shared_ptr<Tensor>> weight_;
};

// -------------------------------------------------

/**
 * Compute the KL-Divergence KL(Target||Model Output) between model output and target
 * @note reduction=batch_mean should probably be used here
 * @note See https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
 * @param input The model output, which should be in log-space
 * @param target The target, which can be in log-space (if log_target=true)
 * @param mode The reduction mode
 * @param log_target True if the target provided is in log-space, false if not
 * @return The loss result
 */
[[nodiscard]] TINYTENSOR_EXPORT auto kld_loss(
    const Tensor &input,
    const Tensor &target,
    ReductionMode mode = ReductionMode::mean,
    bool log_target = false
) -> Tensor;

class TINYTENSOR_EXPORT KLDivLoss : public Module {
public:
    /**
     * Create a KLDivLoss module
     * @param mode The reduction mode
     * @param log_target True if the target provided is in log-space, false if not
     */
    KLDivLoss(ReductionMode mode = ReductionMode::mean, bool log_target = false);

    /**
     * Compute the KL-Divergence KL(Target||Model Output) between model output and target
     * @param input The model output, which should be in log-space
     * @param target The target, which can be in log-space (if log_target=true)
     * @return The loss result
     */
    [[nodiscard]] auto forward(const Tensor &input, const Tensor &target) const -> Tensor {
        return kld_loss(input, target, mode_, log_target_);
    }

    /**
     * Compute the KL-Divergence KL(Target||Model Output) between model output and target
     * @param input The model output, which should be in log-space
     * @param target The target, which can be in log-space (if log_target=true)
     * @return The loss result
     */
    [[nodiscard]] auto operator()(const Tensor &input, const Tensor &target) const -> Tensor {
        return forward(input, target);
    }

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "KLDivLoss";
    }

private:
    ReductionMode mode_;
    bool log_target_;
};

// -------------------------------------------------

/**
 * Compute the Binary Cross Wntropy between input and target probabilities
 * @note Target values should be between 0 and 1
 * @note We clamp log(input) -100
 * @note See https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
 * @param input The input probabilities
 * @param target The target probabilities
 * @param mode The reduction mode
 * @return The loss result
 */
[[nodiscard]] TINYTENSOR_EXPORT auto
    bce_loss(const Tensor &input, const Tensor &target, ReductionMode mode = ReductionMode::mean) -> Tensor;

class TINYTENSOR_EXPORT BCELoss : public Module {
public:
    /**
     * Create a BCELoss module
     * @param mode The reduction mode
     */
    BCELoss(ReductionMode mode = ReductionMode::mean);

    /**
     * Compute the Binary Cross Wntropy between input and target probabilities
     * @note Target values should be between 0 and 1
     * @param input The input probabilities
     * @param target The target probabilities
     * @return The loss result
     */
    [[nodiscard]] auto forward(const Tensor &input, const Tensor &target) const -> Tensor {
        return bce_loss(input, target, mode_);
    }

    /**
     * Compute the Binary Cross Wntropy between input and target probabilities
     * @note Target values should be between 0 and 1
     * @param input The input probabilities
     * @param target The target probabilities
     * @return The loss result
     */
    [[nodiscard]] auto operator()(const Tensor &input, const Tensor &target) const -> Tensor {
        return forward(input, target);
    }

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "BCELoss";
    }

private:
    ReductionMode mode_;
};

// -------------------------------------------------

/**
 * Compute the combined Sigmoid and Binary Cross Entropy between input and target
 * @note Target values should be between 0 and 1
 * @note We clamp log(input) -100
 * @note See https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
 * @param input The input values
 * @param target The target probabilities
 * @param mode The reduction mode
 * @return The loss result
 */
[[nodiscard]] TINYTENSOR_EXPORT auto
    bce_with_logits_loss(const Tensor &input, const Tensor &target, ReductionMode mode = ReductionMode::mean) -> Tensor;

class TINYTENSOR_EXPORT BCEWithLogitsLoss : public Module {
public:
    /**
     * Create a BCEWithLogitsLoss module
     * @param mode The reduction mode
     */
    BCEWithLogitsLoss(ReductionMode mode = ReductionMode::mean);

    /**
     * Compute the combined Sigmoid and Binary Cross Entropy between input and target
     * @note Target values should be between 0 and 1
     * @param input The input values
     * @param target The target probabilities
     * @return The loss result
     */
    [[nodiscard]] auto forward(const Tensor &input, const Tensor &target) const -> Tensor {
        return bce_with_logits_loss(input, target, mode_);
    }

    /**
     * Compute the combined Sigmoid and Binary Cross Entropy between input and target
     * @note Target values should be between 0 and 1
     * @param input The input values
     * @param target The target probabilities
     * @return The loss result
     */
    [[nodiscard]] auto operator()(const Tensor &input, const Tensor &target) const -> Tensor {
        return forward(input, target);
    }

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "BCEWithLogitsLoss";
    }

private:
    ReductionMode mode_;
};

// -------------------------------------------------

/**
 * Compute the Huber loss between input and target
 * @note See https://pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html
 * @param input The input values
 * @param target The target values
 * @param delta Threshold between squared error (below) or absolute error (above)
 * @param mode The reduction mode
 * @return The loss result
 */
[[nodiscard]] TINYTENSOR_EXPORT auto
    huber_loss(const Tensor &input, const Tensor &target, double delta = 1.0, ReductionMode mode = ReductionMode::mean)
        -> Tensor;

class TINYTENSOR_EXPORT HuberLoss : public Module {
public:
    /**
     * Create a HuberLoss module
     * @param delta Threshold between squared error (below) or scaled l1 error (above)
     * @param mode The reduction mode
     */
    HuberLoss(double delta = 1.0, ReductionMode mode = ReductionMode::mean);

    /**
     * Compute the Huber loss between input and target
     * @param input The input values
     * @param target The target values
     * @return The loss result
     */
    [[nodiscard]] auto forward(const Tensor &input, const Tensor &target) const -> Tensor {
        return huber_loss(input, target, delta_, mode_);
    }

    /**
     * Compute the Huber loss between input and target
     * @param input The input values
     * @param target The target values
     * @return The loss result
     */
    [[nodiscard]] auto operator()(const Tensor &input, const Tensor &target) const -> Tensor {
        return forward(input, target);
    }

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "HuberLoss";
    }

private:
    double delta_;
    ReductionMode mode_;
};

// -------------------------------------------------

/**
 * Compute the Smooth L1 loss between input and target
 * @note See https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html
 * @param input The input values
 * @param target The target values
 * @param beta Threshold between squared error (below) or l1 error (above)
 * @param mode The reduction mode
 * @return The loss result
 */
[[nodiscard]] TINYTENSOR_EXPORT auto smooth_l1_loss(
    const Tensor &input,
    const Tensor &target,
    double beta = 1.0,
    ReductionMode mode = ReductionMode::mean
) -> Tensor;

class TINYTENSOR_EXPORT SmoothL1Loss : public Module {
public:
    /**
     * Create a SmoothL1Loss module
     * @param beta Threshold between squared error (below) or l1 error (above)
     * @param mode The reduction mode
     */
    SmoothL1Loss(double beta = 1.0, ReductionMode mode = ReductionMode::mean);

    /**
     * Compute the Smooth L1 loss between input and target
     * @param input The input values
     * @param target The target values
     * @return The loss result
     */
    [[nodiscard]] auto forward(const Tensor &input, const Tensor &target) const -> Tensor {
        return smooth_l1_loss(input, target, beta_, mode_);
    }

    /**
     * Compute the Smooth L1 loss between input and target
     * @param input The input values
     * @param target The target values
     * @return The loss result
     */
    [[nodiscard]] auto operator()(const Tensor &input, const Tensor &target) const -> Tensor {
        return forward(input, target);
    }

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "SmoothL1Loss";
    }

private:
    double beta_;
    ReductionMode mode_;
};

// -------------------------------------------------

/**
 * Compute the Smooth Margin loss between input and target
 * @note See https://pytorch.org/docs/stable/generated/torch.nn.SoftMarginLoss.html
 * @param input The input values
 * @param target The target values
 * @param mode The reduction mode
 * @return The loss result
 */
[[nodiscard]] TINYTENSOR_EXPORT auto
    soft_margin_loss(const Tensor &input, const Tensor &target, ReductionMode mode = ReductionMode::mean) -> Tensor;

class TINYTENSOR_EXPORT SoftMarginLoss : public Module {
public:
    /**
     * Create a SoftMarginLoss module
     * @param mode The reduction mode
     */
    SoftMarginLoss(ReductionMode mode = ReductionMode::mean);

    /**
     * Compute the soft margin loss between input and target
     * @param input The input values
     * @param target The target values
     * @return The loss result
     */
    [[nodiscard]] auto forward(const Tensor &input, const Tensor &target) const -> Tensor {
        return soft_margin_loss(input, target, mode_);
    }

    /**
     * Compute the soft margin loss between input and target
     * @param input The input values
     * @param target The target values
     * @return The loss result
     */
    [[nodiscard]] auto operator()(const Tensor &input, const Tensor &target) const -> Tensor {
        return forward(input, target);
    }

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "SoftMarginLoss";
    }

private:
    ReductionMode mode_;
};

// -------------------------------------------------

}    // namespace tinytensor::nn

#endif    // TINYTENSOR_NN_LOSS_H_
