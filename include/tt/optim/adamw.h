// adamw.h
// AdamW optimizer
// https://arxiv.org/abs/1711.05101

#ifndef TINYTENSOR_NN_OPTIMIZER_ADAMW_H_
#define TINYTENSOR_NN_OPTIMIZER_ADAMW_H_

#include <tt/export.h>
#include <tt/optim/optimizer.h>
#include <tt/tensor.h>

#include <functional>
#include <string>
#include <vector>

namespace tinytensor::optim {

struct TINYTENSOR_EXPORT AdamWBetas {
    double beta1;
    double beta2;
};

// Options for AdamW
// @note See https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
struct TINYTENSOR_EXPORT AdamWOptions {
    RegularizationMode regularization_mode = RegularizationMode::l2;
    double weight_decay = 0;
    AdamWBetas betas = {.beta1 = 0.9, .beta2 = 0.999};
    double eps = 1e-8;
    bool use_amsgrad = false;
    bool maximize = false;
};

class TINYTENSOR_EXPORT AdamW : public Optimizer {
    using TensorRefList = std::vector<std::reference_wrapper<Tensor>>;

public:
    /**
     * Create an AdamW optimizer
     * @note See https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
     * @param params Parameters the optimizer should optimize
     * @param learning_rate The learning rate
     * @param options Additional options for AdamW
     */
    AdamW(const TensorRefList &params, double learning_rate, const AdamWOptions &options = {});

    /**
     * Save the internal state of the optimizer
     * @param path The path to save the optimizer state
     */
    void save(const std::string &path) const override;

    /**
     * Load the internal state of the optimizer
     * @param path The path to the saved optimizer state
     */
    void load(const std::string &path) override;

    /**
     * Add parameters to the optimizer
     * @param params The parameters to add
     */
    void add_parameters(const std::vector<std::reference_wrapper<Tensor>> &params) override;

    /**
     * Perform a single optimization step of the optimizer algorithm
     * @note See https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
     */
    void step() override;

protected:
    double learning_rate_;
    AdamWOptions options_;
    std::vector<Tensor> first_moments_;
    std::vector<Tensor> second_moments_;
    std::vector<Tensor> second_moments_max_;
    std::vector<int> steps_;
};

}    // namespace tinytensor::optim

#endif    // TINYTENSOR_NN_OPTIMIZER_ADAMW_H_
