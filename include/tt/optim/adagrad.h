// adagrad.h
// Adagrad optimizer
// https://jmlr.org/papers/v12/duchi11a.html

#ifndef TINYTENSOR_NN_OPTIMIZER_ADAGRAD_H_
#define TINYTENSOR_NN_OPTIMIZER_ADAGRAD_H_

#include <tt/optim/optimizer.h>
#include <tt/tensor.h>

#include <functional>
#include <string>
#include <vector>

namespace tinytensor::optim {

// Options for Adagrad
// @note See https://pytorch.org/docs/stable/generated/torch.optim.Adagrad.html
struct AdagradOptions {
    RegularizationMode regularization_mode = RegularizationMode::l2;
    double weight_decay = 0;
    double learning_rate_decay = 0;
    double eps = 1e-10;
    bool maximize = false;
};

class Adagrad : public Optimizer {
    using TensorRefList = std::vector<std::reference_wrapper<Tensor>>;

public:
    /**
     * Create an Adagrad optimizer
     * @note See https://pytorch.org/docs/stable/generated/torch.optim.Adagrad.html
     * @param params Parameters the optimizer should optimize
     * @param learning_rate The learning rate
     * @param options Additional options for Adagrad
     */
    Adagrad(const TensorRefList &params, double learning_rate, const AdagradOptions &options = {});

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
     * @note See https://pytorch.org/docs/stable/generated/torch.optim.Adagrad.html
     */
    void step() override;

protected:
    double learning_rate_;
    AdagradOptions options_;
    std::vector<int> steps_;
    std::vector<Tensor> state_sums_;
};

}    // namespace tinytensor::optim

#endif    // TINYTENSOR_NN_OPTIMIZER_ADAGRAD_H_
