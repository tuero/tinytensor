// sgd.h
// Stochastic Gradient Descent optimizer

#ifndef TINYTENSOR_NN_OPTIMIZER_SGD_H_
#define TINYTENSOR_NN_OPTIMIZER_SGD_H_

#include <tt/optim/optimizer.h>
#include <tt/tensor.h>

#include <functional>
#include <string>
#include <vector>

namespace tinytensor::optim {

// Options for SGD
// @note See https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
struct SGDOptions {
    RegularizationMode regularization_mode = RegularizationMode::l2;
    double weight_decay = 0;
    double momentum = 0;
    bool use_nesterov = false;
    bool maximize = false;
};

class SGD : public Optimizer {
    using TensorRefList = std::vector<std::reference_wrapper<Tensor>>;

public:
    /**
     * Create an SGD optimizer
     * @note See https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
     * @param params Parameters the optimizer should optimize
     * @param learning_rate The learning rate
     * @param options Additional options for SGD
     */
    SGD(const TensorRefList &params, double learning_rate, const SGDOptions &options = {});

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
     * @note See https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
     */
    void step() override;

public:
    double learning_rate_;
    SGDOptions options_;
    std::vector<Tensor> velocities_;
};

}    // namespace tinytensor::optim

#endif    // TINYTENSOR_NN_OPTIMIZER_SGD_H_
