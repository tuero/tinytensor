// rmsprop.h
// RMSprop optimizer
// https://arxiv.org/abs/1308.0850

#ifndef TINYTENSOR_NN_OPTIMIZER_RMSPROP_H_
#define TINYTENSOR_NN_OPTIMIZER_RMSPROP_H_

#include <tt/export.h>
#include <tt/optim/optimizer.h>
#include <tt/tensor.h>

#include <functional>
#include <string>
#include <vector>

namespace tinytensor::optim {

// Options for RMSprop
// @note See https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html
struct TINYTENSOR_EXPORT RMSpropOptions {
    RegularizationMode regularization_mode = RegularizationMode::l2;
    double weight_decay = 0;
    double momentum = 0;
    double alpha = 0.99;
    double eps = 1e-8;
    bool center = false;
    bool maximize = false;
};

class TINYTENSOR_EXPORT RMSprop : public Optimizer {
    using TensorRefList = std::vector<std::reference_wrapper<Tensor>>;

public:
    /**
     * Create an RMSprop optimizer
     * @note See https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html
     * @param params Parameters the optimizer should optimize
     * @param learning_rate The learning rate
     * @param options Additional options for RMSprop
     */
    RMSprop(const TensorRefList &params, double learning_rate, const RMSpropOptions &options = {});

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
     * @note See https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html
     */
    void step() override;

protected:
    double learning_rate_;
    RMSpropOptions options_;
    std::vector<Tensor> square_averages_;
    std::vector<Tensor> velocities_;
    std::vector<Tensor> centers_;
};

}    // namespace tinytensor::optim

#endif    // TINYTENSOR_NN_OPTIMIZER_RMSPROP_H_
