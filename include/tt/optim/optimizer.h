// optimizer.h
// Base optimizer

#ifndef TINYTENSOR_NN_OPTIMIZER_H_
#define TINYTENSOR_NN_OPTIMIZER_H_

#include <tt/export.h>
#include <tt/tensor.h>

#include <functional>
#include <string>
#include <vector>

namespace tinytensor::optim {

enum class TINYTENSOR_EXPORT RegularizationMode {
    l1,
    l2
};

class TINYTENSOR_EXPORT Optimizer {
public:
    virtual ~Optimizer() = default;
    Optimizer() = default;

    // @NOTE: Revisit these
    Optimizer(const Optimizer &) = delete;
    Optimizer(Optimizer &&) = default;
    auto operator=(const Optimizer &) -> Optimizer & = delete;
    auto operator=(Optimizer &&) -> Optimizer & = delete;

    /**
     * Perform a single optimization step of the optimizer algorithm
     */
    virtual void step() = 0;

    /**
     * Save the internal state of the optimizer
     * @param path The path to save the optimizer state
     */
    virtual void save(const std::string &path) const = 0;

    /**
     * Load the internal state of the optimizer
     * @param path The path to the saved optimizer state
     */
    virtual void load(const std::string &path) = 0;

    /**
     * Zero the gradients for all saved tensors
     */
    void zero_grad();

    /**
     * Add parameters to the optimizer
     * @param params The parameters to add
     */
    virtual void add_parameters(const std::vector<std::reference_wrapper<Tensor>> &params);

protected:
    std::vector<std::reference_wrapper<Tensor>> params_;
};

}    // namespace tinytensor::optim

#endif    // TINYTENSOR_NN_OPTIMIZER_H_
