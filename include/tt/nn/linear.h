// linear.h
// Linear layer module

#ifndef TINYTENSOR_NN_LINEAR_H_
#define TINYTENSOR_NN_LINEAR_H_

#include <tt/device.h>
#include <tt/export.h>
#include <tt/nn/module.h>
#include <tt/scalar.h>
#include <tt/tensor.h>

#include <memory>
#include <optional>
#include <ostream>
#include <string>

namespace tinytensor::nn {

// A linear layer
class TINYTENSOR_EXPORT Linear : public Module {
public:
    /**
     * Construct a linear layer
     * @param in_features Number of input features
     * @param out_features Number of output features
     * @param bias Boolean flag if a bias should be used
     * @param dtype The dtype of the weights
     * @param device The device the weights should be initialized on
     */
    Linear(int in_features, int out_features, bool bias = true, ScalarType dtype = kDefaultFloat, Device device = kCPU);

    /**
     * Forward pass for Linear layer
     * Shape of input should be (*, in_features) (see torch documentation)
     * @param input The input tensor
     * @return Output tensor of shape (*, out_features)
     */
    [[nodiscard]] auto forward(const Tensor &input) const -> Tensor;

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "Linear";
    }

    std::shared_ptr<Tensor> weight;
    std::optional<std::shared_ptr<Tensor>> bias;

private:
    int in_features_;
    int out_features_;
};

}    // namespace tinytensor::nn

#endif    // TINYTENSOR_NN_LINEAR_H_
