// dropout.h
// Dropout layer module

#ifndef TINYTENSOR_NN_DROPOUT_H_
#define TINYTENSOR_NN_DROPOUT_H_

#include <tt/nn/module.h>
#include <tt/tensor.h>

#include <ostream>
#include <string>

namespace tinytensor::nn {

// A dropout layer
class Dropout : public Module {
public:
    /**
     * Construct a Dropout layer
     * @note Elements not zeroed are also scaled by 1/(1-p) during training
     * @note If evaluation is set, this layer is equivalent to the identity
     * @note See https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html
     * @param p The probability for elements to be zeroed
     */
    Dropout(double p = 0.5);

    [[nodiscard]] auto forward(const Tensor &input) const -> Tensor;

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "Dropout";
    }

private:
    double p_;
};

}    // namespace tinytensor::nn

#endif    // TINYTENSOR_NN_DROPOUT_H_
