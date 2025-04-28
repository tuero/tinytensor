// conv2d.h
// Conv2d layer module

#ifndef TINYTENSOR_NN_CONV2D_H_
#define TINYTENSOR_NN_CONV2D_H_

#include <tt/device.h>
#include <tt/nn/module.h>
#include <tt/scalar.h>
#include <tt/tensor.h>

#include <memory>
#include <optional>
#include <ostream>
#include <string>

namespace tinytensor::nn {

// A 2D Convolutional layer
class Conv2d : public Module {
public:
    /**
     * Construct a Conv2d layer
     * @param in_channels Number of input input_channels
     * @param out_channels Number of output channels
     * @param kernel_size The kernel size
     * @param stride The stride
     * @param padding The amount of padding to apply to each side of the input
     * @param bias Boolean flag if a bias should be used
     * @param dtype The dtype of the weights
     * @param device The device the weights should be initialized on
     */
    Conv2d(
        int in_channels,
        int out_channels,
        int kernel_size,
        int stride,
        int padding,
        bool bias = true,
        ScalarType dtype = kDefaultFloat,
        Device device = kCPU
    );

    /**
     * Forward pass for Conv2d layer
     * Shape of input should be (batch_size, in_channels, h, w)
     * @param input The input tensor
     * @return Output tensor
     */
    [[nodiscard]] auto forward(const Tensor &input) const -> Tensor;

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "Conv2d";
    }

    std::shared_ptr<Tensor> weight;
    std::optional<std::shared_ptr<Tensor>> bias;

private:
    int in_channels_;
    int out_channels_;
    int kernel_size_;
    int stride_;
    int padding_;
    bool has_bias_;
};

}    // namespace tinytensor::nn

#endif    // TINYTENSOR_NN_CONV2D_H_
