// layernorm.h
// LayerNorm layer module

#ifndef TINYTENSOR_NN_LAYERNORM_H_
#define TINYTENSOR_NN_LAYERNORM_H_

#include <tt/device.h>
#include <tt/nn/module.h>
#include <tt/scalar.h>
#include <tt/shape.h>
#include <tt/tensor.h>

#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace tinytensor::nn {

// Options for LayerNorm
struct LayerNormOptions {
    double eps = 1e-5;
    bool affine = true;    // Elementwise learnable affine
    bool bias = true;      // Learnable bias
};

// A layernorm layer
// https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
class LayerNorm : public Module {
public:
    /**
     * Construct a LayerNorm layer
     * @param normalized_shape The expected shape over the last N dimensions, which the mean/var is taken over that many
     * inner dimeisions
     * @param options The LayerNorm options
     * @param dtype The dtype of the weights
     * @param device The device the weights should be initialized on
     */
    LayerNorm(
        const Shape &normalized_shape,
        const LayerNormOptions &options = {},
        ScalarType dtype = kDefaultFloat,
        Device device = kCPU
    );

    /**
     * Forward pass for LayerNorm layer
     * @param input The input tensor
     * @return Output tensor
     */
    [[nodiscard]] auto forward(const Tensor &input) -> Tensor;

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "LayerNorm";
    }

    std::shared_ptr<Tensor> gamma;    // weight
    std::shared_ptr<Tensor> beta;     // bias

private:
    Shape normalized_shape_;
    LayerNormOptions options_;
    std::vector<int> normalized_dims_;
};

}    // namespace tinytensor::nn

#endif    // TINYTENSOR_NN_LAYERNORM_H_
