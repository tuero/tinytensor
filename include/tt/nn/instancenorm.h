// instancenorm.h
// InstanceNorm layer module
// Instancenorm is appled on each channel of channeled data, but layernorm usually applied to entire sample
// Instancenorm also usually doesn't apply affine transform

#ifndef TINYTENSOR_NN_INSTANCENORM_H_
#define TINYTENSOR_NN_INSTANCENORM_H_

#include <tt/device.h>
#include <tt/nn/module.h>
#include <tt/scalar.h>
#include <tt/tensor.h>

#include <memory>
#include <ostream>
#include <string>

namespace tinytensor::nn {

// Options for InstanceNorm
struct InstanceNormOptions {
    double eps = 1e-5;
    double momentum = 0.1;
    bool affine = false;                 // Whether affine params are learnable
    bool track_running_stats = false;    // Track running stats, if false always use batch stats
};

// An instance norm over 3d inputs
// https://pytorch.org/docs/stable/generated/torch.nn.InstanceNorm1d.html
class InstanceNorm1d : public Module {
public:
    /**
     * Construct a InstanceNorm1d layer
     * @param num_features Size C from expected inputs of shape (B, C)
     * @param options The InstanceNorm options
     * @param dtype The dtype of the weights
     * @param device The device the weights should be initialized on
     */
    InstanceNorm1d(
        int num_features,
        const InstanceNormOptions &options = {},
        ScalarType dtype = kDefaultFloat,
        Device device = kCPU
    );

    /**
     * Forward pass for InstanceNorm1d layer
     * @param input The input tensor
     * @return Output tensor
     */
    [[nodiscard]] auto forward(const Tensor &input) -> Tensor;

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "InstanceNorm1d";
    }

    std::shared_ptr<Tensor> gamma;
    std::shared_ptr<Tensor> beta;
    std::shared_ptr<Tensor> moving_mean;
    std::shared_ptr<Tensor> moving_var;

private:
    int num_features_;
    InstanceNormOptions options_;
};

// An instance norm over 4d inputs
// https://pytorch.org/docs/stable/generated/torch.nn.InstanceNorm2d.html
class InstanceNorm2d : public Module {
public:
    /**
     * Construct a InstanceNorm2d layer
     * @param num_features Size C from expected inputs of shape (B, C, H, W)
     * @param options The InstanceNorm options
     * @param dtype The dtype of the weights
     * @param device The device the weights should be initialized on
     */
    InstanceNorm2d(
        int num_features,
        const InstanceNormOptions &options = {},
        ScalarType dtype = kDefaultFloat,
        Device device = kCPU
    );

    /**
     * Forward pass for InstanceNorm2d layer
     * @param input The input tensor
     * @return Output tensor
     */
    [[nodiscard]] auto forward(const Tensor &input) -> Tensor;

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "InstanceNorm2d";
    }

    std::shared_ptr<Tensor> gamma;
    std::shared_ptr<Tensor> beta;
    std::shared_ptr<Tensor> moving_mean;
    std::shared_ptr<Tensor> moving_var;

private:
    int num_features_;
    InstanceNormOptions options_;
};

}    // namespace tinytensor::nn

#endif    // TINYTENSOR_NN_INSTANCENORM_H_
