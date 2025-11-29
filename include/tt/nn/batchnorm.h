// batchnorm.h
// BatchNorm layer module

#ifndef TINYTENSOR_NN_BATCHNORM_H_
#define TINYTENSOR_NN_BATCHNORM_H_

#include <tt/device.h>
#include <tt/export.h>
#include <tt/nn/module.h>
#include <tt/scalar.h>
#include <tt/tensor.h>

#include <memory>
#include <ostream>
#include <string>

namespace tinytensor::nn {

// Options for BatchNorm
struct TINYTENSOR_EXPORT BatchNormOptions {
    double eps = 1e-5;
    double momentum = 0.1;
    bool affine = true;                 // Whether affine params are learnable
    bool track_running_stats = true;    // Track running stats, if false always use batch stats
};

// A batch norm over 2d or 3d inputs
// https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
class TINYTENSOR_EXPORT BatchNorm1d : public Module {
public:
    /**
     * Construct a BatchNorm1d layer
     * @param num_features Size C from expected inputs of shape (B, C) or (B, C, L)
     * @param options The BatchNorm options
     * @param dtype The dtype of the weights
     * @param device The device the weights should be initialized on
     */
    BatchNorm1d(
        int num_features,
        const BatchNormOptions &options = {},
        ScalarType dtype = kDefaultFloat,
        Device device = kCPU
    );

    /**
     * Forward pass for BatchNorm1d layer
     * @param input The input tensor
     * @return Output tensor
     */
    [[nodiscard]] auto forward(const Tensor &input) -> Tensor;

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "BatchNorm1d";
    }

    std::shared_ptr<Tensor> gamma;
    std::shared_ptr<Tensor> beta;
    std::shared_ptr<Tensor> moving_mean;
    std::shared_ptr<Tensor> moving_var;

private:
    int num_features_;
    BatchNormOptions options_;
};

// A batch norm over 4d inputs
// https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
class TINYTENSOR_EXPORT BatchNorm2d : public Module {
public:
    /**
     * Construct a BatchNorm2d layer
     * @param num_features Size C from expected inputs of shape (B, C, H, W)
     * @param options The BatchNorm options
     * @param dtype The dtype of the weights
     * @param device The device the weights should be initialized on
     */
    BatchNorm2d(
        int num_features,
        const BatchNormOptions &options = {},
        ScalarType dtype = kDefaultFloat,
        Device device = kCPU
    );

    /**
     * Forward pass for BatchNorm2d layer
     * @param input The input tensor
     * @return Output tensor
     */
    [[nodiscard]] auto forward(const Tensor &input) -> Tensor;

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "BatchNorm2d";
    }

    std::shared_ptr<Tensor> gamma;
    std::shared_ptr<Tensor> beta;
    std::shared_ptr<Tensor> moving_mean;
    std::shared_ptr<Tensor> moving_var;

private:
    int num_features_;
    BatchNormOptions options_;
};

}    // namespace tinytensor::nn

#endif    // TINYTENSOR_NN_BATCHNORM_H_
