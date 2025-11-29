// pool2d.h
// Pooling layer modules

#ifndef TINYTENSOR_NN_POOL2D_H_
#define TINYTENSOR_NN_POOL2D_H_

#include <tt/export.h>
#include <tt/nn/module.h>
#include <tt/tensor.h>

#include <ostream>
#include <string>

namespace tinytensor::nn {

// A min pooling layer
class TINYTENSOR_EXPORT MinPool2d : public Module {
public:
    /**
     * Construct a MinPool2d layer
     * @param kernel_size The kernel size
     * @param stride The stride
     * @param padding The amount of padding to apply to each side of the input
     */
    MinPool2d(int kernel_size, int stride, int padding);

    [[nodiscard]] auto forward(const Tensor &input) const -> Tensor;

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "MinPool2D";
    }

private:
    int kernel_size_;
    int stride_;
    int padding_;
};

// A max pooling layer
class TINYTENSOR_EXPORT MaxPool2d : public Module {
public:
    /**
     * Construct a MaxPool2d layer
     * @param kernel_size The kernel size
     * @param stride The stride
     * @param padding The amount of padding to apply to each side of the input
     */
    MaxPool2d(int kernel_size, int stride, int padding);

    [[nodiscard]] auto forward(const Tensor &input) const -> Tensor;

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "MaxPool2D";
    }

private:
    int kernel_size_;
    int stride_;
    int padding_;
};

// A average pooling layer
class TINYTENSOR_EXPORT AvgPool2d : public Module {
public:
    /**
     * Construct a AvgPool2d layer
     * @param kernel_size The kernel size
     * @param stride The stride
     * @param padding The amount of padding to apply to each side of the input
     */
    AvgPool2d(int kernel_size, int stride, int padding);

    [[nodiscard]] auto forward(const Tensor &input) const -> Tensor;

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "AvgPool2D";
    }

private:
    int kernel_size_;
    int stride_;
    int padding_;
};

}    // namespace tinytensor::nn

#endif    // TINYTENSOR_NN_POOL2D_H_
