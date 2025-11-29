// activation.h
// Activation layer modules

#ifndef TINYTENSOR_NN_ACTIVATION_H_
#define TINYTENSOR_NN_ACTIVATION_H_

#include <tt/export.h>
#include <tt/nn/module.h>
#include <tt/tensor.h>

#include <ostream>
#include <string>

namespace tinytensor::nn {

// A sigmoid activation layer layer
class TINYTENSOR_EXPORT Sigmoid : public Module {
public:
    Sigmoid() = default;

    [[nodiscard]] auto forward(const Tensor &input) const -> Tensor;

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "Sigmoid";
    }
};

// A log sigmoid activation layer layer
class TINYTENSOR_EXPORT LogSigmoid : public Module {
public:
    LogSigmoid() = default;

    [[nodiscard]] auto forward(const Tensor &input) const -> Tensor;

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "LogSigmoid";
    }
};

// A hard sigmoid activation layer layer
class TINYTENSOR_EXPORT HardSigmoid : public Module {
public:
    HardSigmoid() = default;

    [[nodiscard]] auto forward(const Tensor &input) const -> Tensor;

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "HardSigmoid";
    }
};

// A softplus activation layer layer
class TINYTENSOR_EXPORT Softplus : public Module {
public:
    /**
     * Construct a Softplus activation layer
     * @note https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html
     * @param beta The beta value for Softplus
     * @param threshold Values above this revert to a linear function
     */
    Softplus(double beta = 1, double threshold = 20);

    [[nodiscard]] auto forward(const Tensor &input) const -> Tensor;

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "Softplus";
    }

private:
    double beta_;
    double threshold_;
};

// A relu activation layer layer
class TINYTENSOR_EXPORT ReLU : public Module {
public:
    ReLU() = default;

    [[nodiscard]] auto forward(const Tensor &input) const -> Tensor;

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "ReLU";
    }
};

// A relu activation layer layer
class TINYTENSOR_EXPORT ReLU6 : public Module {
public:
    ReLU6() = default;

    [[nodiscard]] auto forward(const Tensor &input) const -> Tensor;

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "ReLU6";
    }
};

// A leaky relu activation layer layer
class TINYTENSOR_EXPORT LeakyReLU : public Module {
public:
    /**
     * Construct a LeakyReLU activation layer
     * @note https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html
     * @param negative_slope The negative slope value
     */
    LeakyReLU(double negative_slope = 0.01);

    [[nodiscard]] auto forward(const Tensor &input) const -> Tensor;

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "LeakyReLU";
    }

private:
    double negative_slope_;
};

// A ELU activation layer layer
class TINYTENSOR_EXPORT ELU : public Module {
public:
    /**
     * Construct a ELU activation layer
     * @note https://pytorch.org/docs/stable/generated/torch.nn.ELU.html
     * @param alpha The alpha value
     */
    ELU(double alpha = 0.01);

    [[nodiscard]] auto forward(const Tensor &input) const -> Tensor;

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "ELU";
    }

private:
    double alpha_;
};

// A SELU activation layer layer
class TINYTENSOR_EXPORT SELU : public Module {
public:
    [[nodiscard]] auto forward(const Tensor &input) const -> Tensor;

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "SELU";
    }
};

// A sigmoid linear unit activation layer layer
class TINYTENSOR_EXPORT SiLU : public Module {
public:
    [[nodiscard]] auto forward(const Tensor &input) const -> Tensor;

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "SiLU";
    }
};

// A tanh activation layer layer
class TINYTENSOR_EXPORT Tanh : public Module {
public:
    [[nodiscard]] auto forward(const Tensor &input) const -> Tensor;

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "Tanh";
    }
};

// A hard tanh activation layer layer
class TINYTENSOR_EXPORT HardTanh : public Module {
public:
    /**
     * Construct a HardTanh activation layer
     * @note https://pytorch.org/docs/stable/generated/torch.nn.Hardtanh.html
     * @param min Minimum value of the linear region range
     * @param max Maximum value of the linear region range
     */
    HardTanh(double min = -1, double max = 1);

    [[nodiscard]] auto forward(const Tensor &input) const -> Tensor;

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "HardTanh";
    }

private:
    double min_;
    double max_;
};

// A softsign activation layer layer
class TINYTENSOR_EXPORT Softsign : public Module {
public:
    [[nodiscard]] auto forward(const Tensor &input) const -> Tensor;

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "Softsign";
    }
};

// A softmax activation layer layer
class TINYTENSOR_EXPORT Softmax : public Module {
public:
    /**
     * Construct a Softmax activation layer
     * @note https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
     * @param dim The dimension along which the Softmax will be computed
     */
    Softmax(int dim = -1);

    [[nodiscard]] auto forward(const Tensor &input) const -> Tensor;

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "Softmax";
    }

private:
    int dim_;
};

}    // namespace tinytensor::nn

#endif    // TINYTENSOR_NN_ACTIVATION_H_
