// activation.cpp
// Activation layer modules

#include <tt/device.h>
#include <tt/nn/activation.h>
#include <tt/nn/init.h>
#include <tt/scalar.h>
#include <tt/tensor.h>

#include <cmath>
#include <format>
#include <ostream>

namespace tinytensor::nn {

// -------------------------------------------------

auto Sigmoid::forward(const Tensor &input) const -> Tensor {
    return sigmoid(input);
}

void Sigmoid::pretty_print(std::ostream &os) const {
    os << "Sigmoid()";
}

// -------------------------------------------------

auto LogSigmoid::forward(const Tensor &input) const -> Tensor {
    return log_sigmoid(input);
}

void LogSigmoid::pretty_print(std::ostream &os) const {
    os << "LogSigmoid()";
}

// -------------------------------------------------

auto HardSigmoid::forward(const Tensor &input) const -> Tensor {
    return hardsigmoid(input);
}

void HardSigmoid::pretty_print(std::ostream &os) const {
    os << "HardSigmoid()";
}

// -------------------------------------------------

Softplus::Softplus(double beta, double threshold)
    : beta_(beta), threshold_(threshold) {}

auto Softplus::forward(const Tensor &input) const -> Tensor {
    return softplus(input);
}

void Softplus::pretty_print(std::ostream &os) const {
    os << std::format("Softplus(beta={:f}, threshold={:f})", beta_, threshold_);
}

// -------------------------------------------------

auto ReLU::forward(const Tensor &input) const -> Tensor {
    return relu(input);
}

void ReLU::pretty_print(std::ostream &os) const {
    os << "Relu()";
}

// -------------------------------------------------

auto ReLU6::forward(const Tensor &input) const -> Tensor {
    return relu6(input);
}

void ReLU6::pretty_print(std::ostream &os) const {
    os << "Relu6()";
}

// -------------------------------------------------

LeakyReLU::LeakyReLU(double negative_slope)
    : negative_slope_(negative_slope) {}

auto LeakyReLU::forward(const Tensor &input) const -> Tensor {
    return leaky_relu(input, negative_slope_);
}

void LeakyReLU::pretty_print(std::ostream &os) const {
    os << std::format("LeakyReLU(negative_slope={:f})", negative_slope_);
}

// -------------------------------------------------

ELU::ELU(double alpha)
    : alpha_(alpha) {}

auto ELU::forward(const Tensor &input) const -> Tensor {
    return elu(input, alpha_);
}

void ELU::pretty_print(std::ostream &os) const {
    os << std::format("ELU(alpha={:f})", alpha_);
}

// -------------------------------------------------

auto SELU::forward(const Tensor &input) const -> Tensor {
    return selu(input);
}

void SELU::pretty_print(std::ostream &os) const {
    os << "SELU()";
}

// -------------------------------------------------

auto SiLU::forward(const Tensor &input) const -> Tensor {
    return silu(input);
}

void SiLU::pretty_print(std::ostream &os) const {
    os << "SiLU()";
}

// -------------------------------------------------

auto Tanh::forward(const Tensor &input) const -> Tensor {
    return tanh(input);
}

void Tanh::pretty_print(std::ostream &os) const {
    os << "Tanh()";
}

// -------------------------------------------------

HardTanh::HardTanh(double min, double max)
    : min_(min), max_(max) {}

auto HardTanh::forward(const Tensor &input) const -> Tensor {
    return hardtanh(input, min_, max_);
}

void HardTanh::pretty_print(std::ostream &os) const {
    os << std::format("HardTanh(min={:f}, max={:f})", min_, max_);
}

// -------------------------------------------------

auto Softsign::forward(const Tensor &input) const -> Tensor {
    return softsign(input);
}

void Softsign::pretty_print(std::ostream &os) const {
    os << "Softsign()";
}

// -------------------------------------------------

Softmax::Softmax(int dim)
    : dim_(dim) {}

auto Softmax::forward(const Tensor &input) const -> Tensor {
    return softmax(input, dim_);
}

void Softmax::pretty_print(std::ostream &os) const {
    os << std::format("Softmax(dim={:d})", dim_);
}

// -------------------------------------------------

}    // namespace tinytensor::nn
