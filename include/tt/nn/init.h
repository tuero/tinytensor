// init.h
// Initialization methods

#ifndef TINYTENSOR_NN_INIT_H_
#define TINYTENSOR_NN_INIT_H_

#include <tt/tensor.h>

#include <string>
#include <tuple>
#include <unordered_map>

namespace tinytensor::nn {

enum class GainActivation {
    linear,
    conv,
    sigmoid,
    tanh,
    relu,
    leaky_relu,
    selu
};

enum class FanMode {
    fan_in,
    fan_out
};

using GainActivationParams = std::unordered_map<std::string, double>;

/**
 * Calculcate the fan in and fan out
 * @param tensor The tensor, which must be permtued such that dimension 0 is the size of input features, and dimension 1
 * is the size of the output features
 * @return The fan_in and fan_out
 */
auto calc_fan_in_out(const Tensor &tensor) -> std::tuple<double, double>;

/**
 * Calculate the gain value for the given activation function
 * @note If using leaky_relu, 'negative_slope' must be given as a GainActivationParams
 * @gain_activation The activation function used
 * @params Optional params, depending on the activation function used
 * @return the gain value
 */
auto calc_gain(GainActivation gain_activation, const GainActivationParams &params = {}) -> double;

/**
 * Initialize the tensor using a Uniform(low, high) distribution
 * @param tensor The tensor to initialize
 * @param low The left end of the interval
 * @param high The right end of the interval
 */
void uniform_(Tensor tensor, double low, double high);

/**
 * Initialize the tensor using a Normal(mu, std^2) distribution
 * @param tensor The tensor to initialize
 * @param mu The mean of the distribution
 * @param std The variiance of the distribution
 */
void normal_(Tensor tensor, double low, double high);

/**
 * Initialize the tensor with a constant value
 * @param tensor The tensor to initialize
 * @param value The value to initialize with
 */
void constant_(Tensor tensor, double value);

/**
 * Initialize the tensor using Xavier Uniform Initialization
 * @note The tensor must be permtued such that dimension 0 is the size of input features, and dimension 1 is the size of
 * the output features
 * @param tensor The tensor to initialize
 * @param gain The gain value (see calc_gain)
 */
void xavier_uniform_(Tensor tensor, double gain = 1.0);

/**
 * Initialize the tensor using Xavier Normal Initialization
 * @note The tensor must be permtued such that dimension 0 is the size of input features, and dimension 1 is the size of
 * the output features
 * @param tensor The tensor to initialize
 * @param gain The gain value (see calc_gain)
 */
void xavier_normal_(Tensor tensor, double gain = 1.0);

/**
 * Initialize the tensor using Kaiming Uniform Initialization
 * @note The tensor must be permtued such that dimension 0 is the size of input features, and dimension 1 is the size of
 * the output features
 * @param tensor The tensor to initialize
 * @param fan_mode fan_in to preserve the magnitude of the variance of the weights in the forward pass, fan_out to
 * preserve the magnitude for the backward pass
 * @param gain The gain value (see calc_gain)
 */
void kaiming_uniform_(Tensor tensor, double gain = 1.0, FanMode fan_mode = FanMode::fan_in);

/**
 * Initialize the tensor using Kaiming Normal Initialization
 * @note The tensor must be permtued such that dimension 0 is the size of input features, and dimension 1 is the size of
 * the output features
 * @param tensor The tensor to initialize
 * @param fan_mode fan_in to preserve the magnitude of the variance of the weights in the forward pass, fan_out to
 * preserve the magnitude for the backward pass
 * @param gain The gain value (see calc_gain)
 */
void kaiming_normal_(Tensor tensor, double gain = 1.0, FanMode fan_mode = FanMode::fan_in);

}    // namespace tinytensor::nn

#endif    // TINYTENSOR_NN_INIT_H_
