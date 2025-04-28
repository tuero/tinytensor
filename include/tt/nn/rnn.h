// rnn.h
// RNN layer module

#ifndef TINYTENSOR_NN_RNN_H_
#define TINYTENSOR_NN_RNN_H_

#include <tt/device.h>
#include <tt/nn/module.h>
#include <tt/scalar.h>
#include <tt/tensor.h>

#include <memory>
#include <optional>
#include <ostream>
#include <string>

namespace tinytensor::nn {

// Options for RNN
struct RNNOptions {
    int num_layers = 1;
    bool bias = true;
    bool batch_first = false;
    bool bidirectional = false;
};

// An RNN layer
// See https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
class RNN : public Module {
public:
    // Output packed data from RNN
    struct Output {
        Tensor output;    // Output
        Tensor h;         // The final hidden state for each element in the batch
    };

    /**
     * Construct a rnn layer
     * @param input_size Number of input features
     * @param hidden_size Number of features for hidden state
     * @param options The RNN options
     * @param dtype The dtype of the weights
     * @param device The device the weights should be initialized on
     */
    RNN(int input_size,
        int hidden_size,
        const RNNOptions &options = {},
        ScalarType dtype = kDefaultFloat,
        Device device = kCPU);

    /**
     * Forward pass for RNN
     * @param input Tensor of shape (L, input_size) for unbatched input, (L, B, input_size) if batch_first=false,
     *   or (B, L, input_size) when batch_first=true, for L=length and B=batch_size
     * @param h Optional initial hidden state, initialized to zero if not provided. Shape is expected to be
     *   (D*num_layers, hidden_size) for unbatched input, or (D*num_layers, B, hidden_size) for batched input,
     *   for B=batch_size and D=2 if bidirectional, 1 otherwise
     * @return Output of RNN, final hidden state for each element in the sequence.
     *   For unbatched input, output has shape (L, D*hidden_size), (L, batch_size, D*hidden_size) if batch_first=false,
     *   or (batch_size, L, D*hidden_size) when batch_first=true
     *   For unbatched input, hidden state has shape (D*num_layers, hidden_size),
     *   and (D*num_layers, batch_size, hidden_size) for batched input
     */
    [[nodiscard]] auto forward(const Tensor &input, const std::optional<Tensor> &h = std::nullopt) const -> Output;

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "RNN";
    }

    CheckedVec<std::shared_ptr<Tensor>> weights_ih;
    CheckedVec<std::shared_ptr<Tensor>> weights_ih_reverse;
    CheckedVec<std::shared_ptr<Tensor>> weights_hh;
    CheckedVec<std::shared_ptr<Tensor>> weights_hh_reverse;
    CheckedVec<std::shared_ptr<Tensor>> biases_ih;
    CheckedVec<std::shared_ptr<Tensor>> biases_ih_reverse;
    CheckedVec<std::shared_ptr<Tensor>> biases_hh;
    CheckedVec<std::shared_ptr<Tensor>> biases_hh_reverse;

private:
    int input_size_;
    int hidden_size_;
    RNNOptions options_;
};

}    // namespace tinytensor::nn

#endif    // TINYTENSOR_NN_RNN_H_
