// lstm.h
// LSTM layer module

#ifndef TINYTENSOR_NN_LSTM_H_
#define TINYTENSOR_NN_LSTM_H_

#include <tt/device.h>
#include <tt/export.h>
#include <tt/nn/module.h>
#include <tt/scalar.h>
#include <tt/tensor.h>

#include <memory>
#include <optional>
#include <ostream>
#include <string>

namespace tinytensor::nn {

// Options for LSTM
struct TINYTENSOR_EXPORT LSTMOptions {
    int num_layers = 1;
    bool bias = true;
    bool batch_first = false;
    bool bidirectional = false;
};

// An LSTM layer
// See https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
class TINYTENSOR_EXPORT LSTM : public Module {
public:
    // Input initial hidden and cell state
    struct InitState {
        Tensor h;
        Tensor c;
    };

    // Output packed data from LSTM
    struct Output {
        Tensor output;    // Output
        Tensor h;         // The final hidden state for each element in the batch
        Tensor c;         // The final cell state for each element in the batch
    };

    /**
     * Construct an lstm layer
     * @param input_size Number of input features
     * @param hidden_size Number of features for hidden state
     * @param options The LSTM options
     * @param dtype The dtype of the weights
     * @param device The device the weights should be initialized on
     */
    LSTM(
        int input_size,
        int hidden_size,
        const LSTMOptions &options = {},
        ScalarType dtype = kDefaultFloat,
        Device device = kCPU
    );

    /**
     * Forward pass for LSTM
     * @param input Tensor of shape (L, input_size) for unbatched input, (L, B, input_size) if batch_first=false,
     *   or (B, L, input_size) when batch_first=true, for L=length and B=batch_size
     * @param init_state Optional initial hidden and cell state, initialized to zero if not provided. Shape is expected
     *   to be (D*num_layers, hidden_size) for unbatched input, or (D*num_layers, B, hidden_size) for batched input, for
     *   B=batch_size and D=2 if bidirectional, 1 otherwise for both h and c
     * @return Output of LSTM, final hidden state and final cell state for each element in the sequence.
     *   For unbatched input, output has shape (L, D*hidden_size), (L, batch_size, D*hidden_size) if batch_first=false,
     *   or (batch_size, L, D*hidden_size) when batch_first=true
     *   For unbatched input, hidden and cell states have shape (D*num_layers, hidden_size),
     *   and (D*num_layers, batch_size, hidden_size) for batched input
     */
    [[nodiscard]] auto forward(const Tensor &input, const InitState &init_state) const -> Output;
    [[nodiscard]] auto forward(const Tensor &input) const -> Output;

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "LSTM";
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
    [[nodiscard]] auto forward(const Tensor &input, const std::optional<InitState> &init_state) const -> Output;
    int input_size_;
    int hidden_size_;
    LSTMOptions options_;
};

}    // namespace tinytensor::nn

#endif    // TINYTENSOR_NN_LSTM_H_
