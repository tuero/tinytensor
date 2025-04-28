// lstm.cpp
// LSTM layer module

#include <tt/device.h>
#include <tt/exception.h>
#include <tt/index.h>
#include <tt/nn/init.h>
#include <tt/nn/lstm.h>
#include <tt/scalar.h>
#include <tt/shape.h>
#include <tt/tensor.h>
#include <tt/util.h>

#include <algorithm>
#include <cmath>
#include <format>
#include <memory>
#include <optional>
#include <ostream>
#include <ranges>
#include <string>
#include <tuple>
#include <vector>

namespace tinytensor::nn {

// Assert all hidden sizes to > 0
LSTM::LSTM(int input_size, int hidden_size, const LSTMOptions &options, ScalarType dtype, Device device)
    : input_size_(input_size), hidden_size_(hidden_size), options_(options) {
    if (input_size <= 0) {
        TT_EXCEPTION(std::format("Expected input_size > 0, given input_size={:d}", input_size));
    }
    if (hidden_size <= 0) {
        TT_EXCEPTION(std::format("Expected hidden_size > 0, given hidden_size={:d}", hidden_size));
    }
    if (options.num_layers <= 0) {
        TT_EXCEPTION(std::format("Expected num_layers > 0, given num_layers={:d}", options.num_layers));
    }
    double stdev = 1.0 / std::sqrt(hidden_size);
    const auto init_w = [&](const Shape &dims, bool requires_grad) -> std::shared_ptr<Tensor> {
        auto w = std::make_shared<Tensor>(zeros(dims, dtype, device, requires_grad));
        nn::uniform_(*w, -stdev, stdev);
        return w;
    };
    for (const auto i : std::views::iota(0, options.num_layers)) {
        weights_ih.push_back(
            init_w({i == 0 ? input_size : (options_.bidirectional ? 2 : 1) * hidden_size, 4 * hidden_size}, true)
        );
        weights_hh.push_back(init_w({hidden_size, 4 * hidden_size}, true));
        biases_ih.push_back(init_w({4 * hidden_size}, options.bias));
        biases_hh.push_back(init_w({4 * hidden_size}, options.bias));
        if (options_.bidirectional) {
            weights_ih_reverse.push_back(init_w({i == 0 ? input_size : 2 * hidden_size, 4 * hidden_size}, true));
            weights_hh_reverse.push_back(init_w({hidden_size, 4 * hidden_size}, true));
            biases_ih_reverse.push_back(init_w({4 * hidden_size}, options.bias));
            biases_hh_reverse.push_back(init_w({4 * hidden_size}, options.bias));
        }
    }

    // Register params
    for (auto &w : weights_ih) {
        register_param(w);
    }
    for (auto &w : weights_hh) {
        register_param(w);
    }
    for (auto &w : biases_ih) {
        register_param(w);
    }
    for (auto &w : biases_hh) {
        register_param(w);
    }
    for (auto &w : weights_ih_reverse) {
        register_param(w);
    }
    for (auto &w : weights_hh_reverse) {
        register_param(w);
    }
    for (auto &w : biases_ih_reverse) {
        register_param(w);
    }
    for (auto &w : biases_hh_reverse) {
        register_param(w);
    }
}

auto LSTM::forward(const Tensor &input, const InitState &init_state) const -> LSTM::Output {
    return forward(input, std::optional<InitState>{init_state});
}
auto LSTM::forward(const Tensor &input) const -> LSTM::Output {
    return forward(input, std::nullopt);
}

namespace {
struct SortIdx {
    int i;    // Global index
    int b;    // Batch item index
    int t;    // Sequence idx
    int d;    // 0 for forward, 1 for backward
};
}    // namespace

// Input = (L, H_in) unbatched, (L,N,H_in) if batch_first=false, (N,L,H_in) for batch_first=true
auto LSTM::forward(const Tensor &_input, const std::optional<InitState> &_init_state) const -> Output {
    if (_input.dim() != 2 && _input.dim() != 3) {
        TT_EXCEPTION(
            std::format("Expected input to have 2 or 3 dimensions, given input of shape {:s}", _input.shape())
        );
    }

    // Check if batched
    bool is_batched = _input.dim() == 3;
    Tensor input = is_batched ? _input : _input.unsqueeze(1);

    // Always work in (L,N,H_in) mode
    if (options_.batch_first) {
        input = input.permute({1, 0, 2});
    }

    if (input.size(2) != input_size_) {
        TT_EXCEPTION(
            std::format(
                "Expected input of shape {:s} to have input_size of {:d}, given {:d}",
                _input.shape(),
                input_size_,
                _input.size(2)
            )
        );
    }

    int seq_len = input.size(0);
    int batch_size = input.size(1);

    // Starting hidden
    Tensor h_0 = zeros(
        {options_.bidirectional ? 2 : 1, options_.num_layers, batch_size, hidden_size_},
        TensorOptions().dtype(input.dtype()).device(input.device())

    );
    Tensor c_0 = zeros(
        {options_.bidirectional ? 2 : 1, options_.num_layers, batch_size, hidden_size_},
        TensorOptions().dtype(input.dtype()).device(input.device())

    );
    if (_init_state.has_value()) {
        // Check args + get into correct shape
        const auto check_init_state = [&](Tensor init_state, const std::string &init_state_name) -> Tensor {
            if (init_state.dim() != 2 && init_state.dim() != 3) {
                TT_EXCEPTION(
                    std::format(
                        "Expected {0:s} to have 2 or 3 dimensions, given {0:s} of shape {1:s}",
                        init_state_name,
                        init_state.shape()
                    )
                );
            }
            if (is_batched && init_state.dim() == 2) {
                TT_EXCEPTION(std::format("Given batched input but unbatched {:s}", init_state_name));
            }
            init_state = (init_state.dim() == 2) ? init_state.unsqueeze(1) : init_state;
            if (init_state.size(2) != hidden_size_) {
                TT_EXCEPTION(
                    std::format(
                        "Expected given {:s} of shape {:s} to have hidden_size of {:d}, given {:d}",
                        init_state_name,
                        init_state.shape(),
                        hidden_size_,
                        init_state.size(2)
                    )
                );
            }
            int expected_size = options_.num_layers * (options_.bidirectional ? 2 : 1);
            if (init_state.size(0) != expected_size) {
                TT_EXCEPTION(
                    std::format(
                        "Expected given {:s} of shape {:s} to have D*num_layers of {:d}, given {:d}",
                        init_state_name,
                        init_state.shape(),
                        expected_size,
                        init_state.size(0)
                    )
                );
            }
            return init_state;
        };

        auto [h, c] = _init_state.value();
        h = check_init_state(h, "h");
        c = check_init_state(c, "c");

        // Inset into h_0 and c_0
        for (int i : std::views::iota(0, h.size(0))) {
            h_0[{options_.bidirectional ? i % 2 : 0, options_.bidirectional ? i / 2 : i}] = h[i];
        }
        for (int i : std::views::iota(0, c.size(0))) {
            c_0[{options_.bidirectional ? i % 2 : 0, options_.bidirectional ? i / 2 : i}] = c[i];
        }
    }

    TensorList h_output;
    TensorList c_output;

    for (int layer : std::views::iota(0, options_.num_layers)) {
        std::vector<SortIdx> idx_data;
        TensorList outputs;

        // Extract weights from packed
        Tensor w_ii = (*weights_ih[layer])[{indexing::Slice(), indexing::Slice(0 * hidden_size_, 1 * hidden_size_)}];
        Tensor w_if = (*weights_ih[layer])[{indexing::Slice(), indexing::Slice(1 * hidden_size_, 2 * hidden_size_)}];
        Tensor w_ig = (*weights_ih[layer])[{indexing::Slice(), indexing::Slice(2 * hidden_size_, 3 * hidden_size_)}];
        Tensor w_io = (*weights_ih[layer])[{indexing::Slice(), indexing::Slice(3 * hidden_size_, 4 * hidden_size_)}];

        Tensor w_hi = (*weights_hh[layer])[{indexing::Slice(), indexing::Slice(0 * hidden_size_, 1 * hidden_size_)}];
        Tensor w_hf = (*weights_hh[layer])[{indexing::Slice(), indexing::Slice(1 * hidden_size_, 2 * hidden_size_)}];
        Tensor w_hg = (*weights_hh[layer])[{indexing::Slice(), indexing::Slice(2 * hidden_size_, 3 * hidden_size_)}];
        Tensor w_ho = (*weights_hh[layer])[{indexing::Slice(), indexing::Slice(3 * hidden_size_, 4 * hidden_size_)}];

        Tensor b_ii =
            (*biases_ih[layer])[indexing::Slice(0 * hidden_size_, 1 * hidden_size_)].expand({batch_size, hidden_size_});
        Tensor b_if =
            (*biases_ih[layer])[indexing::Slice(1 * hidden_size_, 2 * hidden_size_)].expand({batch_size, hidden_size_});
        Tensor b_ig =
            (*biases_ih[layer])[indexing::Slice(2 * hidden_size_, 3 * hidden_size_)].expand({batch_size, hidden_size_});
        Tensor b_io =
            (*biases_ih[layer])[indexing::Slice(3 * hidden_size_, 4 * hidden_size_)].expand({batch_size, hidden_size_});

        Tensor b_hi =
            (*biases_hh[layer])[indexing::Slice(0 * hidden_size_, 1 * hidden_size_)].expand({batch_size, hidden_size_});
        Tensor b_hf =
            (*biases_hh[layer])[indexing::Slice(1 * hidden_size_, 2 * hidden_size_)].expand({batch_size, hidden_size_});
        Tensor b_hg =
            (*biases_hh[layer])[indexing::Slice(2 * hidden_size_, 3 * hidden_size_)].expand({batch_size, hidden_size_});
        Tensor b_ho =
            (*biases_hh[layer])[indexing::Slice(3 * hidden_size_, 4 * hidden_size_)].expand({batch_size, hidden_size_});

        // Forward traversal
        Tensor h_t = h_0[{0, layer}];
        Tensor c_t = c_0[{0, layer}];
        for (int t : std::views::iota(0, seq_len)) {
            Tensor i_t = sigmoid(matmul(input[t], w_ii) + b_ii + matmul(h_t, w_hi) + b_hi);
            Tensor f_t = sigmoid(matmul(input[t], w_if) + b_if + matmul(h_t, w_hf) + b_hf);
            Tensor g_t = tanh(matmul(input[t], w_ig) + b_ig + matmul(h_t, w_hg) + b_hg);
            Tensor o_t = sigmoid(matmul(input[t], w_io) + b_io + matmul(h_t, w_ho) + b_ho);
            c_t = f_t * c_t + i_t * g_t;
            h_t = o_t * tanh(c_t);
            for (int b : std::views::iota(0, batch_size)) {
                idx_data.push_back(SortIdx{.i = static_cast<int>(idx_data.size()), .b = b, .t = t, .d = 0});
            }
            outputs.push_back(h_t);
        }
        h_output.push_back(h_t);
        c_output.push_back(c_t);

        // Backward traversal
        if (options_.bidirectional) {
            // Extract weights from packed
            Tensor w_ii_r =
                (*weights_ih_reverse[layer])[{indexing::Slice(), indexing::Slice(0 * hidden_size_, 1 * hidden_size_)}];
            Tensor w_if_r =
                (*weights_ih_reverse[layer])[{indexing::Slice(), indexing::Slice(1 * hidden_size_, 2 * hidden_size_)}];
            Tensor w_ig_r =
                (*weights_ih_reverse[layer])[{indexing::Slice(), indexing::Slice(2 * hidden_size_, 3 * hidden_size_)}];
            Tensor w_io_r =
                (*weights_ih_reverse[layer])[{indexing::Slice(), indexing::Slice(3 * hidden_size_, 4 * hidden_size_)}];

            Tensor w_hi_r =
                (*weights_hh_reverse[layer])[{indexing::Slice(), indexing::Slice(0 * hidden_size_, 1 * hidden_size_)}];
            Tensor w_hf_r =
                (*weights_hh_reverse[layer])[{indexing::Slice(), indexing::Slice(1 * hidden_size_, 2 * hidden_size_)}];
            Tensor w_hg_r =
                (*weights_hh_reverse[layer])[{indexing::Slice(), indexing::Slice(2 * hidden_size_, 3 * hidden_size_)}];
            Tensor w_ho_r =
                (*weights_hh_reverse[layer])[{indexing::Slice(), indexing::Slice(3 * hidden_size_, 4 * hidden_size_)}];

            Tensor b_ii_r = (*biases_ih_reverse[layer])[indexing::Slice(0 * hidden_size_, 1 * hidden_size_)].expand(
                {batch_size, hidden_size_}
            );
            Tensor b_if_r = (*biases_ih_reverse[layer])[indexing::Slice(1 * hidden_size_, 2 * hidden_size_)].expand(
                {batch_size, hidden_size_}
            );
            Tensor b_ig_r = (*biases_ih_reverse[layer])[indexing::Slice(2 * hidden_size_, 3 * hidden_size_)].expand(
                {batch_size, hidden_size_}
            );
            Tensor b_io_r = (*biases_ih_reverse[layer])[indexing::Slice(3 * hidden_size_, 4 * hidden_size_)].expand(
                {batch_size, hidden_size_}
            );

            Tensor b_hi_r = (*biases_hh_reverse[layer])[indexing::Slice(0 * hidden_size_, 1 * hidden_size_)].expand(
                {batch_size, hidden_size_}
            );
            Tensor b_hf_r = (*biases_hh_reverse[layer])[indexing::Slice(1 * hidden_size_, 2 * hidden_size_)].expand(
                {batch_size, hidden_size_}
            );
            Tensor b_hg_r = (*biases_hh_reverse[layer])[indexing::Slice(2 * hidden_size_, 3 * hidden_size_)].expand(
                {batch_size, hidden_size_}
            );
            Tensor b_ho_r = (*biases_hh_reverse[layer])[indexing::Slice(3 * hidden_size_, 4 * hidden_size_)].expand(
                {batch_size, hidden_size_}
            );

            Tensor h_t_r = h_0[{1, layer}];
            Tensor c_t_r = c_0[{1, layer}];
            for (int t : std::views::iota(0, seq_len) | std::views::reverse) {
                Tensor i_t_r = sigmoid(matmul(input[t], w_ii_r) + b_ii_r + matmul(h_t_r, w_hi_r) + b_hi_r);
                Tensor f_t_r = sigmoid(matmul(input[t], w_if_r) + b_if_r + matmul(h_t_r, w_hf_r) + b_hf_r);
                Tensor g_t_r = tanh(matmul(input[t], w_ig_r) + b_ig_r + matmul(h_t_r, w_hg_r) + b_hg_r);
                Tensor o_t_r = sigmoid(matmul(input[t], w_io_r) + b_io_r + matmul(h_t_r, w_ho_r) + b_ho_r);
                c_t_r = f_t_r * c_t_r + i_t_r * g_t_r;
                h_t_r = o_t_r * tanh(c_t_r);
                for (int b : std::views::iota(0, batch_size)) {
                    idx_data.push_back(SortIdx{.i = static_cast<int>(idx_data.size()), .b = b, .t = t, .d = 1});
                }
                outputs.push_back(h_t_r);
            }
            h_output.push_back(h_t_r);
            c_output.push_back(c_t_r);
        }
        // Hack to get final output in correct order
        // We can't insert directly into output because in-place on r-value reference for autograd not supported
        std::ranges::sort(idx_data, [](const SortIdx &lhs, const SortIdx &rhs) -> bool {
            return std::tie(lhs.t, lhs.b, lhs.d) < std::tie(rhs.t, rhs.b, rhs.d);
        });
        std::vector<int> indices = idx_data | std::views::transform([](const SortIdx &idx) -> int { return idx.i; })
                                   | tinytensor::to<std::vector<int>>();

        Tensor output = index_select(cat(outputs, 0), indices, 0)
                            .reshape({seq_len, batch_size, (options_.bidirectional ? 2 : 1) * hidden_size_});
        input = output;
    }
    Tensor output = input;

    // Get back to expected shape, whether batched or not
    if (options_.batch_first) {
        output = output.permute({1, 0, 2});
    }
    Tensor h_out = stack(h_output, 0);
    Tensor c_out = stack(c_output, 0);
    if (!is_batched) {
        output = output.squeeze(1);
        h_out = h_out.squeeze(1);
        c_out = c_out.squeeze(1);
    }
    return {.output = output, .h = h_out, .c = c_out};
}

void LSTM::pretty_print(std::ostream &os) const {
    os << std::format(
        "LSTM(in_features={:d}, hidden_size={:d}, num_layers={:d}, bias={:}, bidirectional={:}, dtype={}, device={})",
        input_size_,
        hidden_size_,
        options_.num_layers,
        options_.bias,
        options_.bidirectional,
        weights_hh[0]->dtype(),
        weights_hh[0]->device()
    );
}

}    // namespace tinytensor::nn
