// gru.cpp
// GRU layer module

#include <tt/device.h>
#include <tt/exception.h>
#include <tt/index.h>
#include <tt/nn/gru.h>
#include <tt/nn/init.h>
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
GRU::GRU(int input_size, int hidden_size, const GRUOptions &options, ScalarType dtype, Device device)
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
            init_w({i == 0 ? input_size : (options_.bidirectional ? 2 : 1) * hidden_size, 3 * hidden_size}, true)
        );
        weights_hh.push_back(init_w({hidden_size, 4 * hidden_size}, true));
        biases_ih.push_back(init_w({3 * hidden_size}, options.bias));
        biases_hh.push_back(init_w({3 * hidden_size}, options.bias));
        if (options_.bidirectional) {
            weights_ih_reverse.push_back(init_w({i == 0 ? input_size : 2 * hidden_size, 3 * hidden_size}, true));
            weights_hh_reverse.push_back(init_w({hidden_size, 3 * hidden_size}, true));
            biases_ih_reverse.push_back(init_w({3 * hidden_size}, options.bias));
            biases_hh_reverse.push_back(init_w({3 * hidden_size}, options.bias));
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

namespace {
struct SortIdx {
    int i;    // Global index
    int b;    // Batch item index
    int t;    // Sequence idx
    int d;    // 0 for forward, 1 for backward
};
}    // namespace

// Input = (L, H_in) unbatched, (L,N,H_in) if batch_first=false, (N,L,H_in) for batch_first=true
auto GRU::forward(const Tensor &_input, const std::optional<Tensor> &_h) const -> Output {
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
    if (_h.has_value()) {
        Tensor h = _h.value();
        if (h.dim() != 2 && h.dim() != 3) {
            TT_EXCEPTION(std::format("Expected h to have 2 or 3 dimensions, given h of shape {:s}", h.shape()));
        }
        if (is_batched && h.dim() == 2) {
            TT_EXCEPTION("Given batched input but unbatched h");
        }
        h = (h.dim() == 2) ? h.unsqueeze(1) : h;
        if (h.size(2) != hidden_size_) {
            TT_EXCEPTION(
                std::format(
                    "Expected given h of shape {:s} to have hidden_size of {:d}, given {:d}",
                    h.shape(),
                    hidden_size_,
                    h.size(2)
                )
            );
        }
        int expected_size = options_.num_layers * (options_.bidirectional ? 2 : 1);
        if (h.size(0) != expected_size) {
            TT_EXCEPTION(
                std::format(
                    "Expected given h of shape {:s} to have D*num_layers of {:d}, given {:d}",
                    h.shape(),
                    expected_size,
                    h.size(0)
                )
            );
        }
        // Inset into h_0
        for (int i : std::views::iota(0, h.size(0))) {
            h_0[{options_.bidirectional ? i % 2 : 0, options_.bidirectional ? i / 2 : i}] = h[i];
        }
    }

    TensorList h_output;

    for (int layer : std::views::iota(0, options_.num_layers)) {
        std::vector<SortIdx> idx_data;
        TensorList outputs;

        // Extract weights from packed
        Tensor w_ir = (*weights_ih[layer])[{indexing::Slice(), indexing::Slice(0 * hidden_size_, 1 * hidden_size_)}];
        Tensor w_iz = (*weights_ih[layer])[{indexing::Slice(), indexing::Slice(1 * hidden_size_, 2 * hidden_size_)}];
        Tensor w_in = (*weights_ih[layer])[{indexing::Slice(), indexing::Slice(2 * hidden_size_, 3 * hidden_size_)}];

        Tensor w_hr = (*weights_hh[layer])[{indexing::Slice(), indexing::Slice(0 * hidden_size_, 1 * hidden_size_)}];
        Tensor w_hz = (*weights_hh[layer])[{indexing::Slice(), indexing::Slice(1 * hidden_size_, 2 * hidden_size_)}];
        Tensor w_hn = (*weights_hh[layer])[{indexing::Slice(), indexing::Slice(2 * hidden_size_, 3 * hidden_size_)}];

        Tensor b_ir =
            (*biases_ih[layer])[indexing::Slice(0 * hidden_size_, 1 * hidden_size_)].expand({batch_size, hidden_size_});
        Tensor b_iz =
            (*biases_ih[layer])[indexing::Slice(1 * hidden_size_, 2 * hidden_size_)].expand({batch_size, hidden_size_});
        Tensor b_in =
            (*biases_ih[layer])[indexing::Slice(2 * hidden_size_, 3 * hidden_size_)].expand({batch_size, hidden_size_});

        Tensor b_hr =
            (*biases_hh[layer])[indexing::Slice(0 * hidden_size_, 1 * hidden_size_)].expand({batch_size, hidden_size_});
        Tensor b_hz =
            (*biases_hh[layer])[indexing::Slice(1 * hidden_size_, 2 * hidden_size_)].expand({batch_size, hidden_size_});
        Tensor b_hn =
            (*biases_hh[layer])[indexing::Slice(2 * hidden_size_, 3 * hidden_size_)].expand({batch_size, hidden_size_});

        // Forward traversal
        Tensor h_t = h_0[{0, layer}];
        for (int t : std::views::iota(0, seq_len)) {
            Tensor r_t = sigmoid(matmul(input[t], w_ir) + b_ir + matmul(h_t, w_hr) + b_hr);
            Tensor z_t = sigmoid(matmul(input[t], w_iz) + b_iz + matmul(h_t, w_hz) + b_hz);
            Tensor n_t = tanh(matmul(input[t], w_in) + b_in + r_t * (matmul(h_t, w_hn) + b_hn));
            h_t = (1 - z_t) * n_t + (z_t * h_t);
            for (int b : std::views::iota(0, batch_size)) {
                idx_data.push_back(SortIdx{.i = static_cast<int>(idx_data.size()), .b = b, .t = t, .d = 0});
            }
            outputs.push_back(h_t);
        }
        h_output.push_back(h_t);

        // Backward traversal
        if (options_.bidirectional) {
            // Extract weights from packed
            Tensor w_ir_r =
                (*weights_ih_reverse[layer])[{indexing::Slice(), indexing::Slice(0 * hidden_size_, 1 * hidden_size_)}];
            Tensor w_iz_r =
                (*weights_ih_reverse[layer])[{indexing::Slice(), indexing::Slice(1 * hidden_size_, 2 * hidden_size_)}];
            Tensor w_in_r =
                (*weights_ih_reverse[layer])[{indexing::Slice(), indexing::Slice(2 * hidden_size_, 3 * hidden_size_)}];

            Tensor w_hr_r =
                (*weights_hh_reverse[layer])[{indexing::Slice(), indexing::Slice(0 * hidden_size_, 1 * hidden_size_)}];
            Tensor w_hz_r =
                (*weights_hh_reverse[layer])[{indexing::Slice(), indexing::Slice(1 * hidden_size_, 2 * hidden_size_)}];
            Tensor w_hn_r =
                (*weights_hh_reverse[layer])[{indexing::Slice(), indexing::Slice(2 * hidden_size_, 3 * hidden_size_)}];

            Tensor b_ir_r = (*biases_ih_reverse[layer])[indexing::Slice(0 * hidden_size_, 1 * hidden_size_)].expand(
                {batch_size, hidden_size_}
            );
            Tensor b_iz_r = (*biases_ih_reverse[layer])[indexing::Slice(1 * hidden_size_, 2 * hidden_size_)].expand(
                {batch_size, hidden_size_}
            );
            Tensor b_in_r = (*biases_ih_reverse[layer])[indexing::Slice(2 * hidden_size_, 3 * hidden_size_)].expand(
                {batch_size, hidden_size_}
            );

            Tensor b_hr_r = (*biases_hh_reverse[layer])[indexing::Slice(0 * hidden_size_, 1 * hidden_size_)].expand(
                {batch_size, hidden_size_}
            );
            Tensor b_hz_r = (*biases_hh_reverse[layer])[indexing::Slice(1 * hidden_size_, 2 * hidden_size_)].expand(
                {batch_size, hidden_size_}
            );
            Tensor b_hn_r = (*biases_hh_reverse[layer])[indexing::Slice(2 * hidden_size_, 3 * hidden_size_)].expand(
                {batch_size, hidden_size_}
            );

            Tensor h_t_r = h_0[{1, layer}];
            for (int t : std::views::iota(0, seq_len) | std::views::reverse) {
                Tensor r_t_r = sigmoid(matmul(input[t], w_ir_r) + b_ir_r + matmul(h_t_r, w_hr_r) + b_hr_r);
                Tensor z_t_r = sigmoid(matmul(input[t], w_iz_r) + b_iz_r + matmul(h_t_r, w_hz_r) + b_hz_r);
                Tensor n_t_r = tanh(matmul(input[t], w_in_r) + b_in_r + r_t_r * (matmul(h_t_r, w_hn_r) + b_hn_r));
                h_t_r = (1 - z_t_r) * n_t_r + (z_t_r * h_t_r);
                for (int b : std::views::iota(0, batch_size)) {
                    idx_data.push_back(SortIdx{.i = static_cast<int>(idx_data.size()), .b = b, .t = t, .d = 1});
                }
                outputs.push_back(h_t_r);
            }
            h_output.push_back(h_t_r);
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
    if (!is_batched) {
        output = output.squeeze(1);
        h_out = h_out.squeeze(1);
    }
    return {.output = output, .h = h_out};
}

void GRU::pretty_print(std::ostream &os) const {
    os << std::format(
        "GRU(in_features={:d}, hidden_size={:d}, num_layers={:d}, bias={:}, bidirectional={:}, dtype={}, device={})",
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
