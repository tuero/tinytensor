// embedding.cpp
// Embedding layer module

#include <tt/device.h>
#include <tt/exception.h>
#include <tt/nn/embedding.h>
#include <tt/nn/init.h>
#include <tt/scalar.h>
#include <tt/tensor.h>

#include <cmath>
#include <format>
#include <memory>
#include <ostream>

namespace tinytensor::nn {

Embedding::Embedding(int num_embeddings, int embedding_dim, ScalarType dtype, Device device)
    : weight(std::make_shared<Tensor>(zeros({num_embeddings, embedding_dim}, dtype, device, true))),
      num_embeddings_(num_embeddings),
      embedding_dim_(embedding_dim) {
    if (num_embeddings <= 0) {
        TT_EXCEPTION(std::format("Expected num_embeddings > 0, given num_embeddings={:d}", num_embeddings));
    }
    if (embedding_dim <= 0) {
        TT_EXCEPTION(std::format("Expected embedding_dim > 0, given embedding_dim={:d}", embedding_dim));
    }
    // Weight initialized by Normal(0, 1)
    nn::normal_(*weight, 0, 1);
    register_param(weight);
}

auto Embedding::forward(const Tensor &input) const -> Tensor {
    return embedding(input, *weight);
}

void Embedding::pretty_print(std::ostream &os) const {
    os << std::format(
        "Embedding(num_embeddings={:d}, embedding_dim={:d}, dtype={}, device={})",
        num_embeddings_,
        embedding_dim_,
        weight->dtype(),
        weight->device()
    );
}

}    // namespace tinytensor::nn
