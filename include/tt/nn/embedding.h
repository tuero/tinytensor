// embedding.h
// Embedding layer module

#ifndef TINYTENSOR_NN_EMBEDDING_H_
#define TINYTENSOR_NN_EMBEDDING_H_

#include <tt/device.h>
#include <tt/export.h>
#include <tt/nn/module.h>
#include <tt/scalar.h>
#include <tt/tensor.h>

#include <memory>
#include <ostream>
#include <string>

namespace tinytensor::nn {

// An embedding layer
class TINYTENSOR_EXPORT Embedding : public Module {
public:
    /**
     * Construct a linear layer
     * @param num_embeddings Number of embeddings
     * @param embedding_dim Dimension for each embedding vector
     * @param dtype The dtype of the weights
     * @param device The device the weights should be initialized on
     */
    Embedding(int num_embeddings, int embedding_dim, ScalarType dtype = kDefaultFloat, Device device = kCPU);

    /**
     * Forward pass for Embedding layer
     * @param input The input tensor, of integral type
     * @return Output tensor
     */
    [[nodiscard]] auto forward(const Tensor &input) const -> Tensor;

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "Embedding";
    }

    std::shared_ptr<Tensor> weight;

private:
    int num_embeddings_;
    int embedding_dim_;
};

}    // namespace tinytensor::nn

#endif    // TINYTENSOR_NN_EMBEDDING_H_
