#include "backend_jit.h"
#include "storage_jit.h"
#include <tt/jit/Graph.h>
#include <tt/jit/Ops.h>
#include <tt/shape.h>

namespace tinytensor {

// manually pass Shape and Dtype because StorageBase doesn't hold metadata.
Tensor make_jit_tensor(std::shared_ptr<jit::OpNode> node, Shape shape, ScalarType dtype) {
    // create the JIT storage holding the node
    auto storage = std::make_shared<StorageJIT>(std::move(node));

    // use the Tensor constructor designed for backends:
    // Tensor(std::shared_ptr<StorageBase> storage, ScalarType dtype, Shape shape, Device device);
    return Tensor(std::move(storage), dtype, shape, kJIT);
}

Tensor BackendJIT::relu(const Tensor &tensor) const {
    auto& in_jit = tensor.get_storage<StorageJIT>();

    auto node = jit::GetGlobalGraph().create_node(jit::ReluOp{}, {in_jit.get_node()});

    return make_jit_tensor(node, tensor.shape(), tensor.dtype());
}

Tensor BackendJIT::add(const Tensor &lhs, const Tensor &rhs) const {
    auto& lhs_jit = lhs.get_storage<StorageJIT>();
    auto& rhs_jit = rhs.get_storage<StorageJIT>();

    // assume shapes match or that the user has handled broadcasting explicitly, as per readme.md
    // (Future TODO: Handle explicit views where Tensor shape != OpNode shape)

    auto node = jit::GetGlobalGraph().create_node(
        jit::AddOp{},
        {lhs_jit.get_node(), rhs_jit.get_node()}
    );

    // take the LHS shape/dtype as the truth
    return make_jit_tensor(node, lhs.shape(), lhs.dtype());
}

} // namespace tinytensor