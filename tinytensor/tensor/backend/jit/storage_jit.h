//
// Created by ishaan on 11/27/25.
//

#pragma once
#include <tt/jit/OpNode.h>
#include <tensor/storage_base.h>
#include "tt/device.h"

namespace tinytensor {

class StorageJIT : public StorageBase {
public:
    StorageJIT(std::shared_ptr<jit::OpNode> node)
        : StorageBase(), node_(std::move(node)) {}

    [[nodiscard]] std::shared_ptr<jit::OpNode> get_node() const { return node_; }

private:
    std::shared_ptr<jit::OpNode> node_;
};

} // namespace tinytensor
