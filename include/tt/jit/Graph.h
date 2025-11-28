//
// Created by ishaan on 11/27/25.
//
#pragma once
#include <tt/jit/OpNode.h>

#include <memory>
#include <vector>

namespace tinytensor::jit {

// simple factory/manager for nodes
class JITGraph {
public:
    using NodePtr = std::shared_ptr<OpNode>;

    NodePtr create_node(OpType op, std::vector<NodePtr> inputs) {
        auto node = std::make_shared<OpNode>(std::move(op), std::move(inputs));
        // could keep a list of all_nodes_ here if i wanted to debug
        return node;
    }
};

// singleton to access the builder
inline JITGraph& GetGlobalGraph() {
    static JITGraph g;
    return g;
}

} // namespace tinytensor::jit
