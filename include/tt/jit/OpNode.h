//
// Created by ishaan on 11/27/25.
//

#pragma once
#include <tt/jit/Ops.h>
#include <memory>
#include <vector>

namespace tinytensor::jit {

class OpNode {
public:
    using Ptr = std::shared_ptr<OpNode>;

    OpNode(OpType op, std::vector<Ptr> inputs)
        : op_(std::move(op)), inputs_(std::move(inputs)) {}  // initializer lists in constructor

    [[nodiscard]] const OpType& op() const { return op_; } // getter for op_
    [[nodiscard]] const std::vector<Ptr>& inputs() const { return inputs_; } // getter for inputs_

private:
    OpType op_; // the actual variant of the operation we are dealing with
    std::vector<Ptr> inputs_; // the inputs to said operation
};

} // namespace tinytensor::jit
