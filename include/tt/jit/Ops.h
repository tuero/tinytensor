//
// Created by ishaan on 11/27/25.
//


#pragma once
#include <variant>
#include <vector>
#include <cstdint>

namespace tinytensor::jit {

// leaf nodes
// Represents raw data coming into the graph (e.g. from CPU)
struct InputOp {
    // ee might store a unique id or metadata here later
    int id;
};

// define various ops with different configs
// Relu, Add etc... dont need any more information than what already exists, there may be ops, such as reduceSum
// which will need extra info, their structs will therefore look different
struct ReluOp {};
struct AddOp {};

struct BroadcastOp {
    Shape target_shape;
};

using OpType = std::variant<
    InputOp,
    ReluOp,
    AddOp,
    BroadcastOp
>;

} // namespace tinytensor::jit
