// dag.h
// Autograd DAG for correct gradient computation ordering

#ifndef TINYTENSOR_AUTOGRAD_DAG_H_
#define TINYTENSOR_AUTOGRAD_DAG_H_

#include <tt/tensor.h>

namespace tinytensor::autograd {

/**
 * Build a DAG in topological sotred order using the autograd context parent relationships
 * @param tensor The tensor represnting the base of the DAG
 * @return DAG in topological sorted order rooted in the given tensor
 */
auto build_dag(const Tensor &tensor) -> TensorList;

}    // namespace tinytensor::autograd

#endif    // TINYTENSOR_AUTOGRAD_DAG_H_
