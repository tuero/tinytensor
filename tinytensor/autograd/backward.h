// backward.h
// Backward pass functionality

#ifndef TINYTENSOR_AUTOGRAD_BACKARD_H_
#define TINYTENSOR_AUTOGRAD_BACKARD_H_

#include <tt/tensor.h>

namespace tinytensor::autograd {

void backward(Tensor &tensor, const Tensor &grad, bool retain_graph);

}    // namespace tinytensor::autograd

#endif    // TINYTENSOR_AUTOGRAD_BACKARD_H_
