// index.h
// Index runner

#ifndef TINYTENSOR_BACKEND_CPU_INDEX_H_
#define TINYTENSOR_BACKEND_CPU_INDEX_H_

#include <tt/tensor.h>

namespace tinytensor::cpu {

auto index_mask_runner(const Tensor &input, const Tensor &mask, int Nr) -> Tensor;

auto index_indices_runner(const Tensor &input, const Tensor &indices) -> Tensor;

void index_put_mask_runner(Tensor &input, const Tensor &values, const Tensor &mask);

void index_put_indices_runner(Tensor &input, const Tensor &values, const Tensor &indices);

}    // namespace tinytensor::cpu

#endif    // TINYTENSOR_BACKEND_CPU_INDEX_H_
