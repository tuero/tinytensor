// index.hpp
// Indexing kernels

#ifndef TINYTENSOR_BACKEND_CPU_KERNEL_INDEX_H_
#define TINYTENSOR_BACKEND_CPU_KERNEL_INDEX_H_

#include <tt/scalar.h>

#include "tensor/backend/common/span.h"
#include "tensor/backend/common/util.h"
#include "tensor/backend/cpu/data_types.h"

#include <cassert>
#include <cstdint>

namespace tinytensor::cpu::kernel::index {

// Insert masked input items into res
template <typename T>
void index_mask_kernel(
    const DataInfo<const T> input_info,
    const DataInfo<const uint8_t> mask_info,
    DataInfo<T> res_info,
    int N
) {
    int res_counter = 0;
    for (int i = 0; i < N; ++i) {
        int mask_idx = to_flat_index(i, mask_info.shape, mask_info.stride, mask_info.offset);
        if (mask_info.data[mask_idx] > 0) {
            int input_idx = to_flat_index(i, input_info.shape, input_info.stride, input_info.offset);
            int res_idx = to_flat_index(res_counter, res_info.shape, res_info.stride, res_info.offset);
            res_info.data[res_idx] = input_info.data[input_idx];
            ++res_counter;
        }
    }
}

// Insert indices of input into res
template <typename T>
void index_indices_kernel(
    const DataInfo<const T> input_info,
    const DataInfo<const to_ctype_t<kDefaultInt>> indices_info,
    DataInfo<T> res_info,
    int N
) {
    for (int i = 0; i < N; ++i) {
        int indices_idx = to_flat_index(i, indices_info.shape, indices_info.stride, indices_info.offset);
        auto idx = indices_info.data[indices_idx];
        int input_idx = to_flat_index(idx, input_info.shape, input_info.stride, input_info.offset);
        int res_idx = to_flat_index(i, res_info.shape, res_info.stride, res_info.offset);
        res_info.data[res_idx] = input_info.data[input_idx];
    }
}

// Set masked input items using values
template <typename T>
void index_put_mask_kernel(
    DataInfo<T> input_info,
    const DataInfo<const T> values_info,
    const DataInfo<const uint8_t> mask_info,
    int N
) {
    int res_counter = 0;
    for (int i = 0; i < N; ++i) {
        int mask_idx = to_flat_index(i, mask_info.shape, mask_info.stride, mask_info.offset);
        if (mask_info.data[mask_idx] > 0) {
            int input_idx = to_flat_index(i, input_info.shape, input_info.stride, input_info.offset);
            int values_idx = to_flat_index(res_counter, values_info.shape, values_info.stride, values_info.offset);
            input_info.data[input_idx] = values_info.data[values_idx];
            ++res_counter;
        }
    }
}

// Set indices of input using values
template <typename T>
void index_put_indices_kernel(
    DataInfo<T> input_info,
    const DataInfo<const T> values_info,
    const DataInfo<const to_ctype_t<kDefaultInt>> indices_info,
    int N
) {
    for (int i = 0; i < N; ++i) {
        int indices_idx = to_flat_index(i, indices_info.shape, indices_info.stride, indices_info.offset);
        int values_idx = to_flat_index(i, values_info.shape, values_info.stride, values_info.offset);
        auto idx = indices_info.data[indices_idx];
        int input_idx = to_flat_index(idx, input_info.shape, input_info.stride, input_info.offset);
        input_info.data[input_idx] = values_info.data[values_idx];
    }
}

}    // namespace tinytensor::cpu::kernel::index

#endif    // TINYTENSOR_BACKEND_CPU_INDEX_H_
