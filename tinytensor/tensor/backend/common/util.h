// util.h
// Utility indexing for inside kernels

#ifndef TINYTENSOR_BACKEND_COMMON_UTIL_H_
#define TINYTENSOR_BACKEND_COMMON_UTIL_H_

#include <tt/concepts.h>
#include <tt/macros.h>
#include <tt/shape.h>

#include <cstddef>

namespace tinytensor {

// Converts an array rank (local flat index) to a global index given a shape, stride, and offset
// If given an dim, its treated as a deduction and we skip that dim
template <typename T>
constexpr TT_DEVICE auto to_flat_index(int rank, const T &shape, const T &stride, int offset, int dim = -1) -> int {
    // Shortcut index computation if empty size/stride given (default constructed data info)
    if (shape.size() == 0 || stride.size() == 0) {
        return rank + offset;
    }
    // Single element arrays
    int flat_index = offset;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        const auto idx = static_cast<std::size_t>(i);
        if (dim != -1 && i == dim) {
            continue;
        }
        flat_index += (rank % shape[idx]) * stride[idx];
        rank /= shape[idx];
    }
    return flat_index;
}

}    // namespace tinytensor

#endif    // TINYTENSOR_BACKEND_COMMON_UTIL_H_
