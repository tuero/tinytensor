// datatypes.h
// Helper data types for kernels

#ifndef TINYTENSOR_BACKEND_CPU_DATATYPES_H_
#define TINYTENSOR_BACKEND_CPU_DATATYPES_H_

#include <tt/scalar.h>
#include <tt/shape.h>

#include "tensor/backend/common/span.h"

#include <type_traits>

namespace tinytensor::cpu {

// Device data + offsets to access elements from shared data
template <typename T>
    requires IsScalarType<std::remove_cvref_t<T>>
struct DataInfo {
    HostSpan<T> data;
    HostSpan<const int> shape{};
    HostSpan<const int> stride{};
    int offset = 0;
};

}    // namespace tinytensor::cpu

#endif    // TINYTENSOR_BACKEND_CPU_DATATYPES_H_
