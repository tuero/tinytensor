// datatypes.cuh
// Helper data types for kernels

#ifndef TINYTENSOR_BACKEND_CUDA_DATATYPES_H_
#define TINYTENSOR_BACKEND_CUDA_DATATYPES_H_

#include <tt/exception.h>
#include <tt/scalar.h>
#include <tt/shape.h>
#include <tt/tensor.h>

#include "tensor/backend/common/span.h"

#include <cassert>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

namespace tinytensor::cuda {

template <typename T>
struct DeviceSpan : public NonOwningSpan<T> {
    using NonOwningSpan<T>::NonOwningSpan;
};

// Owning device memory which sits on host
template <typename T>
    requires(!std::is_const_v<T>)
class DeviceMemory {
    // Init buffer of size n
    DeviceMemory(std::size_t n);

    // Init buffer of size n and initialize with given value
    DeviceMemory(std::size_t n, T value);

    // Construct from host vector
    DeviceMemory(const std::vector<std::remove_cv_t<T>> &v);

public:
    DeviceMemory() = default;

    ~DeviceMemory();

    void swap(DeviceMemory<T> &other) noexcept {
        std::swap(_p, other._p);
        std::swap(_n, other._n);
    }

    // Can move construct, but doesn't make sense to copy/move assign
    DeviceMemory(const DeviceMemory &) = delete;
    DeviceMemory(DeviceMemory &&other)
        : _p(other._p), _n(other._n) {
        other._p = nullptr;
    };
    DeviceMemory &operator=(const DeviceMemory &) = delete;
    DeviceMemory &operator=(DeviceMemory &&) = default;

    [[nodiscard]] auto clone() const -> DeviceMemory;

    // Constructors
    [[nodiscard]] static auto AllocateElements(std::size_t n) -> DeviceMemory {
        return {n};
    }
    [[nodiscard]] static auto AllocateElements(std::size_t n, T value) -> DeviceMemory {
        return {n, value};
    }
    [[nodiscard]] static auto AllocateVec(const std::vector<std::remove_cv_t<T>> &v) -> DeviceMemory {
        return {v};
    }

    [[nodiscard]] static auto AllocateArange(std::size_t n) -> DeviceMemory;

    // Convert to host vector
    [[nodiscard]] auto to_vec() const -> std::vector<std::remove_cv_t<T>>;

    // Get a specific item
    [[nodiscard]] auto item(int index) const -> T;

    [[nodiscard]] constexpr auto size() const -> std::size_t {
        return _n;
    }

    [[nodiscard]] auto data_ptr() const -> T * {
        return _p;
    }

    // Cast operator
    operator DeviceSpan<T>() const {
        return DeviceSpan<T>{_p, _n};
    }
    operator DeviceSpan<const T>() const {
        return DeviceSpan<const T>{_p, _n};
    }

private:
    T *_p = nullptr;
    std::size_t _n = 0;
};

// Factory functions
template <typename T>
auto MakeDeviceMemory(const std::vector<std::remove_cv_t<T>> &v) -> DeviceMemory<T> {
    return DeviceMemory<T>::AllocateVec(v);
}

template <typename T>
auto MakeDeviceMemory(std::size_t n, T value) -> DeviceMemory<T> {
    return DeviceMemory<T>::AllocateElements(n, value);
}

inline auto MakeDeviceMemory(const Shape &shape) -> DeviceMemory<int> {
    return DeviceMemory<int>::AllocateVec(shape.to_vec());
}

// Device data + offsets to access elements from shared data
template <typename T>
    requires IsScalarType<std::remove_cvref_t<T>>
struct DataInfo {
    DeviceSpan<T> data;
    DeviceSpan<const int> shape{};
    DeviceSpan<const int> stride{};
    int offset = 0;
};

}    // namespace tinytensor::cuda

// Overload swap
namespace std {
template <typename T>
void swap(tinytensor::cuda::DeviceMemory<T> &lhs, tinytensor::cuda::DeviceMemory<T> &rhs) {
    lhs.swap(rhs);
}
}    // namespace std

#endif    // TINYTENSOR_BACKEND_CUDA_DATATYPES_H_
