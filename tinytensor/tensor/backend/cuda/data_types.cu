// datatypes.cu
// Helper data types for kernels

#include <tt/exception.h>
#include <tt/scalar.h>

#include "tensor/backend/cuda/config.cuh"
#include "tensor/backend/cuda/data_types.cuh"
#include "tensor/backend/cuda/kernel/init.cuh"
#include "tensor/backend/cuda/kernel_launch.cuh"
#include "tensor/backend/cuda/storage_cuda.h"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#include <cstddef>
#include <format>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

namespace tinytensor::cuda {

using namespace kernel::init;

using kBoolCType = to_ctype_t<kBool>;
using kI16CType = to_ctype_t<kI16>;
using kI32CType = to_ctype_t<kI32>;
using kI64CType = to_ctype_t<kI64>;
using kF32CType = to_ctype_t<kF32>;
using kF64CType = to_ctype_t<kF64>;

// Init buffer of size n
template <typename T>
    requires(!std::is_const_v<T>)
DeviceMemory<T>::DeviceMemory(std::size_t n)
    : _n(n) {
    const auto malloc_status = cudaMalloc(&_p, _n * sizeof(T));
    if (malloc_status != cudaSuccess) {
        TT_EXCEPTION("cudaMalloc error: " + std::string(cudaGetErrorString(malloc_status)));
    }
    StorageCUDA::current_bytes_allocated += _n * sizeof(T);
    StorageCUDA::total_bytes_allocated += _n * sizeof(T);
}
template DeviceMemory<kBoolCType>::DeviceMemory(std::size_t n);
template DeviceMemory<kI16CType>::DeviceMemory(std::size_t n);
template DeviceMemory<kI32CType>::DeviceMemory(std::size_t n);
template DeviceMemory<kI64CType>::DeviceMemory(std::size_t n);
template DeviceMemory<kF32CType>::DeviceMemory(std::size_t n);
template DeviceMemory<kF64CType>::DeviceMemory(std::size_t n);
template DeviceMemory<uint64_t>::DeviceMemory(std::size_t n);

// Construct from host vector
template <typename T>
    requires(!std::is_const_v<T>)
DeviceMemory<T>::DeviceMemory(const std::vector<std::remove_cv_t<T>> &v)
    : DeviceMemory(v.size()) {
    const auto memcpy_status = cudaMemcpy(_p, v.data(), _n * sizeof(T), cudaMemcpyHostToDevice);
    if (memcpy_status != cudaSuccess) {
        TT_EXCEPTION("cudaMemcpy error: " + std::string(cudaGetErrorString(memcpy_status)));
    }
    StorageCUDA::current_bytes_allocated += _n * sizeof(T);
    StorageCUDA::total_bytes_allocated += _n * sizeof(T);
}
template DeviceMemory<kBoolCType>::DeviceMemory(const std::vector<kBoolCType> &v);
template DeviceMemory<kI16CType>::DeviceMemory(const std::vector<kI16CType> &v);
template DeviceMemory<kI32CType>::DeviceMemory(const std::vector<kI32CType> &v);
template DeviceMemory<kI64CType>::DeviceMemory(const std::vector<kI64CType> &v);
template DeviceMemory<kF32CType>::DeviceMemory(const std::vector<kF32CType> &v);
template DeviceMemory<kF64CType>::DeviceMemory(const std::vector<kF64CType> &v);
template DeviceMemory<uint64_t>::DeviceMemory(const std::vector<uint64_t> &v);

// Initialize from a value
template <typename T>
    requires(!std::is_const_v<T>)
DeviceMemory<T>::DeviceMemory(std::size_t n, T value)
    : DeviceMemory(n) {
    const auto kernel = init_full_kernel<T>;
    launch(kernel, grid_1d(n), block_1d(), _p, value, static_cast<int>(n));
}
template DeviceMemory<kBoolCType>::DeviceMemory(std::size_t n, kBoolCType value);
template DeviceMemory<kI16CType>::DeviceMemory(std::size_t n, kI16CType value);
template DeviceMemory<kI32CType>::DeviceMemory(std::size_t n, kI32CType value);
template DeviceMemory<kI64CType>::DeviceMemory(std::size_t n, kI64CType value);
template DeviceMemory<kF32CType>::DeviceMemory(std::size_t n, kF32CType value);
template DeviceMemory<kF64CType>::DeviceMemory(std::size_t n, kF64CType value);
template DeviceMemory<uint64_t>::DeviceMemory(std::size_t n, uint64_t value);

// Destruct by calling cude free
template <typename T>
    requires(!std::is_const_v<T>)
DeviceMemory<T>::~DeviceMemory() {
    if (_p) {
        StorageCUDA::current_bytes_allocated -= _n * sizeof(T);
        const auto status = cudaFree(_p);
        if (status != cudaSuccess) {
            TT_ERROR("cudaFree error: " + std::string(cudaGetErrorString(status)));
        }
        _p = nullptr;
        _n = 0;
    }
}
template DeviceMemory<kBoolCType>::~DeviceMemory();
template DeviceMemory<kI16CType>::~DeviceMemory();
template DeviceMemory<kI32CType>::~DeviceMemory();
template DeviceMemory<kI64CType>::~DeviceMemory();
template DeviceMemory<kF32CType>::~DeviceMemory();
template DeviceMemory<kF64CType>::~DeviceMemory();
template DeviceMemory<uint64_t>::~DeviceMemory();

// Clone underyling memory
template <typename T>
    requires(!std::is_const_v<T>)
auto DeviceMemory<T>::clone() const -> DeviceMemory<T> {
    auto dm = DeviceMemory{_n};
    const auto memcpy_status = cudaMemcpy(_p, dm._p, _n * sizeof(T), cudaMemcpyDeviceToDevice);
    if (memcpy_status != cudaSuccess) {
        TT_EXCEPTION("cudaMemcpy error: " + std::string(cudaGetErrorString(memcpy_status)));
    }
    return dm;
}

// Caller will clone and thus we don't have to worry about shape/stride
template <typename T>
    requires(!std::is_const_v<T>)
auto DeviceMemory<T>::to_vec() const -> std::vector<std::remove_cv_t<T>> {
    std::vector<std::remove_cv_t<T>> v(_n);
    const auto status = cudaMemcpy(v.data(), _p, _n * sizeof(T), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        TT_EXCEPTION("cudaMemcpy error: " + std::string(cudaGetErrorString(status)));
    }
    return v;
}

// Get a single item
template <typename T>
    requires(!std::is_const_v<T>)
auto DeviceMemory<T>::item(int index) const -> T {
    if (index < 0 || static_cast<std::size_t>(index) > _n) {
        TT_EXCEPTION(std::format("Index {} is out of bounds for storage of size {}", index, _n));
    }
    T value;
    const auto status = cudaMemcpy(&value, _p + index, sizeof(T), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        TT_EXCEPTION("cudaMemcpy error: " + std::string(cudaGetErrorString(status)));
    }
    return value;
}

// Initialize from aranged sequence
template <typename T>
    requires(!std::is_const_v<T>)
auto DeviceMemory<T>::AllocateArange(std::size_t n) -> DeviceMemory<T> {
    auto device_memory = DeviceMemory(n);
    const auto kernel = init_arange_kernel<T>;
    launch(kernel, grid_1d(n), block_1d(), device_memory.data_ptr(), static_cast<int>(n));
    return device_memory;
}

// Explicit template instantiation by storing function pointers
template <typename... Ts>
constexpr auto instantiate_device_memory() {
    return std::tuple_cat(
        std::make_tuple(
            &DeviceMemory<Ts>::clone,
            &DeviceMemory<Ts>::to_vec,
            &DeviceMemory<Ts>::item,
            &DeviceMemory<Ts>::AllocateArange
        )...
    );
}
template auto instantiate_device_memory<kBoolCType, kI16CType, kI32CType, kI64CType, kF32CType, kF64CType, uint64_t>();

}    // namespace tinytensor::cuda
