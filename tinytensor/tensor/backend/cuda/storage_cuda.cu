// storage_cuda.cu
// Underlying storage for CUDA backend

#include <tt/scalar.h>

#include "tensor/backend/cuda/data_types.cuh"
#include "tensor/backend/cuda/storage_cuda.h"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#include <cstddef>
#include <utility>
#include <variant>
#include <vector>

namespace tinytensor::cuda {

using kBoolCType = to_ctype_t<kBool>;
using kI16CType = to_ctype_t<kI16>;
using kI32CType = to_ctype_t<kI32>;
using kI64CType = to_ctype_t<kI64>;
using kF32CType = to_ctype_t<kF32>;
using kF64CType = to_ctype_t<kF64>;

// Construct from stl vector
template <typename T>
StorageCUDA::StorageCUDA(int device_num, const std::vector<T> &data)
    : dev_memory(MakeDeviceMemory<T>(device_num, data)) {}

template StorageCUDA::StorageCUDA(int device_num, const std::vector<kBoolCType> &data);
template StorageCUDA::StorageCUDA(int device_num, const std::vector<kI16CType> &data);
template StorageCUDA::StorageCUDA(int device_num, const std::vector<kI32CType> &data);
template StorageCUDA::StorageCUDA(int device_num, const std::vector<kI64CType> &data);
template StorageCUDA::StorageCUDA(int device_num, const std::vector<kF32CType> &data);
template StorageCUDA::StorageCUDA(int device_num, const std::vector<kF64CType> &data);

// Construct from device memory
template <typename T>
StorageCUDA::StorageCUDA(DeviceMemory<T> &&other)
    : dev_memory(std::move(other)) {}

template StorageCUDA::StorageCUDA(DeviceMemory<kBoolCType> &&other);
template StorageCUDA::StorageCUDA(DeviceMemory<kI16CType> &&other);
template StorageCUDA::StorageCUDA(DeviceMemory<kI32CType> &&other);
template StorageCUDA::StorageCUDA(DeviceMemory<kI64CType> &&other);
template StorageCUDA::StorageCUDA(DeviceMemory<kF32CType> &&other);
template StorageCUDA::StorageCUDA(DeviceMemory<kF64CType> &&other);

// Construct from device memory
template <typename T>
StorageCUDA::StorageCUDA(int device_num, std::size_t n, T value)
    : dev_memory(MakeDeviceMemory<T>(device_num, n, value)) {}

template StorageCUDA::StorageCUDA(int device_num, std::size_t n, kBoolCType value);
template StorageCUDA::StorageCUDA(int device_num, std::size_t n, kI16CType value);
template StorageCUDA::StorageCUDA(int device_num, std::size_t n, kI32CType value);
template StorageCUDA::StorageCUDA(int device_num, std::size_t n, kI64CType value);
template StorageCUDA::StorageCUDA(int device_num, std::size_t n, kF32CType value);
template StorageCUDA::StorageCUDA(int device_num, std::size_t n, kF64CType value);

template <typename T>
auto StorageCUDA::arange(int device_num, std::size_t n) -> StorageCUDA {
    return DeviceMemory<T>::AllocateArange(device_num, n);
}

template auto StorageCUDA::arange<kBoolCType>(int device_num, std::size_t n) -> StorageCUDA;
template auto StorageCUDA::arange<kI16CType>(int device_num, std::size_t n) -> StorageCUDA;
template auto StorageCUDA::arange<kI32CType>(int device_num, std::size_t n) -> StorageCUDA;
template auto StorageCUDA::arange<kI64CType>(int device_num, std::size_t n) -> StorageCUDA;
template auto StorageCUDA::arange<kF32CType>(int device_num, std::size_t n) -> StorageCUDA;
template auto StorageCUDA::arange<kF64CType>(int device_num, std::size_t n) -> StorageCUDA;

template <typename T>
auto StorageCUDA::item(int index) -> T {
    return std::visit([&](auto &&storage) -> T { return static_cast<T>(storage.item(index)); }, dev_memory);
}
template auto StorageCUDA::item<kBoolCType>(int index) -> uint8_t;
template auto StorageCUDA::item<kI16CType>(int index) -> int16_t;
template auto StorageCUDA::item<kI32CType>(int index) -> int32_t;
template auto StorageCUDA::item<kI64CType>(int index) -> int64_t;
template auto StorageCUDA::item<kF32CType>(int index) -> float;
template auto StorageCUDA::item<kF64CType>(int index) -> double;

auto get_device_count() -> int {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    return device_count;
}

}    // namespace tinytensor::cuda
