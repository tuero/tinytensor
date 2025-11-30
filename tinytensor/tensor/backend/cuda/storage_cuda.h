// storage_cuda.h
// Underlying storage for CUDA backend

#ifndef TINYTENSOR_STORAGE_CUDA_H_
#define TINYTENSOR_STORAGE_CUDA_H_

#include <tt/concepts.h>
#include <tt/scalar.h>

#include "tensor/backend/cuda/data_types.cuh"
#include "tensor/storage_base.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>

namespace tinytensor::cuda {

// CUDA storage is a wrapper around DeviceMemroy
class StorageCUDA : public StorageBase {
    using kBoolCType = to_ctype_t<kBool>;
    using kI16CType = to_ctype_t<kI16>;
    using kI32CType = to_ctype_t<kI32>;
    using kI64CType = to_ctype_t<kI64>;
    using kF32CType = to_ctype_t<kF32>;
    using kF64CType = to_ctype_t<kF64>;

public:
    using StorageT = std::variant<
        DeviceMemory<kBoolCType>,
        DeviceMemory<kI16CType>,
        DeviceMemory<kI32CType>,
        DeviceMemory<kI64CType>,
        DeviceMemory<kF32CType>,
        DeviceMemory<kF64CType>>;

    // Construct from stl vector
    template <typename T>
    StorageCUDA(int device_num, const std::vector<T> &data);

    // Construct from device memory
    template <typename T>
    StorageCUDA(DeviceMemory<T> &&other);

    // Construct from device memory
    template <typename T>
    StorageCUDA(int device_num, std::size_t n, T value);

    ~StorageCUDA() override = default;
    StorageCUDA(const StorageCUDA &) = delete;
    StorageCUDA(StorageCUDA &&) = default;
    auto operator=(const StorageCUDA &) -> StorageCUDA & = delete;
    auto operator=(StorageCUDA &&) -> StorageCUDA & = delete;

    template <typename T>
    static auto arange(int device_num, std::size_t) -> StorageCUDA;

    template <typename T>
    auto item(int index) -> T;

    StorageT dev_memory;

    inline static std::unordered_map<int, uint64_t> current_bytes_allocated;
    inline static std::unordered_map<int, uint64_t> total_bytes_allocated;
};

// Get the number of cuda devices
auto get_device_count() -> int;

}    // namespace tinytensor::cuda

#endif    // TINYTENSOR_STORAGE_CUDA_H_
