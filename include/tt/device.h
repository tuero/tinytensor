// device.h
// Device type

#ifndef TINYTENSOR_DEVICE_H_
#define TINYTENSOR_DEVICE_H_

#include <tt/exception.h>

#include <format>
#include <ostream>
#include <string>

namespace tinytensor {

// Supported backend types
enum class Backend {
    cpu,
    jit,
#ifdef TT_CUDA
    cuda,
#endif
};

constexpr auto to_string(Backend backend) -> std::string {
    switch (backend) {
    case Backend::cpu:
        return "cpu";
    case Backend::jit:
        return "jit";
#ifdef TT_CUDA
    case Backend::cuda:
        return "cuda";
#endif
    }
    TT_EXCEPTION("Unknown device type.");
}

// Device is an enum backend + device ID (multi-device support)
struct Device {
#ifdef TT_CUDA
    constexpr static auto CUDA(int dev_id) -> Device {
        return {.backend = Backend::cuda, .id = dev_id};
    }
#endif

    // Operator!= provided by compiler since C++20
    constexpr auto operator==(const Device &other) const -> bool {
        return backend == other.backend && id == other.id;
    }

    Backend backend;
    int id;
};

// Shortnames
constexpr Device kCPU = Device{.backend = Backend::cpu, .id = 0};
constexpr Device kJIT = Device{.backend = Backend::jit, .id = 0};
#ifdef TT_CUDA
constexpr Device kCUDA = Device{.backend = Backend::cuda, .id = 0};
#endif

inline auto operator<<(std::ostream &os, const Device &device) -> std::ostream & {
    os << to_string(device.backend) << ":" << device.id;
    return os;
}
}    // namespace tinytensor

template <>
struct std::formatter<tinytensor::Device> : std::formatter<std::string> {
    auto format(const tinytensor::Device &device, format_context &ctx) const {
        return formatter<string>::format(std::format("{}:{}", to_string(device.backend), device.id), ctx);
    }
};

#endif    // TINYTENSOR_DEVICE_H_
