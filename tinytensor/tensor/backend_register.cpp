// backend_base.h
// Used to get backend instance

#include "backend_register.h"

#include <tt/device.h>
#include <tt/exception.h>

#include "backend/cpu/backend_cpu.h"
#include "backend_base.h"
#ifdef TT_CUDA
#include "backend/cuda/backend_cuda.h"
#endif

#include <memory>

namespace tinytensor {

BackendBase *get_backend(Backend backend) {
    switch (backend) {
    case Backend::cpu: {
        static const std::unique_ptr<cpu::BackendCPU> backend_cpu = std::make_unique<cpu::BackendCPU>();
        return backend_cpu.get();
    }
#ifdef TT_CUDA
    case Backend::cuda: {
        static const std::unique_ptr<cuda::BackendCUDA> backend_gpu = std::make_unique<cuda::BackendCUDA>();
        return backend_gpu.get();
    }
#endif
    default:
        TT_EXCEPTION("Unknown device type.");
    }
}

BackendBase *get_backend(Device device) {
    return get_backend(device.backend);
}

}    // namespace tinytensor
