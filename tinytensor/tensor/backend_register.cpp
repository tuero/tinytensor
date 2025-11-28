#include "backend_register.h"

#include <tt/device.h>
#include <tt/exception.h>

#include "backend/cpu/backend_cpu.h"
#include "backend/jit/backend_jit.h"
#include "backend_base.h"

#ifdef TT_CUDA
#include "backend/cuda/backend_cuda.h"
#endif

#include <memory>

namespace tinytensor {

BackendBase *get_backend(Device device) {
    switch (device.backend) {
    case Backend::cpu: {
        static const std::unique_ptr<cpu::BackendCPU> backend_cpu = std::make_unique<cpu::BackendCPU>();
        return backend_cpu.get();
    }
    case Backend::jit: {
        // use a static singleton, like the CPU backend
        static const std::unique_ptr<BackendJIT> backend_jit = std::make_unique<BackendJIT>();
        return backend_jit.get();
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

}    // namespace tinytensor