// backend_base.h
// Used to get backend instance

#ifndef TINYTENSOR_BACKEND_REGISTER_H_
#define TINYTENSOR_BACKEND_REGISTER_H_

#include <tt/device.h>

#include "backend_base.h"

namespace tinytensor {

auto get_backend(Device device) -> BackendBase *;

}    // namespace tinytensor

#endif    // TINYTENSOR_BACKEND_REGISTER_H_
