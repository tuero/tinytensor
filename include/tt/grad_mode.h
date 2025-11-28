// grad_mode.h
// Controls whether gradients are tracked

#ifndef TINYTENSOR_AUTOGRAD_GRAD_MODE_H_
#define TINYTENSOR_AUTOGRAD_GRAD_MODE_H_

#include <tt/export.h>

namespace tinytensor::autograd {

// Flag for the current grad mode
class TINYTENSOR_EXPORT GradMode {
public:
    GradMode() = delete;
    static inline bool is_enabled() {
        return grad_mode;
    }
    static inline void set_enabled(bool enabled) {
        grad_mode = enabled;
    }

private:
    inline static thread_local bool grad_mode = true;    // NOLINT(*avoid-non-const-global-variables)
};

// Guard which disable autograd tracking while the instantiated object is in scope
class TINYTENSOR_EXPORT NoGradGuard {
public:
    NoGradGuard()
        : prev_mode(GradMode::is_enabled()) {
        GradMode::set_enabled(false);
    }
    ~NoGradGuard() {
        GradMode::set_enabled(prev_mode);
    }

    // Doesn't make sense to move/copy the guard
    NoGradGuard(const NoGradGuard &) = delete;
    NoGradGuard(NoGradGuard &&) = delete;
    NoGradGuard &operator=(const NoGradGuard &) = delete;
    NoGradGuard &operator=(NoGradGuard &&) = delete;

private:
    bool prev_mode;
};

}    // namespace tinytensor::autograd

#endif    // TINYTENSOR_AUTOGRAD_GRAD_MODE_H_
