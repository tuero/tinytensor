#include <tt/device.h>
#include <tt/scalar.h>
#include <tt/tensor.h>

template <typename T, typename F, typename... Args>
void runner_single_type(const F &f, Args... args) {
    f.template operator()<T>(tinytensor::kCPU, args...);
#ifdef TT_CUDA
    f.template operator()<T>(tinytensor::kCUDA, args...);
#endif
}

template <typename F, typename... Args>
void runner_boolean(F &&f, Args... args) {
    runner_single_type<bool>(std::forward<F>(f), args...);
}
template <typename F, typename... Args>
void runner_integral(F &&f, Args... args) {
    runner_single_type<tinytensor::to_ctype_t<tinytensor::kU8>>(std::forward<F>(f), args...);
    runner_single_type<tinytensor::to_ctype_t<tinytensor::kI16>>(std::forward<F>(f), args...);
    runner_single_type<tinytensor::to_ctype_t<tinytensor::kI32>>(std::forward<F>(f), args...);
    runner_single_type<tinytensor::to_ctype_t<tinytensor::kI64>>(std::forward<F>(f), args...);
}
template <typename F, typename... Args>
void runner_signed_integral(F &&f, Args... args) {
    runner_single_type<tinytensor::to_ctype_t<tinytensor::kI16>>(std::forward<F>(f), args...);
    runner_single_type<tinytensor::to_ctype_t<tinytensor::kI32>>(std::forward<F>(f), args...);
    runner_single_type<tinytensor::to_ctype_t<tinytensor::kI64>>(std::forward<F>(f), args...);
}
template <typename F, typename... Args>
void runner_floating_point(F &&f, Args... args) {
    runner_single_type<tinytensor::to_ctype_t<tinytensor::kF32>>(std::forward<F>(f), args...);
    runner_single_type<tinytensor::to_ctype_t<tinytensor::kF64>>(std::forward<F>(f), args...);
}
template <typename F, typename... Args>
void runner_all(F &&f, Args... args) {
    runner_boolean(std::forward<F>(f), args...);
    runner_integral(std::forward<F>(f), args...);
    runner_floating_point(std::forward<F>(f), args...);
}
template <typename F, typename... Args>
void runner_all_except_bool(F &&f, Args... args) {
    runner_integral(std::forward<F>(f), args...);
    runner_floating_point(std::forward<F>(f), args...);
}
