// util.h
// Custom utility helpers

#ifndef TINYTENSOR_UTIL_H_
#define TINYTENSOR_UTIL_H_

#include <concepts>
#include <ranges>

namespace tinytensor {

namespace detail {

// Type acts as a tag to find the correct operator| overload
template <typename C>
struct to_helper {};

// This actually does the work
template <typename Container, std::ranges::range R>
    requires std::convertible_to<std::ranges::range_value_t<R>, typename Container::value_type>
Container operator|(R &&r, to_helper<Container>) {
    return Container{r.begin(), r.end()};
}
}    // namespace detail

// https://stackoverflow.com/a/60971856/6641216
template <std::ranges::range Container>
    requires(!std::ranges::view<Container>)
auto to() {
    return detail::to_helper<Container>{};
}

// helper type for the visitor
template <class... Ts>
struct overloaded : Ts... {
    using Ts::operator()...;
};

template <typename T>
constexpr auto ceil_div(T x, T y) -> T {
    return (x + y - 1) / y;
}

// unreachable defined for C++23
// https://en.cppreference.com/w/cpp/utility/unreachable
[[noreturn]] inline void unreachable() {
    // Uses compiler specific extensions if possible.
    // Even if no extension is used, undefined behavior is still raised by
    // an empty function body and the noreturn attribute.
#if defined(_MSC_VER) && !defined(__clang__)    // MSVC
    __assume(false);
#else    // GCC, Clang
    __builtin_unreachable();
#endif
}

}    // namespace tinytensor

#endif    // TINYTENSOR_UTIL_H_
