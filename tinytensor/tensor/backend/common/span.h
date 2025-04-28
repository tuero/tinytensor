// span.h
// Common span type (pointer + len)

#ifndef TINYTENSOR_BACKEND_COMMON_SPAN_H_
#define TINYTENSOR_BACKEND_COMMON_SPAN_H_

#include <tt/macros.h>
#include <tt/shape.h>

#include <cassert>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace tinytensor {

// Wrapper around device memory with size
template <typename T>
class NonOwningSpan {
public:
    TT_HOST_DEVICE NonOwningSpan() = default;
    TT_HOST_DEVICE explicit NonOwningSpan(T *p, std::size_t n)
        : _p(p), _n(n) {}

    [[nodiscard]] TT_HOST_DEVICE TT_INLINE auto operator[](std::size_t idx) const -> const T & {
        assert(idx < _n);
        assert(_p);
        return _p[idx];    // NOLINT(*-pointer-arithmetic)
    }
    [[nodiscard]] TT_HOST_DEVICE TT_INLINE auto operator[](std::size_t idx) -> T & {
        assert(idx < _n);
        assert(_p);
        return _p[idx];    // NOLINT(*-pointer-arithmetic)
    }
    [[nodiscard]] TT_HOST_DEVICE TT_INLINE auto operator[](int idx) const -> const T & {
        assert(idx >= 0 && static_cast<std::size_t>(idx) < _n);
        assert(_p);
        return _p[static_cast<std::size_t>(idx)];    // NOLINT(*-pointer-arithmetic)
    }
    [[nodiscard]] TT_HOST_DEVICE TT_INLINE auto operator[](int idx) -> T & {
        assert(idx >= 0 && static_cast<std::size_t>(idx) < _n);
        assert(_p);
        return _p[static_cast<std::size_t>(idx)];    // NOLINT(*-pointer-arithmetic)
    }

    [[nodiscard]] TT_HOST_DEVICE TT_INLINE auto get() const -> const T * {
        return _p;
    }

    [[nodiscard]] TT_HOST_DEVICE TT_INLINE auto get() -> T * {
        return _p;
    }

    [[nodiscard]] TT_HOST_DEVICE TT_INLINE auto size() const -> std::size_t {
        return _n;
    }

    // NonOwningSpan<const T> <- NonOwningSpan<T>
    template <typename U>
        requires std::is_convertible_v<U *, T *>
    TT_HOST_DEVICE NonOwningSpan(const NonOwningSpan<U> &dp)
        : _p(dp.get()), _n(dp.size()) {}

private:
    T *_p = nullptr;
    std::size_t _n = 0;
};

// CPU span can simply inherit and convert from vec
template <typename T>
struct HostSpan : public NonOwningSpan<T> {
    using NonOwningSpan<T>::NonOwningSpan;

    // Converting constructor from non-temporary vector (otherwise span to free'd memory)
    HostSpan(const std::vector<std::remove_cv_t<T>> &v)
        requires std::is_const_v<T>
        : NonOwningSpan<T>(v.data(), v.size()) {}
    HostSpan(std::vector<std::remove_cv_t<T>> &v)
        requires(!std::is_const_v<T>)
        : NonOwningSpan<T>(v.data(), v.size()) {}
    HostSpan(std::vector<std::remove_cv_t<T>> &&) = delete;

    // Converting constructor from non-temporary shape
    HostSpan(const Shape &shape)
        : NonOwningSpan<T>(shape.to_vec().data(), static_cast<std::size_t>(shape.size())) {}
    HostSpan(Shape &&) = delete;
};

}    // namespace tinytensor

#endif    // TINYTENSOR_BACKEND_COMMON_SPAN_H_
