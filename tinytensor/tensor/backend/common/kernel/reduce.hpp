// reduce.hpp
// Common element-wise reduction kernels

#ifndef TINYTENSOR_BACKEND_COMMON_KERNEL_REDUCE_H_
#define TINYTENSOR_BACKEND_COMMON_KERNEL_REDUCE_H_

#include <tt/macros.h>
#include <tt/scalar.h>

#include <cmath>
#include <cstdint>
#include <limits>

#if defined(__CUDACC__)
#include <nvfunctional>
#else
#include <functional>
#endif

namespace tinytensor::common::kernel::reduce {

template <typename T>
struct ValIndex {
    T val;
    int idx;
};

template <typename T>
struct OpMinimum {
    using VT = T;
    using VIT = ValIndex<T>;
    OpMinimum([[maybe_unused]] T n = 0) {};
    static constexpr T padding_value = std::numeric_limits<T>::max();
    TT_STD_FUNC<T(T, T)> TT_HOST_DEVICE operator()() const {
        return [](auto lhs, auto rhs) { return (lhs < rhs) ? lhs : rhs; };
    }
    TT_STD_FUNC<T(VIT, VIT)> TT_HOST_DEVICE value() const {
        return [](const VIT &lhs, const VIT &rhs) { return (lhs.val < rhs.val) ? lhs.val : rhs.val; };
    }
    TT_STD_FUNC<int(VIT, VIT)> TT_HOST_DEVICE index() const {
        return [](const VIT &lhs, const VIT &rhs) { return (lhs.val < rhs.val) ? lhs.idx : rhs.idx; };
    }
    TT_STD_FUNC<T(VIT)> TT_HOST_DEVICE res() const {
        return [](const VIT &input) { return input.val; };
    }
};
template <typename T>
struct OpArgMinimum {
    using VT = T;
    using VIT = ValIndex<T>;
    OpArgMinimum([[maybe_unused]] T n = 0) {};
    static constexpr T padding_value = std::numeric_limits<T>::max();
    TT_STD_FUNC<T(VIT, VIT)> TT_HOST_DEVICE value() const {
        return [](const VIT &lhs, const VIT &rhs) { return (lhs.val < rhs.val) ? lhs.val : rhs.val; };
    }
    TT_STD_FUNC<int(VIT, VIT)> TT_HOST_DEVICE index() const {
        return [](const VIT &lhs, const VIT &rhs) { return (lhs.val < rhs.val) ? lhs.idx : rhs.idx; };
    }
    TT_STD_FUNC<int(VIT)> TT_HOST_DEVICE res() const {
        return [](const VIT &input) { return input.idx; };
    }
};

template <typename T>
struct OpMaximum {
    using VT = T;
    using VIT = ValIndex<T>;
    OpMaximum([[maybe_unused]] T n = 0) {};
    static constexpr T padding_value = std::numeric_limits<T>::lowest();
    TT_STD_FUNC<T(T, T)> TT_HOST_DEVICE operator()() const {
        return [](auto lhs, auto rhs) { return (lhs > rhs) ? lhs : rhs; };
    }
    TT_STD_FUNC<T(VIT, VIT)> TT_HOST_DEVICE value() const {
        return [](const VIT &lhs, const VIT &rhs) { return (lhs.val > rhs.val) ? lhs.val : rhs.val; };
    }
    TT_STD_FUNC<int(VIT, VIT)> TT_HOST_DEVICE index() const {
        return [](const VIT &lhs, const VIT &rhs) { return (lhs.val > rhs.val) ? lhs.idx : rhs.idx; };
    }
    TT_STD_FUNC<T(VIT)> TT_HOST_DEVICE res() const {
        return [](const VIT &input) { return input.val; };
    }
};
template <typename T>
struct OpArgMaximum {
    using VT = T;
    using VIT = ValIndex<T>;
    OpArgMaximum([[maybe_unused]] T n = 0) {};
    static constexpr T padding_value = std::numeric_limits<T>::lowest();
    TT_STD_FUNC<T(VIT, VIT)> TT_HOST_DEVICE value() const {
        return [](const VIT &lhs, const VIT &rhs) { return (lhs.val > rhs.val) ? lhs.val : rhs.val; };
    }
    TT_STD_FUNC<int(VIT, VIT)> TT_HOST_DEVICE index() const {
        return [](const VIT &lhs, const VIT &rhs) { return (lhs.val > rhs.val) ? lhs.idx : rhs.idx; };
    }
    TT_STD_FUNC<int(VIT)> TT_HOST_DEVICE res() const {
        return [](const VIT &input) { return input.idx; };
    }
};

template <typename T>
struct OpSum {
    using VT = T;
    using VIT = ValIndex<T>;
    OpSum([[maybe_unused]] T n = 0) {};
    static constexpr T padding_value = 0;
    TT_STD_FUNC<T(T, T)> TT_HOST_DEVICE operator()() const {
        return [](auto lhs, auto rhs) { return lhs + rhs; };
    }
    TT_STD_FUNC<T(VIT, VIT)> TT_HOST_DEVICE value() const {
        return [](const VIT &lhs, const VIT &rhs) { return lhs.val + rhs.val; };
    }
    TT_STD_FUNC<int(VIT, VIT)> TT_HOST_DEVICE index() const {
        return []([[maybe_unused]] const VIT &lhs, [[maybe_unused]] const VIT &rhs) { return 0; };
    }
    TT_STD_FUNC<T(VIT)> TT_HOST_DEVICE res() const {
        return [](const VIT &input) { return input.val; };
    }
};

// @NOTE: Do not use this in the reduce cuda kernel, as this assumes we do not touch partial computations multiple times
// (we would divide by N more than once)
template <typename T>
struct OpMean {
    using VT = T;
    using VIT = ValIndex<T>;
    OpMean(T _n = 0)
        : n(_n) {};
    static constexpr T padding_value = 0;
    TT_STD_FUNC<T(T, T)> TT_HOST_DEVICE operator()() const {
        return [this](auto lhs, auto rhs) { return (lhs / n) + rhs; };
    }
    TT_STD_FUNC<T(VIT, VIT)> TT_HOST_DEVICE value() const {
        return [this](const VIT &lhs, const VIT &rhs) { return (lhs.val / n) + rhs.val; };
    }
    TT_STD_FUNC<int(VIT, VIT)> TT_HOST_DEVICE index() const {
        return []([[maybe_unused]] const VIT &lhs, [[maybe_unused]] const VIT &rhs) { return 0; };
    }
    TT_STD_FUNC<T(VIT)> TT_HOST_DEVICE res() const {
        return [](const VIT &input) { return input.val; };
    }
    T n;
};

template <typename T>
struct OpAnd {
    using VT = T;
    using VIT = ValIndex<T>;
    OpAnd([[maybe_unused]] T n = 0) {};
    static constexpr T padding_value = static_cast<T>(1);
    TT_STD_FUNC<bool(T, T)> TT_HOST_DEVICE operator()() const {
        return [](auto lhs, auto rhs) { return lhs && rhs; };
    }
    TT_STD_FUNC<bool(VIT, VIT)> TT_HOST_DEVICE value() const {
        return [](const VIT &lhs, const VIT &rhs) { return static_cast<bool>(lhs.val) && static_cast<bool>(rhs.val); };
    }
    TT_STD_FUNC<int(VIT, VIT)> TT_HOST_DEVICE index() const {
        return []([[maybe_unused]] const VIT &lhs, [[maybe_unused]] const VIT &rhs) { return 0; };
    }
    TT_STD_FUNC<bool(VIT)> TT_HOST_DEVICE res() const {
        return [](const VIT &input) { return input.val; };
    }
};

template <typename T>
struct OpOr {
    using VT = T;
    using VIT = ValIndex<T>;
    OpOr([[maybe_unused]] T n = 0) {};
    static constexpr T padding_value = static_cast<T>(0);
    TT_STD_FUNC<bool(T, T)> TT_HOST_DEVICE operator()() const {
        return [](auto lhs, auto rhs) { return lhs || rhs; };
    }
    TT_STD_FUNC<bool(VIT, VIT)> TT_HOST_DEVICE value() const {
        return [](const VIT &lhs, const VIT &rhs) { return static_cast<bool>(lhs.val) || static_cast<bool>(rhs.val); };
    }
    TT_STD_FUNC<int(VIT, VIT)> TT_HOST_DEVICE index() const {
        return []([[maybe_unused]] const VIT &lhs, [[maybe_unused]] const VIT &rhs) { return 0; };
    }
    TT_STD_FUNC<bool(VIT)> TT_HOST_DEVICE res() const {
        return [](const VIT &input) { return input.val; };
    }
};

}    // namespace tinytensor::common::kernel::reduce

#endif    // TINYTENSOR_BACKEND_COMMON_KERNEL_REDUCE_H_
