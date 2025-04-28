// distribution.hpp
// Common distribution kernels

#ifndef TINYTENSOR_BACKEND_COMMON_KERNEL_DISTRIBUTION_H_
#define TINYTENSOR_BACKEND_COMMON_KERNEL_DISTRIBUTION_H_

#include <tt/concepts.h>
#include <tt/macros.h>
#include <tt/random.h>

#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <type_traits>

#if defined(__CUDACC__)
#include <nvfunctional>
#else
#include <functional>
#endif

namespace tinytensor::common::kernel::distribution {

namespace {
// Wider width types for generating random bits
template <typename T>
struct to_wide;

template <>
struct to_wide<uint32_t> {
    using type = uint64_t;
};
template <>
struct to_wide<uint64_t> {
    using type = __uint128_t;
};

template <typename T>
using to_wide_t = to_wide<T>::type;

template <typename T>
struct float_bits;

template <>
struct float_bits<float> {
    using unsigned_t = uint32_t;
    // 127 exponential offset 23 bits shifted
    constexpr static unsigned_t EXP_MASK = static_cast<unsigned_t>(127) << 23;
    constexpr static unsigned_t SHIFT = 9;
};

template <>
struct float_bits<double> {
    using unsigned_t = uint64_t;
    // 1023 exponential offset 52 bits shifted
    constexpr static unsigned_t EXP_MASK = static_cast<unsigned_t>(1023) << 52;
    constexpr static unsigned_t SHIFT = 12;
};
}    // namespace

// https://arxiv.org/pdf/1805.10941.pdf
// https://github.com/gcc-mirror/gcc/blob/master/libstdc%2B%2B-v3/include/bits/uniform_int_dist.h#L252C10-L252C13
TT_HOST_DEVICE inline uint64_t rand_in_range(uint64_t s, Generator &gen) {
    using WideT = __uint128_t;
    static_assert(sizeof(uint64_t) <= 2 * sizeof(WideT));
    WideT m = static_cast<WideT>(gen()) * static_cast<WideT>(s);
    auto l = static_cast<uint64_t>(m);
    if (l < s) {
        uint64_t t = -s % s;    // Wraps around (2^l - s) mod s
        while (l < t) {
            m = static_cast<WideT>(gen()) * static_cast<WideT>(s);
            l = static_cast<uint64_t>(m);
        }
    }
    return static_cast<uint64_t>(m >> std::numeric_limits<uint64_t>::digits);
}

/**
 * Generate a sample from a Uniform distribution of floating point type within a given range [low,
 * high)
 * @param low Lower bound on range to sample in
 * @param high Upper bound on range to sample in
 * @note Based on https://arxiv.org/pdf/1805.10941.pdf
 * @note
 * https://github.com/gcc-mirror/gcc/blob/master/libstdc%2B%2B-v3/include/bits/uniform_int_dist.h#L252C10-L252C13
 */
template <typename T>
    requires std::is_floating_point_v<T>
TT_HOST_DEVICE T uniform_real_sample(T low, T high, Generator &gen) {
    assert(low < high);
    using unsigned_t = float_bits<T>::unsigned_t;
    const auto SHIFT = float_bits<T>::SHIFT;
    const auto EXP_MASK = float_bits<T>::EXP_MASK;
    const auto r = std::bit_cast<T>((static_cast<unsigned_t>(gen()) >> SHIFT) | EXP_MASK) - static_cast<T>(1.0);
    return r * (high - low) + low;
}

/**
 * Generate a sample from a Uniform distribution of floating point type within a given range [low,
 * high)
 * @param low Lower bound on range to sample in
 * @param high Upper bound on range to sample in
 */
template <typename T>
    requires std::is_floating_point_v<T>
struct OpUniformReal {
    TT_STD_FUNC<T(T, T, Generator)> TT_HOST_DEVICE operator()() const {
        return [](T low, T high, Generator gen) { return uniform_real_sample(low, high, gen); };
    }
};

/**
 * Generate a sample from a Uniform distribution of integral type within a given range [low, high)
 * @param low Lower bound on range to sample in
 * @param high Upper bound on range to sample in
 */
template <typename T>
    requires std::is_integral_v<T> || std::is_floating_point_v<T>
struct OpUniformInt {
    TT_STD_FUNC<T(T, T, Generator)> TT_HOST_DEVICE operator()() const {
        return [](T low, T high, Generator gen) {
            const auto range = static_cast<uint64_t>(high - low);
            const auto x = static_cast<T>(rand_in_range(range, gen));
            return static_cast<T>(x + low);
        };
    }
};

/**
 * Generate a sample from a Bernoulli distribution
 * @param p Probability of success
 */
template <typename T>
    requires std::is_floating_point_v<T>
struct OpBernoulli {
    TT_STD_FUNC<T(T, Generator)> TT_HOST_DEVICE operator()() const {
        return [](T p, Generator gen) {
            const auto u = uniform_real_sample(static_cast<T>(0), static_cast<T>(1), gen);
            return static_cast<T>(u < p);
        };
    }
};

/**
 * Generate a sample from a Binomial distribution
 * @param p Probability of success
 * @param num_draws Number of draws of the Bernoulli distribution
 */
template <typename T>
    requires std::is_floating_point_v<T>
struct OpBinomial {
    TT_STD_FUNC<T(T, T, Generator)> TT_HOST_DEVICE operator()() const {
        return [](T p, T num_draws, Generator gen) {
            T sum{};
            for ([[maybe_unused]] int i = 0; i < static_cast<int>(num_draws); ++i) {
                const auto u = uniform_real_sample(static_cast<T>(0), static_cast<T>(1), gen);
                sum += static_cast<T>(u < p);
            }
            return sum;
        };
    }
};

/**
 * Generate a sample from a Geometric distribution
 * @note Uses representation of X Bernoulli trials to get one success, support = {1, 2, 3, ...}
 * @param p Probability of success
 */
template <typename T>
    requires std::is_floating_point_v<T>
struct OpGeometric {
    TT_STD_FUNC<T(T, Generator)> TT_HOST_DEVICE operator()() const {
        return [](T p, Generator gen) {
            const auto u = uniform_real_sample(static_cast<T>(0), static_cast<T>(1), gen);
            return p == 1 ? 1 : std::ceil(std::log1p(-u) / std::log1p(-p));
        };
    }
};

/**
 * Generate a sample from a Poisson distribution
 * @param lambda Mean of the distribution
 * @note Uses method from Devroye, Luc (1986). "Discrete Univariate Distributions" (PDF).
 * Non-Uniform Random Variate Generation. New York, NY: Springer-Verlag. pp. 485–553.
 *       https://en.wikipedia.org/wiki/Poisson_distribution#Random_variate_generation
 */
template <typename T>
    requires std::is_floating_point_v<T>
struct OpPoisson {
    TT_STD_FUNC<T(T, Generator)> TT_HOST_DEVICE operator()() const {
        return [](T lambda, Generator gen) {
            T x{};
            auto p = std::exp(-lambda);
            auto s = p;
            const auto u = uniform_real_sample(static_cast<T>(0), static_cast<T>(1), gen);
            while (u > s) {
                ++x;
                p *= lambda / x;
                s += p;
            }
            return x;
        };
    }
};

/**
 * Generate a sample from an Exponential distribution
 * @param lambda The rate of occurances
 * @note Uses method from
 * https://en.wikipedia.org/wiki/Exponential_distribution#Random_variate_generation
 */
template <typename T>
    requires std::is_floating_point_v<T>
struct OpExponential {
    TT_STD_FUNC<T(T, Generator)> TT_HOST_DEVICE operator()() const {
        return [](T lambda, Generator gen) {
            const auto u = uniform_real_sample(static_cast<T>(0), static_cast<T>(1), gen);
            return -std::log1p(-u) / lambda;
        };
    }
};

/**
 * Generate a sample from a Normal distribution
 * @param mu Mean of the distribution
 * @param std Standard deviation of the distribution
 * @note Uses the Box-Muller transform https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
 */
template <typename T>
    requires std::is_floating_point_v<T>
struct OpNormal {
    TT_STD_FUNC<T(T, T, Generator)> TT_HOST_DEVICE operator()() const {
        return [](T mu, T std, Generator gen) {
            const auto u1 = uniform_real_sample(static_cast<T>(0), static_cast<T>(1), gen);
            const auto u2 = uniform_real_sample(static_cast<T>(0), static_cast<T>(1), gen);
            const auto r =
                std::sqrt(static_cast<T>(-2) * std::log1p(-u1));    // log(1 - u1) for u \in [0, 1) ensures log(> 0)
            const auto theta = static_cast<T>(2) * std::numbers::pi_v<T> * u2;
            const auto z1 = r * std::sin(theta);
            return std * z1 + mu;
        };
    }
};

/**
 * Generate a sample from a Caughy distribution
 * @param loc Mode/Median of the distribution
 * @param scale Half width at maximum
 */
template <typename T>
    requires std::is_floating_point_v<T>
struct OpCauchy {
    TT_STD_FUNC<T(T, T, Generator)> TT_HOST_DEVICE operator()() const {
        return [](T loc, T scale, Generator gen) {
            const auto u = uniform_real_sample(static_cast<T>(0), static_cast<T>(1), gen);
            const auto z = std::tan(std::numbers::pi_v<T> * (u - 0.5));
            return loc + scale * z;
        };
    }
};

/**
 * Generate a sample from a Log-Normal distribution
 * @param mu Mean of the logarithm of the sample
 * @param std Standard deviation of the logarithm of the sample
 */
template <typename T>
    requires std::is_floating_point_v<T>
struct OpLogNormal {
    TT_STD_FUNC<T(T, T, Generator)> TT_HOST_DEVICE operator()() const {
        return [](T mu, T std, Generator gen) {
            auto OP = OpNormal<T>{};
            return std::exp(OP()(mu, std, gen));
        };
    }
};

/**
 * Generate a sample from a Weibull distribution
 * @param scale Scale parameter lamda
 * @param shape Shape parameter k
 * @note https://en.wikipedia.org/wiki/Weibull_distribution#Related_distributions
 */
template <typename T>
    requires std::is_floating_point_v<T>
struct OpWeibull {
    TT_STD_FUNC<T(T, T, Generator)> TT_HOST_DEVICE operator()() const {
        return [](T scale, T shape, Generator gen) {
            const auto u = uniform_real_sample(static_cast<T>(0), static_cast<T>(1), gen);
            const auto one = static_cast<T>(1);
            return scale * std::pow(std::log(one / (one - u)), one / shape);
        };
    }
};

}    // namespace tinytensor::common::kernel::distribution

#endif    // TINYTENSOR_BACKEND_COMMON_KERNEL_DISTRIBUTION_H_
