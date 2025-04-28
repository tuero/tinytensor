// simd.h
// simd implementations

#ifndef TINYTENSOR_BACKEND_CPU_KERNEL_SIMD_H_
#define TINYTENSOR_BACKEND_CPU_KERNEL_SIMD_H_

#include <tt/concepts.h>

#include <cassert>
#include <cstdint>
#include <cstdlib>

// SIMD intrinsics
#if defined(__x86_64__)
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#else
#error Unknown architecture
#endif

namespace tinytensor::cpu::kernel {

// Replacement for compiler vector extension offered by gcc/clang
// Does a better job at getting the compiler to optimize than std::array, etc.

// No vmulq_s64 defined for m1
#if defined(__aarch64__)
inline int64x2_t vmulq_s64(const int64x2_t &lhs, const int64x2_t &rhs) {
    const auto lhs_low = vmovn_s64(lhs);
    const auto rhs_low = vmovn_s64(rhs);
    const auto hi = vmulq_s32(reinterpret_cast<int32x4_t>(rhs), vrev64q_s32(reinterpret_cast<int32x4_t>(lhs)));

    auto t = reinterpret_cast<uint32x4_t>(hi);
    auto a = reinterpret_cast<int64x2_t>(vpaddlq_u32(t));
    auto b = reinterpret_cast<uint64x2_t>(vshlq_n_s64(a, 32));
    auto res = vmlal_u32(b, reinterpret_cast<uint32x2_t>(lhs_low), reinterpret_cast<uint32x2_t>(rhs_low));
    return reinterpret_cast<int64x2_t>(res);
}
#endif

// typedef x64 intrinsics using aarch64 neon intrinsics
#if defined(__x86_64__)
using float32x4_t = __m128;
using float64x2_t = __m128d;
using uint8x16_t = __m128i;
using int16x8_t = __m128i;
using int32x4_t = __m128i;
using int64x2_t = __m128i;

// LOAD WIDE
template <typename T>
    requires IsAnyOf<T, uint8_t, int16_t, int32_t, int64_t>
inline auto _load_wide_int(T *p) -> __m128i {
    return _mm_load_si128(reinterpret_cast<__m128i *>(p));    // NOLINT(*-reinterpret-cast)
}
template <typename T>
    requires IsAnyOf<T, uint8_t, int16_t, int32_t, int64_t>
inline auto _load_wide_int(const T *p) -> __m128i {
    return _mm_load_si128(reinterpret_cast<const __m128i *>(p));    // NOLINT(*-reinterpret-cast)
}

#define vld1q_f32 _mm_load_ps
#define vld1q_f64 _mm_load_pd
#define vld1q_u8  _load_wide_int
#define vld1q_s16 _load_wide_int
#define vld1q_s32 _load_wide_int
#define vld1q_s64 _load_wide_int

// LOAD SINGLE
inline auto _load_single_int(uint8_t value) -> __m128i {
    return _mm_set1_epi8(static_cast<char>(value));
}

#define vdupq_n_f32 _mm_set1_ps
#define vdupq_n_f64 _mm_set1_pd
#define vdupq_n_u8  _load_single_int
#define vdupq_n_s16 _mm_set1_epi16
#define vdupq_n_s32 _mm_set1_epi32
#define vdupq_n_s64 _mm_set1_epi64x

// STORE
template <typename T>
    requires IsAnyOf<T, uint8_t, int16_t, int32_t, int64_t>
void _store_int(T *p, __m128i value) {
    _mm_store_si128(reinterpret_cast<__m128i *>(p), value);    // NOLINT(*-reinterpret-cast)
}

#define vst1q_f32 _mm_store_ps
#define vst1q_f64 _mm_store_pd
#define vst1q_u8  _store_int
#define vst1q_s16 _store_int
#define vst1q_s32 _store_int
#define vst1q_s64 _store_int

// ADD
#define vaddq_f32 _mm_add_ps
#define vaddq_f64 _mm_add_pd
#define vaddq_u8  _mm_add_epi8
#define vaddq_s16 _mm_add_epi16
#define vaddq_s32 _mm_add_epi32
#define vaddq_s64 _mm_add_epi64

// MUL
inline auto _mul_u8(__m128i lhs, __m128i rhs) -> __m128i {
    // unpack and multiply
    __m128i dst_even = _mm_mullo_epi16(lhs, rhs);
    __m128i dst_odd = _mm_mullo_epi16(_mm_srli_epi16(lhs, 8), _mm_srli_epi16(rhs, 8));
    return _mm_or_si128(_mm_slli_epi16(dst_odd, 8), _mm_srli_epi16(_mm_slli_epi16(dst_even, 8), 8));
}

inline auto _mul_s64(__m128i lhs, __m128i rhs) -> __m128i {
    __m128i bswap = _mm_shuffle_epi32(rhs, 0xB1);          // b0H,b0L,b1H,b1L (swap H<->L)
    __m128i prodlh = _mm_mullo_epi32(lhs, bswap);          // a0Lb0H,a0Hb0L,a1Lb1H,a1Hb1L, 32 bit L*H products
    __m128i zero = _mm_setzero_si128();                    // 0
    __m128i prodlh2 = _mm_hadd_epi32(prodlh, zero);        // a0Lb0H+a0Hb0L,a1Lb1H+a1Hb1L,0,0
    __m128i prodlh3 = _mm_shuffle_epi32(prodlh2, 0x73);    // 0, a0Lb0H+a0Hb0L, 0, a1Lb1H+a1Hb1L
    __m128i prodll = _mm_mul_epu32(lhs, rhs);              // a0Lb0L,a1Lb1L, 64 bit unsigned products
    __m128i prod = _mm_add_epi64(prodll, prodlh3);         // a0Lb0L+(a0Lb0H+a0Hb0L)<<32, a1Lb1L+(a1Lb1H+a1Hb1L)<<32
    return prod;
}

#define vmulq_f32 _mm_mul_ps
#define vmulq_f64 _mm_mul_pd
#define vmulq_u8  _mul_u8
#define vmulq_s16 _mm_mullo_epi16
#define vmulq_s32 _mm_mullo_epi32
#define vmulq_s64 _mul_s64
#endif

// Underlying types
template <typename T>
struct wide_type;

template <>
struct wide_type<float> {
    using type = float32x4_t;
    static auto add(float32x4_t lhs, float32x4_t rhs) -> float32x4_t {
        return vaddq_f32(lhs, rhs);
    }
    static auto mul(float32x4_t lhs, float32x4_t rhs) -> float32x4_t {
        return vmulq_f32(lhs, rhs);
    }
    static inline auto load_wide(float *p) -> float32x4_t {
        return vld1q_f32(p);
    }
    static inline auto load_wide(const float *p) -> float32x4_t {
        return vld1q_f32(p);
    }
    static inline auto load_single(float value) -> float32x4_t {
        return vdupq_n_f32(value);
    }
    static void store(float *p, float32x4_t value) {
        return vst1q_f32(p, value);
    }
};
template <>
struct wide_type<double> {
    using type = float64x2_t;
    static auto add(float64x2_t lhs, float64x2_t rhs) -> float64x2_t {
        return vaddq_f64(lhs, rhs);
    }
    static auto mul(float64x2_t lhs, float64x2_t rhs) -> float64x2_t {
        return vmulq_f64(lhs, rhs);
    }
    static inline auto load_wide(double *p) -> float64x2_t {
        return vld1q_f64(p);
    }
    static inline auto load_wide(const double *p) -> float64x2_t {
        return vld1q_f64(p);
    }
    static inline auto load_single(double value) -> float64x2_t {
        return vdupq_n_f64(value);
    }
    static void store(double *p, float64x2_t value) {
        return vst1q_f64(p, value);
    }
};
template <>
struct wide_type<uint8_t> {
    using type = uint8x16_t;
    static auto add(uint8x16_t lhs, uint8x16_t rhs) -> uint8x16_t {
        return vaddq_u8(lhs, rhs);
    }
    static auto mul(uint8x16_t lhs, uint8x16_t rhs) -> uint8x16_t {
        return vmulq_u8(lhs, rhs);
    }
    static inline auto load_wide(uint8_t *p) -> uint8x16_t {
        return vld1q_u8(p);
    }
    static inline auto load_wide(const uint8_t *p) -> uint8x16_t {
        return vld1q_u8(p);
    }
    static inline auto load_single(uint8_t value) -> uint8x16_t {
        return vdupq_n_u8(value);
    }
    static void store(uint8_t *p, uint8x16_t value) {
        return vst1q_u8(p, value);
    }
};
template <>
struct wide_type<int16_t> {
    using type = int16x8_t;
    static auto add(int16x8_t lhs, int16x8_t rhs) -> int16x8_t {
        return vaddq_s16(lhs, rhs);
    }
    static auto mul(int16x8_t lhs, int16x8_t rhs) -> int16x8_t {
        return vmulq_s16(lhs, rhs);
    }
    static inline auto load_wide(int16_t *p) -> int16x8_t {
        return vld1q_s16(p);
    }
    static inline auto load_wide(const int16_t *p) -> int16x8_t {
        return vld1q_s16(p);
    }
    static inline auto load_single(int16_t value) -> int16x8_t {
        return vdupq_n_s16(value);
    }
    static void store(int16_t *p, int16x8_t value) {
        return vst1q_s16(p, value);
    }
};
template <>
struct wide_type<int32_t> {
    using type = int32x4_t;
    static auto add(int32x4_t lhs, int32x4_t rhs) -> int32x4_t {
        return vaddq_s32(lhs, rhs);
    }
    static auto mul(int32x4_t lhs, int32x4_t rhs) -> int32x4_t {
        return vmulq_s32(lhs, rhs);
    }
    static inline auto load_wide(int32_t *p) -> int32x4_t {
        return vld1q_s32(p);
    }
    static inline auto load_wide(const int32_t *p) -> int32x4_t {
        return vld1q_s32(p);
    }
    static inline auto load_single(int32_t value) -> int32x4_t {
        return vdupq_n_s32(value);
    }
    static void store(int32_t *p, int32x4_t value) {
        return vst1q_s32(p, value);
    }
};
template <>
struct wide_type<int64_t> {
    using type = int64x2_t;
    static auto add(int64x2_t lhs, int64x2_t rhs) -> int64x2_t {
        return vaddq_s64(lhs, rhs);
    }
    static auto mul(int64x2_t lhs, int64x2_t rhs) -> int64x2_t {
        return vmulq_s64(lhs, rhs);
    }
    static inline auto load_wide(int64_t *p) -> int64x2_t {
        return vld1q_s64(p);
    }
    static inline auto load_wide(const int64_t *p) -> int64x2_t {
        return vld1q_s64(p);
    }
    static inline auto load_single(int64_t value) -> int64x2_t {
        return vdupq_n_s64(value);
    }
    static void store(int64_t *p, int64x2_t value) {
        return vst1q_s64(p, value);
    }
};

template <typename T>
using wide_type_t = wide_type<T>::type;

}    // namespace tinytensor::cpu::kernel

#endif    // TINYTENSOR_BACKEND_CPU_KERNEL_SIMD_H_
