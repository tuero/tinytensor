// random.cpp
// RNG for distributions

#include <tt/macros.h>
#include <tt/random.h>

#include "backend/common/kernel/distribution.hpp"

#include <cstddef>
#include <cstdint>
#include <ranges>
#include <utility>
#include <vector>

namespace tinytensor {

namespace {
// https://en.wikipedia.org/wiki/Xorshift
// Portable RNG Seed
constexpr uint64_t SPLIT64_S1 = 30;
constexpr uint64_t SPLIT64_S2 = 27;
constexpr uint64_t SPLIT64_S3 = 31;
constexpr uint64_t SPLIT64_C1 = 0x9E3779B97f4A7C15;
constexpr uint64_t SPLIT64_C2 = 0xBF58476D1CE4E5B9;
constexpr uint64_t SPLIT64_C3 = 0x94D049BB133111EB;
TT_HOST_DEVICE auto splitmix64(uint64_t seed) noexcept -> uint64_t {
    uint64_t result = seed + SPLIT64_C1;
    result = (result ^ (result >> SPLIT64_S1)) * SPLIT64_C2;
    result = (result ^ (result >> SPLIT64_S2)) * SPLIT64_C3;
    return result ^ (result >> SPLIT64_S3);
}

constexpr uint64_t XOR64_S1 = 13;
constexpr uint64_t XOR64_S2 = 7;
constexpr uint64_t XOR64_S3 = 17;
TT_HOST_DEVICE auto xorshift64(uint64_t &state) noexcept -> uint64_t {
    uint64_t x = state;
    x ^= x << XOR64_S1;
    x ^= x >> XOR64_S2;
    x ^= x << XOR64_S3;
    state = x;
    return x;
}
}    // namespace

// Generator
TT_HOST_DEVICE Generator::Generator(uint64_t s, bool is_state)
    : state(is_state ? s : splitmix64(s)) {}

TT_HOST_DEVICE Generator::Generator(uint64_t seed)
    : Generator(seed, false) {}

TT_HOST_DEVICE auto Generator::from_state(uint64_t _state) -> Generator {
    return {_state, true};
}

TT_HOST_DEVICE auto Generator::operator()() noexcept -> uint64_t {
    return xorshift64(state);
}

TT_HOST_DEVICE void Generator::set_state(uint64_t _state) {
    state = _state;
}

class GeneratorSingleton {
public:
    GeneratorSingleton() = delete;

    static auto get_generator() -> Generator & {
        static thread_local Generator gen(default_seed);
        return gen;
    }

    // @NOTE: Not thread safe
    static void set_seed(uint64_t seed) {
        default_seed = seed;
        get_generator().set_state(splitmix64(default_seed));
    }

private:
    static inline uint64_t default_seed = 0;
};

void set_default_generator_seed(uint64_t seed) {
    GeneratorSingleton::set_seed(seed);
}

auto get_default_generator() -> Generator & {
    return GeneratorSingleton::get_generator();
}

void shuffle(std::vector<int> &data, Generator &gen) {
    for (std::size_t i : std::views::iota(static_cast<std::size_t>(0), data.size())) {
        auto j =
            static_cast<std::size_t>(common::kernel::distribution::rand_in_range(static_cast<uint64_t>(i + 1), gen));
        std::swap(data[i], data[j]);
    }
}

}    // namespace tinytensor
