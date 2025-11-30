// random.h
// RNG for distributions

#ifndef TINYTENSOR_RANDOM_H_
#define TINYTENSOR_RANDOM_H_

#include <tt/export.h>
#include <tt/macros.h>

#include <cstdint>
#include <vector>

namespace tinytensor {

// RNG object for all distribution functions
//
class TINYTENSOR_EXPORT Generator {
public:
    /**
     * Create a Generator from a seed
     * @param seed The seed
     */
    TT_HOST_DEVICE Generator(uint64_t seed);

    /**
     * Factory to create a generator from a state
     */
    TT_HOST_DEVICE static auto from_state(uint64_t state) -> Generator;

    /**
     * Generate a random set of 64 bits
     */
    TT_HOST_DEVICE auto operator()() noexcept -> uint64_t;

    /**
     * Set the state of the generator object
     */
    TT_HOST_DEVICE void set_state(uint64_t state);

private:
    TT_HOST_DEVICE Generator(uint64_t s, bool is_state);

    uint64_t state;
};

TINYTENSOR_EXPORT void shuffle(std::vector<int> &data, Generator &gen);

/**
 * Set the default generator state using the passed seed
 * @param seed The seed
 */
TINYTENSOR_EXPORT void set_default_generator_seed(uint64_t seed);

/**
 * Get the default generator.
 * For reproducibility, its recomended to not rely on this and create your own Generator
 */
TINYTENSOR_EXPORT auto get_default_generator() -> Generator &;

}    // namespace tinytensor

#endif    // TINYTENSOR_RANDOM_H_
