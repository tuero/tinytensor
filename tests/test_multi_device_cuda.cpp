// test_multi_device_cuda.cpp
// Test multi-device cuda support

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <tt/device.h>
#include <tt/exception.h>
#include <tt/tensor.h>

#include "doctest.h"
#include "test_util.h"

#include <tuple>

using namespace tinytensor;

#ifdef TT_CUDA
// NOLINTNEXTLINE
TEST_CASE("Multi device cuda") {
    if (get_device_count(Backend::cuda) > 1) {
        // Two devices on device 0
        {
            Tensor t1 = ones({4, 4}, TensorOptions().device({.backend = Backend::cuda, .id = 0}));
            Tensor t2 = ones({4, 4}, TensorOptions().device({.backend = Backend::cuda, .id = 0}));
            Tensor result = t1 + t2;
            CHECK(allclose(2 * t1, result));
        }
        // Two devices on device 2
        {
            Tensor t1 = ones({4, 4}, TensorOptions().device({.backend = Backend::cuda, .id = 1}));
            Tensor t2 = ones({4, 4}, TensorOptions().device({.backend = Backend::cuda, .id = 1}));
            Tensor result = t1 + t2;
            CHECK(allclose(2 * t1, result));
        }
        // One on each device, should throw exception
        {
            Tensor t1 = ones({4, 4}, TensorOptions().device({.backend = Backend::cuda, .id = 0}));
            Tensor t2 = ones({4, 4}, TensorOptions().device({.backend = Backend::cuda, .id = 1}));
            CHECK_THROWS_AS(std::ignore = t1 + t2, TTException);
        }
        // One on each device, moving to the same should then work
        {
            const Device cuda0{.backend = Backend::cuda, .id = 0};
            const Device cuda1{.backend = Backend::cuda, .id = 1};
            Tensor t1 = ones({4, 4}, TensorOptions().device(cuda0));
            Tensor t2 = ones({4, 4}, TensorOptions().device(cuda1));
            Tensor t3 = t1.to(cuda1);
            Tensor result = t3 + t2;
            CHECK(allclose(2 * t2, result));
        }
    }
}
#endif
