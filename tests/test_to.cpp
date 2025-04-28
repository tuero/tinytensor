// test_to.cpp
// Test to related methods

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <tt/device.h>
#include <tt/exception.h>
#include <tt/index.h>
#include <tt/scalar.h>
#include <tt/shape.h>
#include <tt/tensor.h>

#include "doctest.h"
#include "test_util.h"

#include <cmath>
#include <vector>

using namespace tinytensor;

// NOLINTNEXTLINE
TEST_CASE("To") {
    auto test_to1 = []<typename T>(Device device) {
        Tensor a = arange({2, 3, 4}, to_scalar<T>::type, device);
        Tensor b = permute(a, {1, 2, 0});
        b = b[{indexing::Slice(), 1}];
        b = b.unsqueeze(1);
        Tensor c = b.to(to_scalar<int>::type);
        std::vector<int> res = {1, 13, 5, 17, 9, 21};
        Tensor expected(res, {3, 1, 2}, device);
        CHECK(allclose(c, expected));
        CHECK_NE(c.data_ptr(), b.data_ptr());
    };
    runner_floating_point(test_to1);

    auto test_to2 = []<typename T>(Device device) {
        Tensor a = arange({2, 3, 4}, to_scalar<T>::type, device);
        Tensor b = permute(a, {1, 2, 0});
        b = b[{indexing::Slice(), 1}];
        b = b.unsqueeze(1);
        Tensor c = b.to(to_scalar<float>::type);
        std::vector<float> res = {1, 13, 5, 17, 9, 21};
        Tensor expected(res, {3, 1, 2}, device);
        CHECK(allclose(c, expected));
        CHECK_NE(c.data_ptr(), b.data_ptr());
    };
    runner_integral(test_to2);

    auto test_to_same = []<typename T>(Device device) {
        Tensor a = full(static_cast<T>(0), {2, 3, 4}, to_scalar<T>::type, device);
        Tensor b = permute(a, {1, 2, 0});
        b = b[{indexing::Slice(), 1}];
        b = b.unsqueeze(1);
        Tensor c = b.to(to_scalar<T>::type);
        std::vector<T> res(6, static_cast<T>(0));
        Tensor expected(res, {3, 1, 2}, device);
        CHECK(allclose(c, expected));
        CHECK_EQ(c.data_ptr(), b.data_ptr());
    };
    runner_all(test_to_same);
}
