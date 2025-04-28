// test_to_vec.cpp
// Test to_vec related methods

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
TEST_CASE("to_vec") {
    auto test_to_vec_bool = []<typename T>(Device device) {
        Tensor a = arange({2, 3, 4}, kI32, device);
        Tensor b = permute(a, {1, 2, 0});
        b = b[{indexing::Slice(), 1}];
        b = b.unsqueeze(1);
        std::vector<bool> expected(6, true);
        std::vector<bool> res = b.to_vec<bool>();
        CHECK_EQ(expected, res);
    };
    runner_boolean(test_to_vec_bool);

    auto test_to_vec = []<typename T>(Device device) {
        Tensor a = arange({2, 3, 4}, to_scalar<T>::type, device);
        Tensor b = permute(a, {1, 2, 0});
        b = b[{indexing::Slice(), 1}];
        b = b.unsqueeze(1);
        std::vector<T> expected = {1, 13, 5, 17, 9, 21};
        std::vector<T> res = b.to_vec<T>();
        CHECK_EQ(expected, res);
    };
    runner_all_except_bool(test_to_vec);
}
