// test_clone.cpp
// Test clone related methods

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
TEST_CASE("Clone") {
    auto test_clone_bool = []<typename T>(Device device) {
        Tensor a = (arange({2, 3, 4}, kI32, device) % 2).to(kBool);
        Tensor b = permute(a, {1, 2, 0});
        b = b[{indexing::Slice(), 1}];
        b = b.unsqueeze(1);
        Tensor c = b.clone();
        std::vector<bool> res = {true, true, true, true, true, true};
        Tensor expected(res, {3, 1, 2}, device);
        CHECK(allclose(c, expected));
        CHECK_NE(c.data_ptr(), b.data_ptr());
    };
    runner_boolean(test_clone_bool);

    auto test_clone = []<typename T>(Device device) {
        Tensor a = arange({2, 3, 4}, to_scalar<T>::type, device);
        Tensor b = permute(a, {1, 2, 0});
        b = b[{indexing::Slice(), 1}];
        b = b.unsqueeze(1);
        Tensor c = b.clone();
        std::vector<T> res = {1, 13, 5, 17, 9, 21};
        Tensor expected(res, {3, 1, 2}, device);
        CHECK(all(isclose(c, expected)));
        CHECK_NE(c.data_ptr(), b.data_ptr());
    };
    runner_all_except_bool(test_clone);
}
