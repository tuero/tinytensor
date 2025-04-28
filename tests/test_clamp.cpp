// test_clamp.cpp
// Test clamp related methods

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
TEST_CASE("Clamp") {
    auto test_clamp1 = []<typename T>(Device device) {
        T min = 2;
        T max = 8;
        std::vector<T> v = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
        std::vector<T> res = {2, 2, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8, 8, 8, 8, 8, 7, 6, 5, 4, 3, 2, 2, 2};
        Tensor a(v, {3, 2, 4}, device);
        Tensor expected(res, {3, 2, 4}, device);
        CHECK(allclose(clamp(a, ClampOptions().min(min).max(max)), expected));
    };
    runner_all_except_bool(test_clamp1);

    auto test_clamp2 = []<typename T>(Device device) {
        T min = 2;
        T max = 8;
        std::vector<T> v = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
        std::vector<T> res = {2, 2, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8, 8, 8, 8, 8, 7, 6, 5, 4, 3, 2, 2, 2};
        Tensor a(v, {3, 2, 4}, device);
        const auto options = TensorOptions().dtype(to_scalar<T>::type).device(device);
        Tensor _min = full(min, {3, 2, 4}, options);
        Tensor _max = full(max, {3, 2, 4}, options);
        Tensor expected(res, {3, 2, 4}, device);
        CHECK(allclose(clamp(a, _min, _max), expected));
    };
    runner_all_except_bool(test_clamp2);

    auto test_clamp_inplace1 = []<typename T>(Device device) {
        T min = 2;
        T max = 8;
        std::vector<T> v = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
        std::vector<T> res = {2, 2, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8, 8, 8, 8, 8, 7, 6, 5, 4, 3, 2, 2, 2};
        Tensor a(v, {3, 2, 4}, device);
        Tensor expected(res, {3, 2, 4}, device);
        auto result = a.clamp_(ClampOptions().min(min).max(max));
        CHECK(allclose(result, expected));
        CHECK_EQ(result.data_ptr(), a.data_ptr());
    };
    runner_all_except_bool(test_clamp_inplace1);

    auto test_clamp_inplace2 = []<typename T>(Device device) {
        T min = 2;
        T max = 8;
        std::vector<T> v = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
        std::vector<T> res = {2, 2, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8, 8, 8, 8, 8, 7, 6, 5, 4, 3, 2, 2, 2};
        Tensor a(v, {3, 2, 4}, device);
        const auto options = TensorOptions().dtype(to_scalar<T>::type).device(device);
        Tensor _min = full(min, {3, 2, 4}, options);
        Tensor _max = full(max, {3, 2, 4}, options);
        Tensor expected(res, {3, 2, 4}, device);
        auto result = a.clamp_(_min, _max);
        CHECK(allclose(result, expected));
        CHECK_EQ(result.data_ptr(), a.data_ptr());
    };
    runner_all_except_bool(test_clamp_inplace2);
}
