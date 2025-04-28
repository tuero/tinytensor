// test_inplace_operator.cpp
// Test inplace arithmetic related methods

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
TEST_CASE("operator=") {
    auto test_value = []<typename T>(Device device) {
        Tensor a = arange({2, 3, 4}, to_scalar<T>::type, device);
        Tensor b = permute(a, {1, 2, 0});
        b = b[{indexing::Slice(), 1}];
        b = 20;
        std::vector<T> res_global;
        for (int i = 0; i < 24; ++i) {
            res_global.push_back(static_cast<T>(i));
        }
        res_global[1] = 20;
        res_global[13] = 20;
        res_global[5] = 20;
        res_global[17] = 20;
        res_global[9] = 20;
        res_global[21] = 20;
        Tensor expected_local = full(static_cast<T>(20), {3, 2}, device);
        Tensor expected_global(res_global, {2, 3, 4}, device);
        CHECK(allclose(b, expected_local));
        CHECK(allclose(a, expected_global));
    };
    runner_floating_point(test_value);

    auto test_scalar = []<typename T>(Device device) {
        Tensor a = arange({2, 3, 4}, to_scalar<T>::type, device);
        Tensor b = permute(a, {1, 2, 0});
        b = b[{indexing::Slice(), 1}];
        b = Scalar(20);
        std::vector<T> res_global;
        for (int i = 0; i < 24; ++i) {
            res_global.push_back(static_cast<T>(i));
        }
        res_global[1] = 20;
        res_global[13] = 20;
        res_global[5] = 20;
        res_global[17] = 20;
        res_global[9] = 20;
        res_global[21] = 20;
        Tensor expected_local = full(static_cast<T>(20), {3, 2}, device);
        Tensor expected_global(res_global, {2, 3, 4}, device);
        CHECK(allclose(b, expected_local));
        CHECK(allclose(a, expected_global));
    };
    runner_floating_point(test_scalar);

    auto test_lvalue = []<typename T>(Device device) {
        Tensor a = arange({2, 3, 4}, to_scalar<T>::type, device);
        Tensor expected_global = a.clone();
        Tensor b = permute(a, {1, 2, 0});
        b = b[{indexing::Slice(), 1}];
        b = b.unsqueeze(1);
        b = arange({3, 1, 2}, TensorOptions().device(device).dtype(to_scalar<T>::type));
        Tensor expected_local = arange({3, 1, 2}, TensorOptions().device(device).dtype(to_scalar<T>::type));
        CHECK(allclose(b, expected_local));
        CHECK(allclose(a, expected_global));
    };
    runner_floating_point(test_lvalue);

    auto test_rvalue = []<typename T>(Device device) {
        Tensor a = arange({2, 3, 4}, to_scalar<T>::type, device);
        Tensor b = permute(a, {1, 2, 0});
        b[{indexing::Slice(), 1}] = 20;
        std::vector<T> res_local = {0, 12, 20, 20, 2, 14, 3,  15, 4,  16, 20, 20,
                                    6, 18, 7,  19, 8, 20, 20, 20, 10, 22, 11, 23};
        std::vector<T> res_global = {0,  20, 2,  3,  4,  20, 6,  7,  8,  20, 10, 11,
                                     12, 20, 14, 15, 16, 20, 18, 19, 20, 20, 22, 23};
        Tensor expected_local(res_local, {3, 4, 2}, device);
        Tensor expected_global(res_global, {2, 3, 4}, device);
        CHECK(allclose(b, expected_local));
        CHECK(allclose(a, expected_global));
    };
    runner_floating_point(test_rvalue);
}

// NOLINTNEXTLINE
TEST_CASE("operator+=") {
    auto test_scalar = []<typename T>(Device device) {
        Tensor a = arange({2, 3, 4}, to_scalar<T>::type, device);
        Tensor b = permute(a, {1, 2, 0});
        b = b[{indexing::Slice(), 1}];
        b += 10;
        std::vector<T> res_local = {11, 23, 15, 27, 19, 31};
        std::vector<T> res_global = {0,  11, 2,  3,  4,  15, 6,  7,  8,  19, 10, 11,
                                     12, 23, 14, 15, 16, 27, 18, 19, 20, 31, 22, 23};
        Tensor expected_local(res_local, {3, 2}, device);
        Tensor expected_global(res_global, {2, 3, 4}, device);
        CHECK(allclose(b, expected_local));
        CHECK(allclose(a, expected_global));
    };
    runner_floating_point(test_scalar);

    auto test_array = []<typename T>(Device device) {
        Tensor a = arange({2, 3, 4}, to_scalar<T>::type, device);
        Tensor b = permute(a, {1, 2, 0});
        b = b[{indexing::Slice(), 1}];
        b += arange({3, 2}, to_scalar<T>::type, device);
        std::vector<T> res_local = {1, 13, 5, 17, 9, 21};
        res_local[0] += 0;
        res_local[1] += 1;
        res_local[2] += 2;
        res_local[3] += 3;
        res_local[4] += 4;
        res_local[5] += 5;
        std::vector<T> res_global;
        for (int i = 0; i < 24; ++i) {
            res_global.push_back(static_cast<T>(i));
        }
        res_global[1] += 0;
        res_global[13] += 1;
        res_global[5] += 2;
        res_global[17] += 3;
        res_global[9] += 4;
        res_global[21] += 5;
        Tensor expected_local(res_local, {3, 2}, device);
        Tensor expected_global(res_global, {2, 3, 4}, device);
        CHECK(allclose(b, expected_local));
        CHECK(allclose(a, expected_global));
    };
    runner_floating_point(test_array);
}

// NOLINTNEXTLINE
TEST_CASE("operator-=") {
    auto test_scalar = []<typename T>(Device device) {
        Tensor a = arange({2, 3, 4}, to_scalar<T>::type, device);
        Tensor b = permute(a, {1, 2, 0});
        b = b[{indexing::Slice(), 1}];
        b -= 10;
        std::vector<T> res_local = {-9, 3, -5, 7, -1, 11};
        std::vector<T> res_global = {0,  -9, 2,  3,  4,  -5, 6,  7,  8,  -1, 10, 11,
                                     12, 3,  14, 15, 16, 7,  18, 19, 20, 11, 22, 23};
        Tensor expected_local(res_local, {3, 2}, device);
        Tensor expected_global(res_global, {2, 3, 4}, device);
        CHECK(allclose(b, expected_local));
        CHECK(allclose(a, expected_global));
    };
    runner_floating_point(test_scalar);

    auto test_array = []<typename T>(Device device) {
        Tensor a = arange({2, 3, 4}, to_scalar<T>::type, device);
        Tensor b = permute(a, {1, 2, 0});
        b = b[{indexing::Slice(), 1}];
        b -= arange({3, 2}, to_scalar<T>::type, device);
        std::vector<T> res_local = {1, 13, 5, 17, 9, 21};
        res_local[0] -= 0;
        res_local[1] -= 1;
        res_local[2] -= 2;
        res_local[3] -= 3;
        res_local[4] -= 4;
        res_local[5] -= 5;
        std::vector<T> res_global;
        for (int i = 0; i < 24; ++i) {
            res_global.push_back(static_cast<T>(i));
        }
        res_global[1] -= 0;
        res_global[13] -= 1;
        res_global[5] -= 2;
        res_global[17] -= 3;
        res_global[9] -= 4;
        res_global[21] -= 5;
        Tensor expected_local(res_local, {3, 2}, device);
        Tensor expected_global(res_global, {2, 3, 4}, device);
        CHECK(allclose(b, expected_local));
        CHECK(allclose(a, expected_global));
    };
    runner_floating_point(test_array);
}

// NOLINTNEXTLINE
TEST_CASE("operator*=") {
    auto test_scalar = []<typename T>(Device device) {
        Tensor a = arange({2, 3, 4}, to_scalar<T>::type, device);
        Tensor b = permute(a, {1, 2, 0});
        b = b[{indexing::Slice(), 1}];
        b *= 2;
        std::vector<T> res_local = {2, 26, 10, 34, 18, 42};
        std::vector<T> res_global = {0,  2,  2,  3,  4,  10, 6,  7,  8,  18, 10, 11,
                                     12, 26, 14, 15, 16, 34, 18, 19, 20, 42, 22, 23};
        Tensor expected_local(res_local, {3, 2}, device);
        Tensor expected_global(res_global, {2, 3, 4}, device);
        CHECK(allclose(b, expected_local));
        CHECK(allclose(a, expected_global));
    };
    runner_floating_point(test_scalar);

    auto test_array = []<typename T>(Device device) {
        Tensor a = arange({2, 3, 4}, to_scalar<T>::type, device);
        Tensor b = permute(a, {1, 2, 0});
        b = b[{indexing::Slice(), 1}];
        b *= arange({3, 2}, to_scalar<T>::type, device);
        std::vector<T> res_local = {1, 13, 5, 17, 9, 21};
        res_local[0] *= 0;
        res_local[1] *= 1;
        res_local[2] *= 2;
        res_local[3] *= 3;
        res_local[4] *= 4;
        res_local[5] *= 5;
        std::vector<T> res_global;
        for (int i = 0; i < 24; ++i) {
            res_global.push_back(static_cast<T>(i));
        }
        res_global[1] *= 0;
        res_global[13] *= 1;
        res_global[5] *= 2;
        res_global[17] *= 3;
        res_global[9] *= 4;
        res_global[21] *= 5;
        Tensor expected_local(res_local, {3, 2}, device);
        Tensor expected_global(res_global, {2, 3, 4}, device);
        CHECK(allclose(b, expected_local));
        CHECK(allclose(a, expected_global));
    };
    runner_floating_point(test_array);
}

// NOLINTNEXTLINE
TEST_CASE("operator/=") {
    auto test_scalar = []<typename T>(Device device) {
        Tensor a = arange({2, 3, 4}, to_scalar<T>::type, device);
        Tensor b = permute(a, {1, 2, 0});
        b = b[{indexing::Slice(), 1}];
        b /= 2;
        std::vector<T> res_local = {1, 13, 5, 17, 9, 21};
        for (auto &x : res_local) {
            x /= 2;
        }
        std::vector<T> res_global;
        for (int i = 0; i < 24; ++i) {
            res_global.push_back(static_cast<T>(i));
        }
        res_global[1] /= 2;
        res_global[5] /= 2;
        res_global[9] /= 2;
        res_global[13] /= 2;
        res_global[17] /= 2;
        res_global[21] /= 2;
        Tensor expected_local(res_local, {3, 2}, device);
        Tensor expected_global(res_global, {2, 3, 4}, device);
        CHECK(allclose(b, expected_local));
        CHECK(allclose(a, expected_global));
    };
    runner_floating_point(test_scalar);

    auto test_array = []<typename T>(Device device) {
        Tensor a = arange({2, 3, 4}, to_scalar<T>::type, device);
        Tensor b = permute(a, {1, 2, 0});
        b = b[{indexing::Slice(), 1}];
        b /= (arange({3, 2}, to_scalar<T>::type, device) + 1);
        std::vector<T> res_local = {1, 13, 5, 17, 9, 21};
        res_local[0] /= 1;
        res_local[1] /= 2;
        res_local[2] /= 3;
        res_local[3] /= 4;
        res_local[4] /= 5;
        res_local[5] /= 6;
        std::vector<T> res_global;
        for (int i = 0; i < 24; ++i) {
            res_global.push_back(static_cast<T>(i));
        }
        res_global[1] /= 1;
        res_global[13] /= 2;
        res_global[5] /= 3;
        res_global[17] /= 4;
        res_global[9] /= 5;
        res_global[21] /= 6;
        Tensor expected_local(res_local, {3, 2}, device);
        Tensor expected_global(res_global, {2, 3, 4}, device);
        CHECK(allclose(b, expected_local));
        CHECK(allclose(a, expected_global));
    };
    runner_floating_point(test_array);
}
