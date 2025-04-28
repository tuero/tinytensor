// test_shape.cpp
// Test shape related methods

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
#include <tuple>
#include <vector>

using namespace tinytensor;

// NOLINTNEXTLINE
TEST_CASE("Shape Squeeze") {
    auto test_squeeze1 = []<typename T>(Device device) {
        Tensor a = arange({1, 4, 3}, to_scalar<T>::type, device);
        Tensor expected = arange({4, 3}, to_scalar<T>::type, device);
        CHECK(allclose(a.squeeze(0), expected));
        CHECK(allclose(squeeze(a, 0), expected));
        CHECK(allclose(a.squeeze(-3), expected));
        CHECK(allclose(squeeze(a, -3), expected));
        CHECK_EQ(a.squeeze(0).data_ptr(), a.data_ptr());
        CHECK_EQ(squeeze(a, 0).data_ptr(), a.data_ptr());
        CHECK_EQ(a.squeeze(-3).data_ptr(), a.data_ptr());
        CHECK_EQ(squeeze(a, -3).data_ptr(), a.data_ptr());
    };
    runner_all_except_bool(test_squeeze1);

    auto test_squeeze2 = []<typename T>(Device device) {
        Tensor a = arange({4, 1, 3}, to_scalar<T>::type, device);
        Tensor expected = arange({4, 3}, to_scalar<T>::type, device);
        CHECK(allclose(a.squeeze(1), expected));
        CHECK(allclose(squeeze(a, 1), expected));
        CHECK(allclose(a.squeeze(-2), expected));
        CHECK(allclose(squeeze(a, -2), expected));
        CHECK_EQ(a.squeeze(1).data_ptr(), a.data_ptr());
        CHECK_EQ(squeeze(a, 1).data_ptr(), a.data_ptr());
        CHECK_EQ(a.squeeze(-2).data_ptr(), a.data_ptr());
        CHECK_EQ(squeeze(a, -2).data_ptr(), a.data_ptr());
    };
    runner_all_except_bool(test_squeeze2);

    auto test_squeeze3 = []<typename T>(Device device) {
        Tensor a = arange({4, 3, 1}, to_scalar<T>::type, device);
        Tensor expected = arange({4, 3}, to_scalar<T>::type, device);
        CHECK(allclose(a.squeeze(2), expected));
        CHECK(allclose(squeeze(a, 2), expected));
        CHECK(allclose(a.squeeze(-1), expected));
        CHECK(allclose(squeeze(a, -1), expected));
        CHECK_EQ(a.squeeze(2).data_ptr(), a.data_ptr());
        CHECK_EQ(squeeze(a, 2).data_ptr(), a.data_ptr());
        CHECK_EQ(a.squeeze(-1).data_ptr(), a.data_ptr());
        CHECK_EQ(squeeze(a, -1).data_ptr(), a.data_ptr());
    };
    runner_all_except_bool(test_squeeze3);

    auto test_squeeze4 = []<typename T>(Device device) {
        std::vector<T> data;
        for (int i = 0; i < 12; ++i) {
            data.push_back(static_cast<T>(i));
        }
        Tensor a = arange({4, 3}, to_scalar<T>::type, device);
        Tensor expected = arange({4, 3}, to_scalar<T>::type, device);
        CHECK(allclose(a.squeeze(0), expected));
        CHECK(allclose(squeeze(a, 0), expected));
    };
    runner_all_except_bool(test_squeeze4);

    auto test_squeeze_bounds = []<typename T>(Device device) {
        Tensor a = zeros({1, 4, 3}, to_scalar<T>::type, device);
        CHECK_THROWS_AS(std::ignore = a.squeeze(3), TTException);
        CHECK_THROWS_AS(std::ignore = squeeze(a, 3), TTException);
        CHECK_THROWS_AS(std::ignore = a.squeeze(-4), TTException);
        CHECK_THROWS_AS(std::ignore = squeeze(a, -4), TTException);
    };
    runner_all_except_bool(test_squeeze_bounds);
}

// NOLINTNEXTLINE
TEST_CASE("Shape Unsqueeze") {
    auto test_unsqueeze1 = []<typename T>(Device device) {
        Tensor a = arange({4, 3}, to_scalar<T>::type, device);
        Tensor expected = arange({1, 4, 3}, to_scalar<T>::type, device);
        CHECK(allclose(a.unsqueeze(0), expected));
        CHECK(allclose(unsqueeze(a, 0), expected));
        CHECK(allclose(a.unsqueeze(-3), expected));
        CHECK(allclose(unsqueeze(a, -3), expected));
        CHECK_EQ(a.unsqueeze(0).data_ptr(), a.data_ptr());
        CHECK_EQ(unsqueeze(a, 0).data_ptr(), a.data_ptr());
        CHECK_EQ(a.unsqueeze(-3).data_ptr(), a.data_ptr());
        CHECK_EQ(unsqueeze(a, -3).data_ptr(), a.data_ptr());
    };
    runner_all_except_bool(test_unsqueeze1);

    auto test_unsqueeze2 = []<typename T>(Device device) {
        Tensor a = arange({4, 3}, to_scalar<T>::type, device);
        Tensor expected = arange({4, 1, 3}, to_scalar<T>::type, device);
        CHECK(allclose(a.unsqueeze(1), expected));
        CHECK(allclose(unsqueeze(a, 1), expected));
        CHECK(allclose(a.unsqueeze(-2), expected));
        CHECK(allclose(unsqueeze(a, -2), expected));
        CHECK_EQ(a.unsqueeze(1).data_ptr(), a.data_ptr());
        CHECK_EQ(unsqueeze(a, 1).data_ptr(), a.data_ptr());
        CHECK_EQ(a.unsqueeze(-2).data_ptr(), a.data_ptr());
        CHECK_EQ(unsqueeze(a, -2).data_ptr(), a.data_ptr());
    };
    runner_all_except_bool(test_unsqueeze2);

    auto test_unsqueeze3 = []<typename T>(Device device) {
        Tensor a = arange({4, 3}, to_scalar<T>::type, device);
        Tensor expected = arange({4, 3, 1}, to_scalar<T>::type, device);
        CHECK(allclose(a.unsqueeze(2), expected));
        CHECK(allclose(unsqueeze(a, 2), expected));
        CHECK(allclose(a.unsqueeze(-1), expected));
        CHECK(allclose(unsqueeze(a, -1), expected));
        CHECK_EQ(a.unsqueeze(2).data_ptr(), a.data_ptr());
        CHECK_EQ(unsqueeze(a, 2).data_ptr(), a.data_ptr());
        CHECK_EQ(a.unsqueeze(-1).data_ptr(), a.data_ptr());
        CHECK_EQ(unsqueeze(a, -1).data_ptr(), a.data_ptr());
    };
    runner_all_except_bool(test_unsqueeze3);

    auto test_unsqueeze_bounds = []<typename T>(Device device) {
        Tensor a = zeros({4, 3}, to_scalar<T>::type, device);
        CHECK_THROWS_AS(std::ignore = a.unsqueeze(3), TTException);
        CHECK_THROWS_AS(std::ignore = unsqueeze(a, 3), TTException);
        CHECK_THROWS_AS(std::ignore = a.unsqueeze(-4), TTException);
        CHECK_THROWS_AS(std::ignore = unsqueeze(a, -4), TTException);
    };
    runner_all_except_bool(test_unsqueeze_bounds);
}

// NOLINTNEXTLINE
TEST_CASE("Shape Broadcast") {
    auto test_broadcast1 = []<typename T>(Device device) {
        std::vector<T> data_expected;
        for (int i = 0; i < 12; ++i) {
            data_expected.push_back(static_cast<T>(i));
        }
        for (int i = 0; i < 12; ++i) {
            data_expected.push_back(static_cast<T>(i));
        }
        Tensor a = arange({1, 4, 3}, to_scalar<T>::type, device);
        Tensor expected(data_expected, {2, 4, 3}, device);
        CHECK(allclose(a.broadcast_to({2, 4, 3}), expected));
        CHECK(allclose(broadcast_to(a, {2, 4, 3}), expected));
        CHECK(allclose(a.expand({2, 4, 3}), expected));
        CHECK(allclose(expand(a, {2, 4, 3}), expected));
        CHECK_EQ(a.broadcast_to({2, 4, 3}).data_ptr(), a.data_ptr());
        CHECK_EQ(broadcast_to(a, {2, 4, 3}).data_ptr(), a.data_ptr());
        CHECK_EQ(a.expand({2, 4, 3}).data_ptr(), a.data_ptr());
        CHECK_EQ(expand(a, {2, 4, 3}).data_ptr(), a.data_ptr());
    };
    runner_all_except_bool(test_broadcast1);

    auto test_broadcast2 = []<typename T>(Device device) {
        std::vector<T> data_expected = {0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5, 6, 7, 8, 6, 7, 8, 9, 10, 11, 9, 10, 11};
        Tensor a = arange({4, 1, 3}, to_scalar<T>::type, device);
        Tensor expected(data_expected, {4, 2, 3}, device);
        CHECK(allclose(a.broadcast_to({4, 2, 3}), expected));
        CHECK(allclose(broadcast_to(a, {4, 2, 3}), expected));
        CHECK(allclose(a.expand({4, 2, 3}), expected));
        CHECK(allclose(expand(a, {4, 2, 3}), expected));
        CHECK_EQ(a.broadcast_to({4, 2, 3}).data_ptr(), a.data_ptr());
        CHECK_EQ(broadcast_to(a, {4, 2, 3}).data_ptr(), a.data_ptr());
        CHECK_EQ(a.expand({4, 2, 3}).data_ptr(), a.data_ptr());
        CHECK_EQ(expand(a, {4, 2, 3}).data_ptr(), a.data_ptr());
    };
    runner_all_except_bool(test_broadcast2);

    auto test_broadcast3 = []<typename T>(Device device) {
        std::vector<T> data_expected = {0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11};
        Tensor a = arange({4, 3, 1}, to_scalar<T>::type, device);
        Tensor expected(data_expected, {4, 3, 2}, device);
        CHECK(allclose(a.broadcast_to({4, 3, 2}), expected));
        CHECK(allclose(broadcast_to(a, {4, 3, 2}), expected));
        CHECK(allclose(a.expand({4, 3, 2}), expected));
        CHECK(allclose(expand(a, {4, 3, 2}), expected));
    };
    runner_all_except_bool(test_broadcast3);

    auto test_broadcast_bounds1 = []<typename T>(Device device) {
        Tensor a = zeros({4, 3, 1}, to_scalar<T>::type, device);
        CHECK_THROWS_AS(std::ignore = a.broadcast_to({4, 2}), TTException);
        CHECK_THROWS_AS(std::ignore = broadcast_to(a, {4, 2}), TTException);
        CHECK_THROWS_AS(std::ignore = a.expand({4, 2}), TTException);
        CHECK_THROWS_AS(std::ignore = expand(a, {4, 2}), TTException);
    };
    runner_all_except_bool(test_broadcast_bounds1);

    auto test_broadcast_bounds2 = []<typename T>(Device device) {
        Tensor a = zeros({4, 3, 1}, to_scalar<T>::type, device);
        CHECK_THROWS_AS(std::ignore = a.broadcast_to({4, 2, 2}), TTException);
        CHECK_THROWS_AS(std::ignore = broadcast_to(a, {4, 2, 2}), TTException);
        CHECK_THROWS_AS(std::ignore = a.expand({4, 2, 2}), TTException);
        CHECK_THROWS_AS(std::ignore = expand(a, {4, 2, 2}), TTException);
    };
    runner_all_except_bool(test_broadcast_bounds2);

    CHECK(can_broadcast_to(Shape({3, 1, 2}), Shape({4, 3, 2, 2})));
    CHECK_FALSE(can_broadcast_to(Shape({3, 1, 2}), Shape({4, 3, 2, 1})));
    CHECK_FALSE(can_broadcast_to(Shape({3, 1, 2}), Shape({4, 4, 2, 2})));

    CHECK(are_broadcastable(Shape({5, 7, 3}), Shape({5, 7, 3})));
    CHECK(are_broadcastable(Shape({5, 3, 4, 1}), Shape({3, 1, 1})));
    CHECK_FALSE(are_broadcastable(Shape({5, 2, 4, 1}), Shape({3, 1, 1})));

    CHECK_EQ(broadcast_result_shape(Shape({5, 1, 4, 1}), Shape({3, 1, 1})), Shape({5, 3, 4, 1}));
    CHECK_EQ(broadcast_result_shape(Shape({1}), Shape({3, 1, 7})), Shape({3, 1, 7}));
    CHECK_THROWS_AS(std::ignore = broadcast_result_shape(Shape({5, 2, 4, 1}), Shape({3, 1, 1})), TTException);
}

// NOLINTNEXTLINE
TEST_CASE("Shape Reshape") {
    auto test_reshape_more = []<typename T>(Device device) {
        Tensor a = arange({4, 3}, to_scalar<T>::type, device);
        Tensor expected = arange({1, 4, 3}, to_scalar<T>::type, device);
        CHECK(allclose(a.reshape({1, 4, 3}), expected));
        CHECK(allclose(reshape(a, {1, 4, 3}), expected));
        CHECK_EQ(a.reshape({1, 4, 3}).data_ptr(), a.data_ptr());
        CHECK_EQ(reshape(a, {1, 4, 3}).data_ptr(), a.data_ptr());
    };
    runner_all_except_bool(test_reshape_more);

    auto test_reshape_less = []<typename T>(Device device) {
        Tensor a = arange({4, 1, 3}, to_scalar<T>::type, device);
        Tensor expected = arange({12}, to_scalar<T>::type, device);
        CHECK(allclose(a.reshape({12}), expected));
        CHECK(allclose(reshape(a, {12}), expected));
        CHECK_EQ(a.reshape({12}).data_ptr(), a.data_ptr());
        CHECK_EQ(reshape(a, {12}).data_ptr(), a.data_ptr());
    };
    runner_all_except_bool(test_reshape_less);

    auto test_reshape_error = []<typename T>(Device device) {
        Tensor a = zeros({4, 1, 3}, to_scalar<T>::type, device);
        CHECK_THROWS_AS(std::ignore = a.reshape({4, 2, 2}), TTException);
        CHECK_THROWS_AS(std::ignore = reshape(a, {4, 2, 2}), TTException);
        CHECK_THROWS_AS(std::ignore = a.reshape({4, 2, 2}), TTException);
        CHECK_THROWS_AS(std::ignore = reshape(a, {4, 2, 2}), TTException);
    };
    runner_all_except_bool(test_reshape_error);
}

// NOLINTNEXTLINE
TEST_CASE("Shape Flatten") {
    auto test_flatten1 = []<typename T>(Device device) {
        Tensor a = arange({4, 3}, to_scalar<T>::type, device);
        Tensor expected = arange({12}, to_scalar<T>::type, device);
        CHECK(allclose(a.flatten(), expected));
        CHECK(allclose(flatten(a), expected));
        CHECK_EQ(a.flatten().data_ptr(), a.data_ptr());
        CHECK_EQ(flatten(a).data_ptr(), a.data_ptr());
    };
    runner_all_except_bool(test_flatten1);

    auto test_flatten2 = []<typename T>(Device device) {
        Tensor a = arange({12}, to_scalar<T>::type, device);
        Tensor expected = arange({12}, to_scalar<T>::type, device);
        CHECK(allclose(a.flatten(), expected));
        CHECK(allclose(flatten(a), expected));
        CHECK_EQ(a.flatten().data_ptr(), a.data_ptr());
        CHECK_EQ(flatten(a).data_ptr(), a.data_ptr());
    };
    runner_all_except_bool(test_flatten2);
}

// NOLINTNEXTLINE
TEST_CASE("Shape permute") {
    auto test_permute = []<typename T>(Device device) {
        Tensor a = arange({3, 1, 2}, to_scalar<T>::type, device);
        std::vector<T> data = {0, 2, 4, 1, 3, 5};
        Tensor expected(data, {1, 2, 3}, device);
        CHECK(allclose(a.permute({1, 2, 0}), expected));
        CHECK(allclose(permute(a, {1, 2, 0}), expected));
        CHECK_EQ(a.permute({1, 2, 0}).data_ptr(), a.data_ptr());
        CHECK_EQ(permute(a, {1, 2, 0}).data_ptr(), a.data_ptr());
    };
    runner_all_except_bool(test_permute);
}

// NOLINTNEXTLINE
TEST_CASE("Shape Multi") {
    auto test_multi = []<typename T>(Device device) {
        Tensor a = arange({2, 3, 4}, to_scalar<T>::type, device);
        CHECK_EQ(a.stride(), Shape({12, 4, 1}));
        CHECK_EQ(a.offset(), 0);
        CHECK(a.is_contiguous());

        Tensor b = permute(a, {1, 2, 0});
        b = b[{indexing::Slice(), 1}];
        CHECK_EQ(b.shape(), Shape({3, 2}));
        CHECK_EQ(b.stride(), Shape({4, 12}));
        CHECK_EQ(b.offset(), 1);
        CHECK_FALSE(b.is_contiguous());
        CHECK_EQ(a.data_ptr(), b.data_ptr());

        // Reshape on non-contiguous needs to copy
        Tensor c = b.unsqueeze(0);
        CHECK_EQ(c.shape(), Shape({1, 3, 2}));
        CHECK_EQ(c.stride(), Shape({6, 2, 1}));
        CHECK_EQ(c.offset(), 0);
        CHECK(c.is_contiguous());
        CHECK_NE(a.data_ptr(), c.data_ptr());

        // Reshape on non-contiguous needs to copy
        Tensor d = b.unsqueeze(1);
        CHECK_EQ(d.shape(), Shape({3, 1, 2}));
        CHECK_EQ(d.stride(), Shape({2, 2, 1}));
        CHECK_EQ(d.offset(), 0);
        CHECK(d.is_contiguous());
        CHECK_NE(a.data_ptr(), d.data_ptr());

        // Reshape on non-contiguous needs to copy
        Tensor e = b.unsqueeze(2);
        CHECK_EQ(e.shape(), Shape({3, 2, 1}));
        CHECK_EQ(e.stride(), Shape({2, 1, 1}));
        CHECK_EQ(e.offset(), 0);
        CHECK(e.is_contiguous());
        CHECK_NE(a.data_ptr(), e.data_ptr());

        Tensor f = d.expand({4, 3, 2, 2});
        CHECK_EQ(f.shape(), Shape({4, 3, 2, 2}));
        CHECK_EQ(f.stride(), Shape({0, 2, 0, 1}));
        CHECK_EQ(f.offset(), 0);
        CHECK_FALSE(f.is_contiguous());
        CHECK_EQ(d.data_ptr(), f.data_ptr());

        Tensor g = f.reshape({12, 4});
        CHECK_EQ(g.shape(), Shape({12, 4}));
        CHECK_EQ(g.stride(), Shape({4, 1}));
        CHECK_EQ(g.offset(), 0);
        CHECK(g.is_contiguous());
        CHECK_NE(g.data_ptr(), f.data_ptr());
    };
    runner_all_except_bool(test_multi);
}
