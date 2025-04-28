// test_misc.cpp
// Test misc related methods

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <tt/device.h>
#include <tt/exception.h>
#include <tt/index.h>
#include <tt/scalar.h>
#include <tt/shape.h>
#include <tt/tensor.h>

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#ifdef TT_TORCH
#include <torch/torch.h>
#undef CHECK
#endif

#include "doctest.h"
#include "test_util.h"

using namespace tinytensor;

// NOLINTNEXTLINE
TEST_CASE("where") {
    auto test_where = []<typename T>(Device device) {
        std::vector<T> lhs_data = {1, 2, 3, 4, 5, 6};
        std::vector<T> rhs_data = {10, 20, 30, 40, 50, 60};
        std::vector<bool> cond_data = {true, false, false, true, false, true};
        std::vector<T> expected_data = {1, 20, 30, 4, 50, 6};

        Tensor lhs(lhs_data, {2, 3}, device);
        Tensor rhs(rhs_data, {2, 3}, device);
        Tensor cond(cond_data, {2, 3}, device);
        Tensor expected(expected_data, {2, 3}, device);

        CHECK(allclose(where(cond, lhs, rhs), expected));
    };
    runner_all_except_bool(test_where);

    auto test_where_scalar = []<typename T>(Device device) {
        std::vector<bool> cond_data = {true, false, false, true, false, true};
        std::vector<T> expected_data = {1, 2, 2, 1, 2, 1};

        Tensor cond(cond_data, {2, 3}, device);
        Tensor expected(expected_data, {2, 3}, device);

        CHECK(allclose(where(cond, static_cast<T>(1), static_cast<T>(2)), expected));
    };
    runner_all_except_bool(test_where_scalar);
}

// NOLINTNEXTLINE
TEST_CASE("gather") {
    auto test_gather0 = []<typename T>(Device device) {
        std::vector<T> input_data = {25, 32, 23, 44, 41, 27, 49, 6,  50, 33, 8,  51, 4,  52, 47, 11, 22, 17, 36, 19,
                                     48, 3,  55, 10, 5,  59, 29, 12, 13, 39, 0,  1,  24, 56, 57, 40, 28, 46, 14, 30,
                                     43, 45, 37, 42, 35, 31, 34, 38, 9,  7,  53, 20, 21, 54, 15, 26, 58, 16, 2,  18};
        std::vector<int> indices_data = {1, 4, 3, 1, 2, 1, 2, 2, 2, 1, 4, 1};
        std::vector<T> expected_data = {4, 7, 14, 11, 13, 17, 0, 1, 24, 3, 2, 10};

        Tensor input(input_data, {5, 4, 3}, device);
        Tensor indices(indices_data, {1, 4, 3}, device);
        Tensor expected(expected_data, {1, 4, 3}, device);

        CHECK(allclose(gather(input, indices, 0), expected));
    };
    runner_all_except_bool(test_gather0);

    auto test_gather1 = []<typename T>(Device device) {
        std::vector<T> input_data = {25, 32, 23, 44, 41, 27, 49, 6,  50, 33, 8,  51, 4,  52, 47, 11, 22, 17, 36, 19,
                                     48, 3,  55, 10, 5,  59, 29, 12, 13, 39, 0,  1,  24, 56, 57, 40, 28, 46, 14, 30,
                                     43, 45, 37, 42, 35, 31, 34, 38, 9,  7,  53, 20, 21, 54, 15, 26, 58, 16, 2,  18};
        std::vector<int> indices_data = {0, 2, 0, 3, 2, 3, 2, 2, 2, 0, 3, 0, 0, 3, 3};
        std::vector<T> expected_data = {25, 6, 23, 3, 19, 10, 0, 1, 24, 28, 34, 14, 9, 2, 18};

        Tensor input(input_data, {5, 4, 3}, device);
        Tensor indices(indices_data, {5, 1, 3}, device);
        Tensor expected(expected_data, {5, 1, 3}, device);

        CHECK(allclose(gather(input, indices, 1), expected));
    };
    runner_all_except_bool(test_gather1);

    auto test_gather2 = []<typename T>(Device device) {
        std::vector<T> input_data = {25, 32, 23, 44, 41, 27, 49, 6,  50, 33, 8,  51, 4,  52, 47, 11, 22, 17, 36, 19,
                                     48, 3,  55, 10, 5,  59, 29, 12, 13, 39, 0,  1,  24, 56, 57, 40, 28, 46, 14, 30,
                                     43, 45, 37, 42, 35, 31, 34, 38, 9,  7,  53, 20, 21, 54, 15, 26, 58, 16, 2,  18};
        std::vector<int> indices_data = {2, 2, 1, 1, 0, 0, 1, 0, 0, 0, 0, 2, 2, 0, 2, 0, 1, 0, 0, 1};
        std::vector<T> expected_data = {23, 27, 6, 8, 4, 11, 19, 3, 5, 12, 0, 40, 14, 30, 35, 31, 7, 20, 15, 2};

        Tensor input(input_data, {5, 4, 3}, device);
        Tensor indices(indices_data, {5, 4, 1}, device);
        Tensor expected(expected_data, {5, 4, 1}, device);

        CHECK(allclose(gather(input, indices, 2), expected));
    };
    runner_all_except_bool(test_gather2);
}

#ifdef TT_TORCH

namespace {
template <typename T>
std::vector<T> rand_vec(int size, std::mt19937 &gen) {
    std::vector<T> res;
    res.reserve(static_cast<std::size_t>(size));
    std::uniform_real_distribution<float> dis(-5.0, 5.0);
    for (int i = 0; i < size; ++i) {
        res.push_back(static_cast<T>(dis(gen)));
    }
    return res;
}

template <typename T>
auto tensor_to_vec(const torch::Tensor &tensor) -> std::vector<T> {
    return std::vector<T>(tensor.data_ptr<T>(), tensor.data_ptr<T>() + tensor.numel());
}
}    // namespace

// NOLINTNEXTLINE
TEST_CASE("Embedding") {
    auto test = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<int> d_input = {1, 2, 4, 5, 4, 3, 2, 9};
        std::vector<T> d_weight = rand_vec<T>(10 * 4, gen);

        Tensor x_input(d_input, {2, 4}, device);
        Tensor x_weight(d_weight, {10, 4}, device);
        Tensor x_result = embedding(x_input, x_weight);

        const auto options_f = torch::TensorOptions().dtype(torch::kFloat);
        const auto options_i = torch::TensorOptions().dtype(torch::kInt32);
        torch::Tensor t_input = torch::from_blob(d_input.data(), {2, 4}, options_i);
        torch::Tensor t_weight = torch::from_blob(d_weight.data(), {10, 4}, options_f);

        Tensor expected(tensor_to_vec<T>(torch::embedding(t_weight, t_input)), {2, 4, 4}, device);

        CHECK(allclose(x_result, expected));
    };
    runner_single_type<float>(test);
}

#endif
