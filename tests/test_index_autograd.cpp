// test_index_autograd.cpp
// Test the index autograd ops

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <tt/device.h>
#include <tt/exception.h>
#include <tt/scalar.h>
#include <tt/tensor.h>

#include <algorithm>
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

std::vector<bool> rand_mask(int size, std::mt19937 &gen) {
    std::vector<bool> res;
    res.reserve(static_cast<std::size_t>(size));
    std::uniform_int_distribution<int> dis;
    for (int i = 0; i < size; ++i) {
        res.push_back(static_cast<bool>(dis(gen) & 1));
    }
    return res;
}

}    // namespace

// NOLINTNEXTLINE
TEST_CASE("Index autograd") {
    auto test = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> d = rand_vec<T>(4 * 4 * 4, gen);
        std::vector<T> grad_out = rand_vec<T>(4 * 2, gen);

        Tensor x(d, {4, 4, 4}, device, true);
        Tensor output_grad(grad_out, {4, 2}, device);

        Tensor x2 = x[{indexing::Slice(), 2, indexing::Slice(1, 3)}];
        x2.backward(output_grad);

        const auto options_f = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor t = torch::from_blob(d.data(), {4, 4, 4}, options_f.requires_grad(true));
        torch::Tensor t_grad = torch::from_blob(grad_out.data(), {4, 2}, options_f);

        torch::Tensor t2 = t.index({torch::indexing::Slice(), 2, torch::indexing::Slice(1, 3)});
        t2.backward(t_grad);
        torch::Tensor expected_grad = t.grad();
        std::vector<T> expected_grad_data(
            expected_grad.data_ptr<T>(),
            expected_grad.data_ptr<T>() + expected_grad.numel()
        );
        Tensor expected(expected_grad_data, {4, 4, 4}, device);

        auto close_options = CloseOptions().atol(1e-4).equal_nan();
        CHECK(allclose(x.grad().value(), expected, close_options));
    };
    runner_single_type<float>(test);
}

// NOLINTNEXTLINE
TEST_CASE("Index index autograd") {
    // Indices is equivalent to masking with same indices as true
    auto test_indices = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> d_lhs = rand_vec<T>(4 * 4, gen);
        std::vector<T> d_rhs = rand_vec<T>(4 * 4, gen);
        std::vector<bool> d_mask = rand_mask(4 * 4, gen);

        std::vector<int> d_indices;
        for (std::size_t i = 0; i < 16; ++i) {
            if (d_mask[i]) {
                d_indices.push_back(static_cast<int>(i));
            }
        }
        std::vector<T> grad_out = rand_vec<T>(static_cast<int>(d_indices.size()), gen);

        Tensor x_lhs1(d_lhs, {4, 4}, device, true);
        Tensor x_lhs2(d_lhs, {4, 4}, device, true);
        Tensor x_indices(d_indices, device);
        Tensor x_mask(d_mask, {4, 4}, device);
        Tensor output_grad(grad_out, device);

        Tensor x2 = index(x_lhs1, x_indices);
        x2.backward(output_grad);
        Tensor x3 = x_lhs2[x_mask];
        x3.backward(output_grad);

        auto close_options = CloseOptions().atol(1e-4).equal_nan();
        CHECK(allclose(x_lhs1.grad().value(), x_lhs2.grad().value(), close_options));
    };
    runner_single_type<float>(test_indices);

    auto test_mask = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> d = rand_vec<T>(4 * 4, gen);
        std::vector<bool> d_mask = rand_mask(4 * 4, gen);

        std::vector<int> d_mask_torch;
        int mask_count = 0;
        for (std::size_t i = 0; i < 16; ++i) {
            d_mask_torch.push_back(static_cast<int>(d_mask[i]));
            if (d_mask[i]) {
                ++mask_count;
            }
        }
        std::vector<T> grad_out = rand_vec<T>(mask_count, gen);

        Tensor x(d, {4, 4}, device, true);
        Tensor mask(d_mask, {4, 4}, device);
        Tensor output_grad(grad_out, device);

        Tensor x2 = x[mask];
        x2.backward(output_grad);

        const auto options_i = torch::TensorOptions().dtype(torch::kInt);
        const auto options_f = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor t = torch::from_blob(d.data(), {4, 4}, options_f.requires_grad(true));
        torch::Tensor t_mask = torch::from_blob(d_mask_torch.data(), {4, 4}, options_i).to(torch::kBool);
        torch::Tensor t_grad = torch::from_blob(grad_out.data(), {static_cast<int>(grad_out.size())}, options_f);

        torch::Tensor t2 = t.index({t_mask});
        t2.backward(t_grad);
        torch::Tensor expected_grad = t.grad();
        std::vector<T> expected_grad_data(
            expected_grad.data_ptr<T>(),
            expected_grad.data_ptr<T>() + expected_grad.numel()
        );
        Tensor expected(expected_grad_data, {4, 4}, device);

        auto close_options = CloseOptions().atol(1e-4).equal_nan();
        CHECK(allclose(x.grad().value(), expected, close_options));
    };
    runner_single_type<float>(test_mask);
}

// NOLINTNEXTLINE
TEST_CASE("Index index_put autograd") {
    // Indices is equivalent to masking with same indices as true
    auto test_indices = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> d_lhs = rand_vec<T>(4 * 4, gen);
        std::vector<T> d_rhs = rand_vec<T>(4 * 4, gen);
        std::vector<bool> d_mask = rand_mask(4 * 4, gen);
        std::vector<T> grad_out = rand_vec<T>(4 * 4, gen);

        std::vector<int> d_indices;
        std::vector<T> d_rhs_selected;
        for (std::size_t i = 0; i < 16; ++i) {
            if (d_mask[i]) {
                d_indices.push_back(static_cast<int>(i));
                d_rhs_selected.push_back(d_rhs[i]);
            }
        }

        Tensor x_lhs1(d_lhs, {4, 4}, device, true);
        Tensor x_lhs2(d_lhs, {4, 4}, device, true);
        Tensor x_rhs(d_rhs_selected, device, true);
        Tensor x_indices(d_indices, device);
        Tensor x_mask(d_mask, {4, 4}, device);
        Tensor output_grad(grad_out, {4, 4}, device);

        Tensor x2 = index_put(x_lhs1, x_indices, x_rhs);
        x2.backward(output_grad);
        Tensor x3 = index_put(x_lhs2, x_mask, x_rhs);
        x3.backward(output_grad);

        auto close_options = CloseOptions().atol(1e-4).equal_nan();
        CHECK(allclose(x_lhs1.grad().value(), x_lhs2.grad().value(), close_options));
    };
    runner_single_type<float>(test_indices);

    auto test_mask = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> d_lhs = rand_vec<T>(4 * 4, gen);
        std::vector<T> d_rhs = rand_vec<T>(4 * 4, gen);
        std::vector<bool> d_mask = rand_mask(4 * 4, gen);
        std::vector<T> grad_out = rand_vec<T>(4 * 4, gen);

        std::vector<int> d_mask_torch;
        std::vector<T> d_rhs_selected;
        for (std::size_t i = 0; i < 16; ++i) {
            d_mask_torch.push_back(static_cast<int>(d_mask[i]));
            if (d_mask[i]) {
                d_rhs_selected.push_back(d_rhs[i]);
            }
        }

        Tensor x_lhs(d_lhs, {4, 4}, device, true);
        Tensor x_rhs(d_rhs_selected, device, true);
        Tensor mask(d_mask, {4, 4}, device);
        Tensor output_grad(grad_out, {4, 4}, device);

        Tensor x2 = index_put(x_lhs, mask, x_rhs);
        x2.backward(output_grad);

        const auto options_i = torch::TensorOptions().dtype(torch::kInt);
        const auto options_f = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor t_lhs = torch::from_blob(d_lhs.data(), {4, 4}, options_f.requires_grad(true));
        torch::Tensor t_rhs = torch::from_blob(
            d_rhs_selected.data(),
            {static_cast<int>(d_rhs_selected.size())},
            options_f.requires_grad(true)
        );
        torch::Tensor t_mask = torch::from_blob(d_mask_torch.data(), {4, 4}, options_i).to(torch::kBool);
        torch::Tensor t_grad = torch::from_blob(grad_out.data(), {4, 4}, options_f);

        torch::Tensor t2 = t_lhs.index_put({t_mask}, t_rhs);
        t2.backward(t_grad);
        torch::Tensor expected_grad = t_lhs.grad();
        std::vector<T> expected_grad_data(
            expected_grad.data_ptr<T>(),
            expected_grad.data_ptr<T>() + expected_grad.numel()
        );
        Tensor expected(expected_grad_data, {4, 4}, device);

        auto close_options = CloseOptions().atol(1e-4).equal_nan();
        CHECK(allclose(x_lhs.grad().value(), expected, close_options));
    };
    runner_single_type<float>(test_mask);
}

// NOLINTNEXTLINE
TEST_CASE("Index index_select autograd") {
    auto test_dim0 = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> d = rand_vec<T>(4 * 4, gen);
        std::vector<int> _indices = {3, 3, 0, 2, 0, 2};
        Tensor indices(_indices, {6}, device);
        std::vector<T> grad_out = rand_vec<T>(6 * 4, gen);

        Tensor x1(d, {4, 4}, device, true);
        Tensor output_grad(grad_out, {6, 4}, device);

        Tensor x2 = index_select(x1, indices, 0);
        x2.backward(output_grad);

        const auto options_i = torch::TensorOptions().dtype(torch::kInt);
        const auto options_f = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor t_indices = torch::from_blob(_indices.data(), {6}, options_i).to(torch::kInt64);
        torch::Tensor t1 = torch::from_blob(d.data(), {4, 4}, options_f.requires_grad(true));
        torch::Tensor t_grad = torch::from_blob(grad_out.data(), {6, 4}, options_f);

        torch::Tensor t2 = t1.index_select(0, t_indices);
        t2.backward(t_grad);
        torch::Tensor expected_grad = t1.grad();
        std::vector<T> expected_grad_data(
            expected_grad.data_ptr<T>(),
            expected_grad.data_ptr<T>() + expected_grad.numel()
        );
        Tensor expected(expected_grad_data, {4, 4}, device);

        auto close_options = CloseOptions().atol(1e-4).equal_nan();
        CHECK(allclose(x1.grad().value(), expected, close_options));
    };
    runner_single_type<float>(test_dim0);

    auto test_dim1 = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> d = rand_vec<T>(4 * 4, gen);
        std::vector<int> _indices = {3, 3, 0, 2, 0, 2};
        Tensor indices(_indices, {6}, device);
        std::vector<T> grad_out = rand_vec<T>(6 * 4, gen);

        Tensor x1(d, {4, 4}, device, true);
        Tensor output_grad(grad_out, {4, 6}, device);

        Tensor x2 = index_select(x1, indices, 1);
        x2.backward(output_grad);

        const auto options_i = torch::TensorOptions().dtype(torch::kInt);
        const auto options_f = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor t_indices = torch::from_blob(_indices.data(), {6}, options_i).to(torch::kInt64);
        torch::Tensor t1 = torch::from_blob(d.data(), {4, 4}, options_f.requires_grad(true));
        torch::Tensor t_grad = torch::from_blob(grad_out.data(), {4, 6}, options_f);

        torch::Tensor t2 = t1.index_select(1, t_indices);
        t2.backward(t_grad);
        torch::Tensor expected_grad = t1.grad();
        std::vector<T> expected_grad_data(
            expected_grad.data_ptr<T>(),
            expected_grad.data_ptr<T>() + expected_grad.numel()
        );
        Tensor expected(expected_grad_data, {4, 4}, device);

        auto close_options = CloseOptions().atol(1e-4).equal_nan();
        CHECK(allclose(x1.grad().value(), expected, close_options));
    };
    runner_single_type<float>(test_dim1);
}

// NOLINTNEXTLINE
TEST_CASE("Index repeat autograd") {
    auto test_dim0 = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> d = rand_vec<T>(3 * 2, gen);
        std::vector<T> grad_out = rand_vec<T>(6 * 4, gen);

        Tensor x1(d, {3, 2}, device, true);
        Tensor output_grad(grad_out, {6, 4}, device);

        Tensor x2 = repeat(x1, {2, 2});
        x2.backward(output_grad);

        const auto options_f = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor t1 = torch::from_blob(d.data(), {3, 2}, options_f.requires_grad(true));
        torch::Tensor t_grad = torch::from_blob(grad_out.data(), {6, 4}, options_f);

        torch::Tensor t2 = t1.repeat({2, 2});
        t2.backward(t_grad);
        torch::Tensor expected_grad = t1.grad();
        std::vector<T> expected_grad_data(
            expected_grad.data_ptr<T>(),
            expected_grad.data_ptr<T>() + expected_grad.numel()
        );
        Tensor expected(expected_grad_data, {3, 2}, device);

        auto close_options = CloseOptions().atol(1e-4).equal_nan();
        CHECK(allclose(x1.grad().value(), expected, close_options));
    };
    runner_single_type<float>(test_dim0);
}

// NOLINTNEXTLINE
TEST_CASE("Index repeat_interleave autograd") {
    auto test_dim0 = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> d = rand_vec<T>(4 * 4, gen);
        std::vector<T> grad_out = rand_vec<T>(4 * 8, gen);
        constexpr int num_repeats = 2;

        Tensor x1(d, {4, 4}, device, true);
        Tensor output_grad(grad_out, {4, 8}, device);

        Tensor x2 = repeat_interleave(x1, num_repeats, 1);
        x2.backward(output_grad);

        const auto options_f = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor t1 = torch::from_blob(d.data(), {4, 4}, options_f.requires_grad(true));
        torch::Tensor t_grad = torch::from_blob(grad_out.data(), {4, 8}, options_f);

        torch::Tensor t2 = t1.repeat_interleave(num_repeats, 1);
        t2.backward(t_grad);
        torch::Tensor expected_grad = t1.grad();
        std::vector<T> expected_grad_data(
            expected_grad.data_ptr<T>(),
            expected_grad.data_ptr<T>() + expected_grad.numel()
        );
        Tensor expected(expected_grad_data, {4, 4}, device);

        auto close_options = CloseOptions().atol(1e-4).equal_nan();
        CHECK(allclose(x1.grad().value(), expected, close_options));
    };
    runner_single_type<float>(test_dim0);
}

// NOLINTNEXTLINE
TEST_CASE("Index gather autograd") {
    auto test = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> d = rand_vec<T>(3 * 4 * 3, gen);
        std::vector<int> _indices = {0, 1, 0, 1, 0, 1, 0, 1, 0};
        Tensor indices(_indices, {3, 1, 3}, device);
        std::vector<T> grad_out = rand_vec<T>(3 * 1 * 3, gen);

        Tensor x1(d, {3, 4, 3}, device, true);
        Tensor output_grad(grad_out, {3, 1, 3}, device);

        Tensor x2 = gather(x1, indices, 1);
        x2.backward(output_grad);

        const auto options_f = torch::TensorOptions().dtype(torch::kFloat);
        const auto options_i = torch::TensorOptions().dtype(torch::kInt);
        torch::Tensor t1 = torch::from_blob(d.data(), {3, 4, 3}, options_f.requires_grad(true));
        torch::Tensor t_indices = torch::from_blob(_indices.data(), {3, 1, 3}, options_i).to(torch::kInt64);
        torch::Tensor t_grad = torch::from_blob(grad_out.data(), {3, 1, 3}, options_f);

        torch::Tensor t2 = torch::gather(t1, 1, t_indices);
        t2.backward(t_grad);
        torch::Tensor expected_grad = t1.grad();
        std::vector<T> expected_grad_data(
            expected_grad.data_ptr<T>(),
            expected_grad.data_ptr<T>() + expected_grad.numel()
        );
        Tensor expected(expected_grad_data, {3, 4, 3}, device);

        auto close_options = CloseOptions().atol(1e-4).equal_nan();
        CHECK(allclose(x1.grad().value(), expected, close_options));
    };
    runner_single_type<float>(test);
}

#endif
