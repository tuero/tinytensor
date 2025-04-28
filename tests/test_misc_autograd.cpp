// test_misc_autograd.cpp
// Test the misc autograd ops

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
TEST_CASE("Misc where autograd") {
    auto test = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> d_lhs = rand_vec<T>(4 * 4, gen);
        std::vector<T> d_rhs = rand_vec<T>(4 * 4, gen);
        std::vector<bool> d_mask = rand_mask(4 * 4, gen);
        std::vector<T> grad_out = rand_vec<T>(4 * 4, gen);
        // Need int mask to bool for torch
        std::vector<int> d_mask_torch;

        for (const auto &v : d_mask) {
            d_mask_torch.push_back(static_cast<int>(v));
        }

        Tensor lhs(d_lhs, {4, 4}, device, true);
        Tensor rhs(d_rhs, {4, 4}, device, true);
        Tensor mask(d_mask, {4, 4}, device);
        Tensor output_grad(grad_out, {4, 4}, device, true);

        Tensor x2 = where(mask, lhs, rhs);
        x2.backward(output_grad);

        const auto options_f = torch::TensorOptions().dtype(torch::kFloat);
        const auto options_i = torch::TensorOptions().dtype(torch::kInt);
        torch::Tensor t_lhs = torch::from_blob(d_lhs.data(), {4, 4}, options_f.requires_grad(true));
        torch::Tensor t_rhs = torch::from_blob(d_rhs.data(), {4, 4}, options_f.requires_grad(true));
        torch::Tensor t_mask = torch::from_blob(d_mask_torch.data(), {4, 4}, options_i).to(torch::kBool);
        torch::Tensor t_grad = torch::from_blob(grad_out.data(), {4, 4}, options_f);

        torch::Tensor t2 = torch::where(t_mask, t_lhs, t_rhs);
        t2.backward(t_grad);
        torch::Tensor expected_grad_lhs = t_lhs.grad();
        torch::Tensor expected_grad_rhs = t_rhs.grad();
        std::vector<T> expected_grad_lhs_data(
            expected_grad_lhs.data_ptr<T>(),
            expected_grad_lhs.data_ptr<T>() + expected_grad_lhs.numel()
        );
        std::vector<T> expected_grad_rhs_data(
            expected_grad_rhs.data_ptr<T>(),
            expected_grad_rhs.data_ptr<T>() + expected_grad_rhs.numel()
        );
        Tensor expected_lhs(expected_grad_lhs_data, {4, 4}, device);
        Tensor expected_rhs(expected_grad_rhs_data, {4, 4}, device);

        auto close_options = CloseOptions().atol(1e-4).equal_nan();
        CHECK(allclose(lhs.grad().value(), expected_lhs, close_options));
        CHECK(allclose(rhs.grad().value(), expected_rhs, close_options));
    };
    runner_single_type<float>(test);
}

#endif
