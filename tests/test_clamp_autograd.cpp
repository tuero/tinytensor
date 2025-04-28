// test_clamp_autograd.cpp
// Test the clamp autograd ops

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <tt/device.h>
#include <tt/exception.h>
#include <tt/scalar.h>
#include <tt/tensor.h>

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

}    // namespace

#ifdef TT_TORCH

// NOLINTNEXTLINE
TEST_CASE("Misc clamp autograd") {
    auto test = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> d = rand_vec<T>(4 * 4, gen);
        std::vector<T> d_min = rand_vec<T>(4 * 4, gen);
        std::vector<T> grad_out = rand_vec<T>(4 * 4, gen);
        // Need int mask to bool for torch
        std::vector<T> d_max;
        for (const auto &v : d_min) {
            d_max.push_back(v + 2);
        }

        Tensor x(d, {4, 4}, device, true);
        Tensor x_min(d_min, {4, 4}, device, true);
        Tensor x_max(d_max, {4, 4}, device, true);
        Tensor output_grad(grad_out, {4, 4}, device, true);

        Tensor x2 = clamp(x, x_min, x_max);
        x2.backward(output_grad);

        const auto options_f = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor t = torch::from_blob(d.data(), {4, 4}, options_f.requires_grad(true));
        torch::Tensor t_min = torch::from_blob(d_min.data(), {4, 4}, options_f.requires_grad(true));
        torch::Tensor t_max = torch::from_blob(d_max.data(), {4, 4}, options_f.requires_grad(true));
        torch::Tensor t_grad = torch::from_blob(grad_out.data(), {4, 4}, options_f);

        torch::Tensor t2 = torch::clamp(t, t_min, t_max);
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
    runner_single_type<float>(test);
}

#endif
