// test_loss.cpp
// Test the loss functions

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <tt/device.h>
#include <tt/exception.h>
#include <tt/nn/loss.h>
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
std::vector<T> rand_vec_unit(int size, std::mt19937 &gen) {
    std::vector<T> res;
    res.reserve(static_cast<std::size_t>(size));
    std::uniform_real_distribution<float> dis(0, 1.0);
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
TEST_CASE("Loss L1") {
    auto test = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> d_input = rand_vec<T>(4 * 4 * 4, gen);
        std::vector<T> d_target = rand_vec<T>(4 * 4 * 4, gen);

        Tensor x_input(d_input, {4, 4, 4}, device);
        Tensor x_target(d_target, {4, 4, 4}, device);
        Tensor x_loss = nn::l1_loss(x_input, x_target, nn::ReductionMode::mean);

        const auto options_f = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor t_input = torch::from_blob(d_input.data(), {4, 4, 4}, options_f);
        torch::Tensor t_target = torch::from_blob(d_target.data(), {4, 4, 4}, options_f);
        auto torch_loss_options = torch::nn::functional::L1LossFuncOptions(torch::kMean);
        torch::Tensor t_loss = torch::nn::functional::l1_loss(t_input, t_target, torch_loss_options);

        Tensor expected(tensor_to_vec<T>(t_loss), {1}, device);

        CHECK(allclose(x_loss, expected));
    };
    runner_single_type<float>(test);
}

// NOLINTNEXTLINE
TEST_CASE("Loss L2") {
    auto test = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> d_input = rand_vec<T>(4 * 4 * 4, gen);
        std::vector<T> d_target = rand_vec<T>(4 * 4 * 4, gen);

        Tensor x_input(d_input, {4, 4, 4}, device);
        Tensor x_target(d_target, {4, 4, 4}, device);
        Tensor x_loss = nn::mse_loss(x_input, x_target, nn::ReductionMode::mean);

        const auto options_f = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor t_input = torch::from_blob(d_input.data(), {4, 4, 4}, options_f);
        torch::Tensor t_target = torch::from_blob(d_target.data(), {4, 4, 4}, options_f);
        auto torch_loss_options = torch::nn::functional::MSELossFuncOptions(torch::kMean);
        torch::Tensor t_loss = torch::nn::functional::mse_loss(t_input, t_target, torch_loss_options);

        Tensor expected(tensor_to_vec<T>(t_loss), {1}, device);

        CHECK(allclose(x_loss, expected));
    };
    runner_single_type<float>(test);
}

// NOLINTNEXTLINE
TEST_CASE("Loss CrossEntropy") {
    auto test1 = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> d_input = rand_vec<T>(4 * 3, gen);
        std::vector<T> d_weight = rand_vec<T>(3, gen);

        Tensor x_input(d_input, {4, 3}, device);
        Tensor x_target = arange({4}, kI32, device) % 3;
        Tensor x_weight(d_weight, {3}, device);
        Tensor x_loss = nn::cross_entropy_loss(x_input, x_target, x_weight, nn::ReductionMode::mean);

        const auto options_f = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor t_input = torch::from_blob(d_input.data(), {4, 3}, options_f);
        torch::Tensor t_target = torch::arange(4) % 3;
        torch::Tensor t_weight = torch::from_blob(d_weight.data(), {3}, options_f);
        auto torch_loss_options = torch::nn::functional::CrossEntropyFuncOptions().weight(t_weight);
        torch::Tensor t_loss = torch::nn::functional::cross_entropy(t_input, t_target, torch_loss_options);

        Tensor expected(tensor_to_vec<T>(t_loss), {1}, device);

        CHECK(allclose(x_loss, expected));
    };
    runner_single_type<float>(test1);

    auto test2 = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> d_input = rand_vec<T>(4 * 3 * 2 * 2, gen);
        std::vector<T> d_weight = rand_vec<T>(3, gen);

        Tensor x_input(d_input, {4, 3, 2, 2}, device);
        Tensor x_target = arange({4, 2, 2}, kI32, device) % 3;
        Tensor x_weight(d_weight, {3}, device);
        Tensor x_loss = nn::cross_entropy_loss(x_input, x_target, x_weight, nn::ReductionMode::mean);

        const auto options_f = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor t_input = torch::from_blob(d_input.data(), {4, 3, 2, 2}, options_f);
        torch::Tensor t_target = (torch::arange(4 * 2 * 2) % 3).reshape({4, 2, 2});
        torch::Tensor t_weight = torch::from_blob(d_weight.data(), {3}, options_f);
        auto torch_loss_options = torch::nn::functional::CrossEntropyFuncOptions().weight(t_weight);
        torch::Tensor t_loss = torch::nn::functional::cross_entropy(t_input, t_target, torch_loss_options);

        Tensor expected(tensor_to_vec<T>(t_loss), {1}, device);

        CHECK(allclose(x_loss, expected));
    };
    runner_single_type<float>(test2);
}

// NOLINTNEXTLINE
TEST_CASE("Loss NLL") {
    auto test1 = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> d_input = rand_vec<T>(4 * 3, gen);
        std::vector<T> d_weight = rand_vec<T>(3, gen);

        Tensor x_input(d_input, {4, 3}, device);
        Tensor x_target = arange({4}, kI32, device) % 3;
        Tensor x_weight(d_weight, {3}, device);
        Tensor x_loss = nn::nll_loss(x_input, x_target, x_weight, nn::ReductionMode::mean);

        const auto options_f = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor t_input = torch::from_blob(d_input.data(), {4, 3}, options_f);
        torch::Tensor t_target = torch::arange(4) % 3;
        torch::Tensor t_weight = torch::from_blob(d_weight.data(), {3}, options_f);
        auto torch_loss_options = torch::nn::functional::NLLLossFuncOptions().weight(t_weight);
        torch::Tensor t_loss = torch::nn::functional::nll_loss(t_input, t_target, torch_loss_options);

        Tensor expected(tensor_to_vec<T>(t_loss), {1}, device);

        CHECK(allclose(x_loss, expected));
    };
    runner_single_type<float>(test1);

    auto test2 = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> d_input = rand_vec<T>(4 * 3 * 2 * 2, gen);
        std::vector<T> d_weight = rand_vec<T>(3, gen);

        Tensor x_input(d_input, {4, 3, 2, 2}, device);
        Tensor x_target = arange({4, 2, 2}, kI32, device) % 3;
        Tensor x_weight(d_weight, {3}, device);
        Tensor x_loss = nn::nll_loss(x_input, x_target, x_weight, nn::ReductionMode::mean);

        const auto options_f = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor t_input = torch::from_blob(d_input.data(), {4, 3, 2, 2}, options_f);
        torch::Tensor t_target = (torch::arange(4 * 2 * 2) % 3).reshape({4, 2, 2});
        torch::Tensor t_weight = torch::from_blob(d_weight.data(), {3}, options_f);
        auto torch_loss_options = torch::nn::functional::NLLLossFuncOptions().weight(t_weight);
        torch::Tensor t_loss = torch::nn::functional::nll_loss(t_input, t_target, torch_loss_options);

        Tensor expected(tensor_to_vec<T>(t_loss), {1}, device);

        CHECK(allclose(x_loss, expected));
    };
    runner_single_type<float>(test2);
}

// NOLINTNEXTLINE
TEST_CASE("Loss KLD") {
    auto test1 = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> d_input = rand_vec_unit<T>(4 * 4 * 4, gen);
        std::vector<T> d_target = rand_vec_unit<T>(4 * 4 * 4, gen);

        Tensor x_input(d_input, {4, 4, 4}, device);
        Tensor x_target(d_target, {4, 4, 4}, device);
        Tensor x_loss = nn::kld_loss(x_input, x_target, nn::ReductionMode::batch_mean);

        const auto options_f = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor t_input = torch::from_blob(d_input.data(), {4, 4, 4}, options_f);
        torch::Tensor t_target = torch::from_blob(d_target.data(), {4, 4, 4}, options_f);
        auto torch_loss_options = torch::nn::functional::KLDivFuncOptions(torch::kBatchMean);
        torch::Tensor t_loss = torch::nn::functional::kl_div(t_input, t_target, torch_loss_options);

        Tensor expected(tensor_to_vec<T>(t_loss), {1}, device);

        CHECK(allclose(x_loss, expected));
    };
    runner_single_type<float>(test1);

    auto test2 = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> d_input = rand_vec_unit<T>(4 * 4 * 4, gen);
        std::vector<T> d_target = rand_vec_unit<T>(4 * 4 * 4, gen);

        Tensor x_input(d_input, {4, 4, 4}, device);
        Tensor x_target(d_target, {4, 4, 4}, device);
        Tensor x_loss = nn::kld_loss(x_input, log(x_target), nn::ReductionMode::batch_mean, true);

        const auto options_f = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor t_input = torch::from_blob(d_input.data(), {4, 4, 4}, options_f);
        torch::Tensor t_target = torch::from_blob(d_target.data(), {4, 4, 4}, options_f);
        auto torch_loss_options = torch::nn::functional::KLDivFuncOptions(torch::kBatchMean).log_target(true);
        torch::Tensor t_loss = torch::nn::functional::kl_div(t_input, t_target.log(), torch_loss_options);

        Tensor expected(tensor_to_vec<T>(t_loss), {1}, device);

        CHECK(allclose(x_loss, expected));
    };
    runner_single_type<float>(test2);
}

// NOLINTNEXTLINE
TEST_CASE("Loss BCE") {
    auto test = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> d_input = rand_vec_unit<T>(4 * 4 * 4, gen);
        std::vector<T> d_target = rand_vec_unit<T>(4 * 4 * 4, gen);

        Tensor x_input(d_input, {4, 4, 4}, device);
        Tensor x_target(d_target, {4, 4, 4}, device);
        Tensor x_loss = nn::bce_loss(x_input, x_target, nn::ReductionMode::mean);

        const auto options_f = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor t_input = torch::from_blob(d_input.data(), {4, 4, 4}, options_f);
        torch::Tensor t_target = torch::from_blob(d_target.data(), {4, 4, 4}, options_f);
        auto torch_loss_options = torch::nn::functional::BinaryCrossEntropyFuncOptions().reduction(torch::kMean);
        torch::Tensor t_loss = torch::nn::functional::binary_cross_entropy(t_input, t_target, torch_loss_options);

        Tensor expected(tensor_to_vec<T>(t_loss), {1}, device);

        CHECK(allclose(x_loss, expected));
    };
    runner_single_type<float>(test);
}

// NOLINTNEXTLINE
TEST_CASE("Loss BCE Logits") {
    auto test = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> d_input = rand_vec<T>(4 * 4 * 4, gen);
        std::vector<T> d_target = rand_vec_unit<T>(4 * 4 * 4, gen);

        Tensor x_input(d_input, {4, 4, 4}, device);
        Tensor x_target(d_target, {4, 4, 4}, device);
        Tensor x_loss = nn::bce_with_logits_loss(x_input, x_target, nn::ReductionMode::mean);

        const auto options_f = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor t_input = torch::from_blob(d_input.data(), {4, 4, 4}, options_f);
        torch::Tensor t_target = torch::from_blob(d_target.data(), {4, 4, 4}, options_f);
        auto torch_loss_options =
            torch::nn::functional::BinaryCrossEntropyWithLogitsFuncOptions().reduction(torch::kMean);
        torch::Tensor t_loss =
            torch::nn::functional::binary_cross_entropy_with_logits(t_input, t_target, torch_loss_options);

        Tensor expected(tensor_to_vec<T>(t_loss), {1}, device);

        CHECK(allclose(x_loss, expected));
    };
    runner_single_type<float>(test);
}

// NOLINTNEXTLINE
TEST_CASE("Loss Huber") {
    auto test = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> d_input = rand_vec<T>(4 * 4 * 4, gen);
        std::vector<T> d_target = rand_vec<T>(4 * 4 * 4, gen);

        Tensor x_input(d_input, {4, 4, 4}, device);
        Tensor x_target(d_target, {4, 4, 4}, device);
        Tensor x_loss = nn::huber_loss(x_input, x_target, 1.0, nn::ReductionMode::mean);

        const auto options_f = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor t_input = torch::from_blob(d_input.data(), {4, 4, 4}, options_f);
        torch::Tensor t_target = torch::from_blob(d_target.data(), {4, 4, 4}, options_f);
        auto torch_loss_options = torch::nn::functional::HuberLossFuncOptions().delta(1.0).reduction(torch::kMean);
        torch::Tensor t_loss = torch::nn::functional::huber_loss(t_input, t_target, torch_loss_options);

        Tensor expected(tensor_to_vec<T>(t_loss), {1}, device);

        CHECK(allclose(x_loss, expected));
    };
    runner_single_type<float>(test);
}

// NOLINTNEXTLINE
TEST_CASE("Loss Smooth L1") {
    auto test = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> d_input = rand_vec<T>(4 * 4 * 4, gen);
        std::vector<T> d_target = rand_vec<T>(4 * 4 * 4, gen);

        Tensor x_input(d_input, {4, 4, 4}, device);
        Tensor x_target(d_target, {4, 4, 4}, device);
        Tensor x_loss = nn::smooth_l1_loss(x_input, x_target, 1.0, nn::ReductionMode::mean);

        const auto options_f = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor t_input = torch::from_blob(d_input.data(), {4, 4, 4}, options_f);
        torch::Tensor t_target = torch::from_blob(d_target.data(), {4, 4, 4}, options_f);
        auto torch_loss_options = torch::nn::functional::SmoothL1LossFuncOptions().beta(1.0).reduction(torch::kMean);
        torch::Tensor t_loss = torch::nn::functional::smooth_l1_loss(t_input, t_target, torch_loss_options);

        Tensor expected(tensor_to_vec<T>(t_loss), {1}, device);

        CHECK(allclose(x_loss, expected));
    };
    runner_single_type<float>(test);
}

// NOLINTNEXTLINE
TEST_CASE("Loss SoftMargin") {
    auto test = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> d_input = rand_vec<T>(4 * 4 * 4, gen);
        std::vector<T> d_target = rand_vec<T>(4 * 4 * 4, gen);

        Tensor x_input(d_input, {4, 4, 4}, device);
        Tensor x_target(d_target, {4, 4, 4}, device);
        Tensor x_loss = nn::soft_margin_loss(x_input, x_target, nn::ReductionMode::mean);

        const auto options_f = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor t_input = torch::from_blob(d_input.data(), {4, 4, 4}, options_f);
        torch::Tensor t_target = torch::from_blob(d_target.data(), {4, 4, 4}, options_f);
        auto torch_loss_options = torch::nn::functional::SoftMarginLossFuncOptions().reduction(torch::kMean);
        torch::Tensor t_loss = torch::nn::functional::soft_margin_loss(t_input, t_target, torch_loss_options);

        Tensor expected(tensor_to_vec<T>(t_loss), {1}, device);

        CHECK(allclose(x_loss, expected));
    };
    runner_single_type<float>(test);
}

#endif
