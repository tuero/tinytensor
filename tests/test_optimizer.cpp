// test_optimizer.cpp
// Test the optimizers

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <tt/device.h>
#include <tt/exception.h>
#include <tt/nn/loss.h>
#include <tt/optim/adagrad.h>
#include <tt/optim/adam.h>
#include <tt/optim/adamw.h>
#include <tt/optim/rmsprop.h>
#include <tt/optim/sgd.h>
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
auto tensor_to_vec(const torch::Tensor &tensor) -> std::vector<T> {
    return std::vector<T>(tensor.data_ptr<T>(), tensor.data_ptr<T>() + tensor.numel());
}
}    // namespace

// NOLINTNEXTLINE
TEST_CASE("Optimizer SGD") {
    auto test_vanilla = []<typename T>(Device device) {
        constexpr double lr = 0.1;
        std::mt19937 gen(0);
        std::vector<T> d_input = rand_vec<T>(4 * 4 * 4, gen);
        std::vector<T> d_weight = rand_vec<T>(4 * 4 * 4, gen);
        std::vector<T> d_target = rand_vec<T>(4 * 4 * 4, gen);

        Tensor x_input(d_input, {4, 4, 4}, device);
        Tensor x_weight(d_weight, {4, 4, 4}, device, true);
        Tensor x_target(d_target, {4, 4, 4}, device);
        Tensor x_weight_pre = x_weight.clone();
        optim::SGD sgd({x_weight}, lr);
        Tensor x = relu(x_input * x_weight);
        Tensor x_loss = nn::mse_loss(x, x_target, nn::ReductionMode::mean);
        x_loss.backward();
        sgd.step();

        const auto options_f = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor t_input = torch::from_blob(d_input.data(), {4, 4, 4}, options_f);
        torch::Tensor t_weight = torch::from_blob(d_weight.data(), {4, 4, 4}, options_f.requires_grad(true));
        torch::Tensor t_target = torch::from_blob(d_target.data(), {4, 4, 4}, options_f);
        torch::optim::SGD sgd_torch({t_weight}, torch::optim::SGDOptions(lr));
        torch::Tensor t = torch::nn::functional::relu(t_input * t_weight);
        auto torch_loss_options = torch::nn::functional::MSELossFuncOptions(torch::kMean);
        torch::Tensor t_loss = torch::nn::functional::mse_loss(t, t_target, torch_loss_options);
        t_loss.backward();
        sgd_torch.step();

        Tensor expected(tensor_to_vec<T>(t_weight), {4, 4, 4}, device);

        CHECK_FALSE(allclose(x_weight, x_weight_pre));
        CHECK(allclose(x_weight, expected));
    };
    runner_single_type<float>(test_vanilla);

    auto test_weight_decay = []<typename T>(Device device) {
        constexpr double lr = 0.1;
        constexpr double wd = 0.2;
        std::mt19937 gen(0);
        std::vector<T> d_input = rand_vec<T>(4 * 4 * 4, gen);
        std::vector<T> d_weight = rand_vec<T>(4 * 4 * 4, gen);
        std::vector<T> d_target = rand_vec<T>(4 * 4 * 4, gen);

        Tensor x_input(d_input, {4, 4, 4}, device);
        Tensor x_weight(d_weight, {4, 4, 4}, device, true);
        Tensor x_target(d_target, {4, 4, 4}, device);
        Tensor x_weight_pre = x_weight.clone();
        optim::SGD sgd({x_weight}, lr, {.weight_decay = wd});
        Tensor x = relu(x_input * x_weight);
        Tensor x_loss = nn::mse_loss(x, x_target, nn::ReductionMode::mean);
        x_loss.backward();
        sgd.step();

        const auto options_f = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor t_input = torch::from_blob(d_input.data(), {4, 4, 4}, options_f);
        torch::Tensor t_weight = torch::from_blob(d_weight.data(), {4, 4, 4}, options_f.requires_grad(true));
        torch::Tensor t_target = torch::from_blob(d_target.data(), {4, 4, 4}, options_f);
        torch::optim::SGD sgd_torch({t_weight}, torch::optim::SGDOptions(lr).weight_decay(wd));
        torch::Tensor t = torch::nn::functional::relu(t_input * t_weight);
        auto torch_loss_options = torch::nn::functional::MSELossFuncOptions(torch::kMean);
        torch::Tensor t_loss = torch::nn::functional::mse_loss(t, t_target, torch_loss_options);
        t_loss.backward();
        sgd_torch.step();

        Tensor expected(tensor_to_vec<T>(t_weight), {4, 4, 4}, device);

        CHECK_FALSE(allclose(x_weight, x_weight_pre));
        CHECK(allclose(x_weight, expected));
    };
    runner_single_type<float>(test_weight_decay);

    auto test_momentum = []<typename T>(Device device) {
        constexpr double lr = 0.1;
        constexpr double momentum = 0.2;
        std::mt19937 gen(0);
        std::vector<T> d_input = rand_vec<T>(4 * 4 * 4, gen);
        std::vector<T> d_weight = rand_vec<T>(4 * 4 * 4, gen);
        std::vector<T> d_target = rand_vec<T>(4 * 4 * 4, gen);

        Tensor x_input(d_input, {4, 4, 4}, device);
        Tensor x_weight(d_weight, {4, 4, 4}, device, true);
        Tensor x_target(d_target, {4, 4, 4}, device);
        Tensor x_weight_pre = x_weight.clone();
        optim::SGD sgd({x_weight}, lr, {.momentum = momentum, .use_nesterov = true});
        Tensor x = relu(x_input * x_weight);
        Tensor x_loss = nn::mse_loss(x, x_target, nn::ReductionMode::mean);
        x_loss.backward();
        sgd.step();

        const auto options_f = torch::TensorOptions().dtype(torch::kFloat64);
        torch::Tensor t_input = torch::from_blob(d_input.data(), {4, 4, 4}, options_f);
        torch::Tensor t_weight = torch::from_blob(d_weight.data(), {4, 4, 4}, options_f.requires_grad(true));
        torch::Tensor t_target = torch::from_blob(d_target.data(), {4, 4, 4}, options_f);
        torch::optim::SGD sgd_torch({t_weight}, torch::optim::SGDOptions(lr).momentum(momentum).nesterov(true));
        torch::Tensor t = torch::nn::functional::relu(t_input * t_weight);
        auto torch_loss_options = torch::nn::functional::MSELossFuncOptions(torch::kMean);
        torch::Tensor t_loss = torch::nn::functional::mse_loss(t, t_target, torch_loss_options);
        t_loss.backward();
        sgd_torch.step();

        Tensor expected(tensor_to_vec<T>(t_weight), {4, 4, 4}, device);

        CHECK_FALSE(allclose(x_weight, x_weight_pre));
        CHECK(allclose(x_weight, expected));
    };
    runner_single_type<double>(test_momentum);
}

// NOLINTNEXTLINE
TEST_CASE("Optimizer RMSprop") {
    auto test_vanilla = []<typename T>(Device device) {
        constexpr double lr = 0.1;
        std::mt19937 gen(0);
        std::vector<T> d_input = rand_vec<T>(4 * 4 * 4, gen);
        std::vector<T> d_weight = rand_vec<T>(4 * 4 * 4, gen);
        std::vector<T> d_target = rand_vec<T>(4 * 4 * 4, gen);

        Tensor x_input(d_input, {4, 4, 4}, device);
        Tensor x_weight(d_weight, {4, 4, 4}, device, true);
        Tensor x_target(d_target, {4, 4, 4}, device);
        Tensor x_weight_pre = x_weight.clone();
        optim::RMSprop rmsprop({x_weight}, lr);
        Tensor x = relu(x_input * x_weight);
        Tensor x_loss = nn::mse_loss(x, x_target, nn::ReductionMode::mean);
        x_loss.backward();
        rmsprop.step();

        const auto options_f = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor t_input = torch::from_blob(d_input.data(), {4, 4, 4}, options_f);
        torch::Tensor t_weight = torch::from_blob(d_weight.data(), {4, 4, 4}, options_f.requires_grad(true));
        torch::Tensor t_target = torch::from_blob(d_target.data(), {4, 4, 4}, options_f);
        torch::optim::RMSprop rmsprop_torch({t_weight}, torch::optim::RMSpropOptions(lr));
        torch::Tensor t = torch::nn::functional::relu(t_input * t_weight);
        auto torch_loss_options = torch::nn::functional::MSELossFuncOptions(torch::kMean);
        torch::Tensor t_loss = torch::nn::functional::mse_loss(t, t_target, torch_loss_options);
        t_loss.backward();
        rmsprop_torch.step();

        Tensor expected(tensor_to_vec<T>(t_weight), {4, 4, 4}, device);

        CHECK_FALSE(allclose(x_weight, x_weight_pre));
        CHECK(allclose(x_weight, expected));
    };
    runner_single_type<float>(test_vanilla);

    auto test_momentum = []<typename T>(Device device) {
        constexpr double lr = 0.1;
        constexpr double momentum = 0.2;
        std::mt19937 gen(0);
        std::vector<T> d_input = rand_vec<T>(4 * 4 * 4, gen);
        std::vector<T> d_weight = rand_vec<T>(4 * 4 * 4, gen);
        std::vector<T> d_target = rand_vec<T>(4 * 4 * 4, gen);

        Tensor x_input(d_input, {4, 4, 4}, device);
        Tensor x_weight(d_weight, {4, 4, 4}, device, true);
        Tensor x_target(d_target, {4, 4, 4}, device);
        Tensor x_weight_pre = x_weight.clone();
        optim::RMSprop rmsprop({x_weight}, lr, {.momentum = momentum});
        Tensor x = relu(x_input * x_weight);
        Tensor x_loss = nn::mse_loss(x, x_target, nn::ReductionMode::mean);
        x_loss.backward();
        rmsprop.step();

        const auto options_f = torch::TensorOptions().dtype(torch::kFloat64);
        torch::Tensor t_input = torch::from_blob(d_input.data(), {4, 4, 4}, options_f);
        torch::Tensor t_weight = torch::from_blob(d_weight.data(), {4, 4, 4}, options_f.requires_grad(true));
        torch::Tensor t_target = torch::from_blob(d_target.data(), {4, 4, 4}, options_f);
        torch::optim::RMSprop rmsprop_torch({t_weight}, torch::optim::RMSpropOptions(lr).momentum(momentum));
        torch::Tensor t = torch::nn::functional::relu(t_input * t_weight);
        auto torch_loss_options = torch::nn::functional::MSELossFuncOptions(torch::kMean);
        torch::Tensor t_loss = torch::nn::functional::mse_loss(t, t_target, torch_loss_options);
        t_loss.backward();
        rmsprop_torch.step();

        Tensor expected(tensor_to_vec<T>(t_weight), {4, 4, 4}, device);

        CHECK_FALSE(allclose(x_weight, x_weight_pre));
        CHECK(allclose(x_weight, expected));
    };
    runner_single_type<double>(test_momentum);

    auto test_momentum_center = []<typename T>(Device device) {
        constexpr double lr = 0.1;
        constexpr double momentum = 0.2;
        std::mt19937 gen(0);
        std::vector<T> d_input = rand_vec<T>(4 * 4 * 4, gen);
        std::vector<T> d_weight = rand_vec<T>(4 * 4 * 4, gen);
        std::vector<T> d_target = rand_vec<T>(4 * 4 * 4, gen);

        Tensor x_input(d_input, {4, 4, 4}, device);
        Tensor x_weight(d_weight, {4, 4, 4}, device, true);
        Tensor x_target(d_target, {4, 4, 4}, device);
        Tensor x_weight_pre = x_weight.clone();
        optim::RMSprop rmsprop({x_weight}, lr, {.momentum = momentum, .center = true});
        Tensor x = relu(x_input * x_weight);
        Tensor x_loss = nn::mse_loss(x, x_target, nn::ReductionMode::mean);
        x_loss.backward();
        rmsprop.step();

        const auto options_f = torch::TensorOptions().dtype(torch::kFloat64);
        torch::Tensor t_input = torch::from_blob(d_input.data(), {4, 4, 4}, options_f);
        torch::Tensor t_weight = torch::from_blob(d_weight.data(), {4, 4, 4}, options_f.requires_grad(true));
        torch::Tensor t_target = torch::from_blob(d_target.data(), {4, 4, 4}, options_f);
        torch::optim::RMSprop rmsprop_torch(
            {t_weight},
            torch::optim::RMSpropOptions(lr).momentum(momentum).centered(true)
        );
        torch::Tensor t = torch::nn::functional::relu(t_input * t_weight);
        auto torch_loss_options = torch::nn::functional::MSELossFuncOptions(torch::kMean);
        torch::Tensor t_loss = torch::nn::functional::mse_loss(t, t_target, torch_loss_options);
        t_loss.backward();
        rmsprop_torch.step();

        Tensor expected(tensor_to_vec<T>(t_weight), {4, 4, 4}, device);

        CHECK_FALSE(allclose(x_weight, x_weight_pre));
        CHECK(allclose(x_weight, expected));
    };
    runner_single_type<double>(test_momentum_center);
}

// NOLINTNEXTLINE
TEST_CASE("Optimizer Adam") {
    auto test_vanilla = []<typename T>(Device device) {
        constexpr double lr = 0.1;
        std::mt19937 gen(0);
        std::vector<T> d_input = rand_vec<T>(4 * 4 * 4, gen);
        std::vector<T> d_weight = rand_vec<T>(4 * 4 * 4, gen);
        std::vector<T> d_target = rand_vec<T>(4 * 4 * 4, gen);

        Tensor x_input(d_input, {4, 4, 4}, device);
        Tensor x_weight(d_weight, {4, 4, 4}, device, true);
        Tensor x_target(d_target, {4, 4, 4}, device);
        Tensor x_weight_pre = x_weight.clone();
        optim::Adam adam({x_weight}, lr);
        Tensor x = relu(x_input * x_weight);
        Tensor x_loss = nn::mse_loss(x, x_target, nn::ReductionMode::mean);
        x_loss.backward();
        adam.step();

        const auto options_f = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor t_input = torch::from_blob(d_input.data(), {4, 4, 4}, options_f);
        torch::Tensor t_weight = torch::from_blob(d_weight.data(), {4, 4, 4}, options_f.requires_grad(true));
        torch::Tensor t_target = torch::from_blob(d_target.data(), {4, 4, 4}, options_f);
        torch::optim::Adam adam_torch({t_weight}, torch::optim::AdamOptions(lr));
        torch::Tensor t = torch::nn::functional::relu(t_input * t_weight);
        auto torch_loss_options = torch::nn::functional::MSELossFuncOptions(torch::kMean);
        torch::Tensor t_loss = torch::nn::functional::mse_loss(t, t_target, torch_loss_options);
        t_loss.backward();
        adam_torch.step();

        Tensor expected(tensor_to_vec<T>(t_weight), {4, 4, 4}, device);

        CHECK_FALSE(allclose(x_weight, x_weight_pre));
        CHECK(allclose(x_weight, expected));
    };
    runner_single_type<float>(test_vanilla);

    auto test_weight_decay = []<typename T>(Device device) {
        constexpr double lr = 0.1;
        constexpr double wd = 0.2;
        std::mt19937 gen(0);
        std::vector<T> d_input = rand_vec<T>(4 * 4 * 4, gen);
        std::vector<T> d_weight = rand_vec<T>(4 * 4 * 4, gen);
        std::vector<T> d_target = rand_vec<T>(4 * 4 * 4, gen);

        Tensor x_input(d_input, {4, 4, 4}, device);
        Tensor x_weight(d_weight, {4, 4, 4}, device, true);
        Tensor x_target(d_target, {4, 4, 4}, device);
        Tensor x_weight_pre = x_weight.clone();
        optim::Adam adam({x_weight}, lr, {.weight_decay = wd});
        Tensor x = relu(x_input * x_weight);
        Tensor x_loss = nn::mse_loss(x, x_target, nn::ReductionMode::mean);
        x_loss.backward();
        adam.step();

        const auto options_f = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor t_input = torch::from_blob(d_input.data(), {4, 4, 4}, options_f);
        torch::Tensor t_weight = torch::from_blob(d_weight.data(), {4, 4, 4}, options_f.requires_grad(true));
        torch::Tensor t_target = torch::from_blob(d_target.data(), {4, 4, 4}, options_f);
        torch::optim::Adam adam_torch({t_weight}, torch::optim::AdamOptions(lr).weight_decay(wd));
        torch::Tensor t = torch::nn::functional::relu(t_input * t_weight);
        auto torch_loss_options = torch::nn::functional::MSELossFuncOptions(torch::kMean);
        torch::Tensor t_loss = torch::nn::functional::mse_loss(t, t_target, torch_loss_options);
        t_loss.backward();
        adam_torch.step();

        Tensor expected(tensor_to_vec<T>(t_weight), {4, 4, 4}, device);

        CHECK_FALSE(allclose(x_weight, x_weight_pre));
        CHECK(allclose(x_weight, expected));
    };
    runner_single_type<float>(test_weight_decay);

    auto test_ams = []<typename T>(Device device) {
        constexpr double lr = 0.1;
        constexpr double wd = 0.2;
        std::mt19937 gen(0);
        std::vector<T> d_input = rand_vec<T>(4 * 4 * 4, gen);
        std::vector<T> d_weight = rand_vec<T>(4 * 4 * 4, gen);
        std::vector<T> d_target = rand_vec<T>(4 * 4 * 4, gen);

        Tensor x_input(d_input, {4, 4, 4}, device);
        Tensor x_weight(d_weight, {4, 4, 4}, device, true);
        Tensor x_target(d_target, {4, 4, 4}, device);
        Tensor x_weight_pre = x_weight.clone();
        optim::Adam adam({x_weight}, lr, {.weight_decay = wd, .use_amsgrad = true});
        Tensor x = relu(x_input * x_weight);
        Tensor x_loss = nn::mse_loss(x, x_target, nn::ReductionMode::mean);
        x_loss.backward();
        adam.step();

        const auto options_f = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor t_input = torch::from_blob(d_input.data(), {4, 4, 4}, options_f);
        torch::Tensor t_weight = torch::from_blob(d_weight.data(), {4, 4, 4}, options_f.requires_grad(true));
        torch::Tensor t_target = torch::from_blob(d_target.data(), {4, 4, 4}, options_f);
        torch::optim::Adam adam_torch({t_weight}, torch::optim::AdamOptions(lr).weight_decay(wd).amsgrad(true));
        torch::Tensor t = torch::nn::functional::relu(t_input * t_weight);
        auto torch_loss_options = torch::nn::functional::MSELossFuncOptions(torch::kMean);
        torch::Tensor t_loss = torch::nn::functional::mse_loss(t, t_target, torch_loss_options);
        t_loss.backward();
        adam_torch.step();

        Tensor expected(tensor_to_vec<T>(t_weight), {4, 4, 4}, device);

        CHECK_FALSE(allclose(x_weight, x_weight_pre));
        CHECK(allclose(x_weight, expected));
    };
    runner_single_type<float>(test_ams);
}

// NOLINTNEXTLINE
TEST_CASE("Optimizer AdamW") {
    auto test_vanilla = []<typename T>(Device device) {
        constexpr double lr = 0.1;
        std::mt19937 gen(0);
        std::vector<T> d_input = rand_vec<T>(4 * 4 * 4, gen);
        std::vector<T> d_weight = rand_vec<T>(4 * 4 * 4, gen);
        std::vector<T> d_target = rand_vec<T>(4 * 4 * 4, gen);

        Tensor x_input(d_input, {4, 4, 4}, device);
        Tensor x_weight(d_weight, {4, 4, 4}, device, true);
        Tensor x_target(d_target, {4, 4, 4}, device);
        Tensor x_weight_pre = x_weight.clone();
        optim::AdamW adamw({x_weight}, lr);
        Tensor x = relu(x_input * x_weight);
        Tensor x_loss = nn::mse_loss(x, x_target, nn::ReductionMode::mean);
        x_loss.backward();
        adamw.step();

        const auto options_f = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor t_input = torch::from_blob(d_input.data(), {4, 4, 4}, options_f);
        torch::Tensor t_weight = torch::from_blob(d_weight.data(), {4, 4, 4}, options_f.requires_grad(true));
        torch::Tensor t_target = torch::from_blob(d_target.data(), {4, 4, 4}, options_f);
        torch::optim::AdamW adamw_torch({t_weight}, torch::optim::AdamWOptions(lr).weight_decay(0));
        torch::Tensor t = torch::nn::functional::relu(t_input * t_weight);
        auto torch_loss_options = torch::nn::functional::MSELossFuncOptions(torch::kMean);
        torch::Tensor t_loss = torch::nn::functional::mse_loss(t, t_target, torch_loss_options);
        t_loss.backward();
        adamw_torch.step();

        Tensor expected(tensor_to_vec<T>(t_weight), {4, 4, 4}, device);

        CHECK_FALSE(allclose(x_weight, x_weight_pre));
        CHECK(allclose(x_weight, expected));
    };
    runner_single_type<float>(test_vanilla);

    auto test_weight_decay = []<typename T>(Device device) {
        constexpr double lr = 0.1;
        constexpr double wd = 0.2;
        std::mt19937 gen(0);
        std::vector<T> d_input = rand_vec<T>(4 * 4 * 4, gen);
        std::vector<T> d_weight = rand_vec<T>(4 * 4 * 4, gen);
        std::vector<T> d_target = rand_vec<T>(4 * 4 * 4, gen);

        Tensor x_input(d_input, {4, 4, 4}, device);
        Tensor x_weight(d_weight, {4, 4, 4}, device, true);
        Tensor x_target(d_target, {4, 4, 4}, device);
        Tensor x_weight_pre = x_weight.clone();
        optim::AdamW adamw({x_weight}, lr, {.weight_decay = wd});
        Tensor x = relu(x_input * x_weight);
        Tensor x_loss = nn::mse_loss(x, x_target, nn::ReductionMode::mean);
        x_loss.backward();
        adamw.step();

        const auto options_f = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor t_input = torch::from_blob(d_input.data(), {4, 4, 4}, options_f);
        torch::Tensor t_weight = torch::from_blob(d_weight.data(), {4, 4, 4}, options_f.requires_grad(true));
        torch::Tensor t_target = torch::from_blob(d_target.data(), {4, 4, 4}, options_f);
        torch::optim::AdamW adamw_torch({t_weight}, torch::optim::AdamWOptions(lr).weight_decay(wd));
        torch::Tensor t = torch::nn::functional::relu(t_input * t_weight);
        auto torch_loss_options = torch::nn::functional::MSELossFuncOptions(torch::kMean);
        torch::Tensor t_loss = torch::nn::functional::mse_loss(t, t_target, torch_loss_options);
        t_loss.backward();
        adamw_torch.step();

        Tensor expected(tensor_to_vec<T>(t_weight), {4, 4, 4}, device);

        CHECK_FALSE(allclose(x_weight, x_weight_pre));
        CHECK(allclose(x_weight, expected));
    };
    runner_single_type<float>(test_weight_decay);

    auto test_ams = []<typename T>(Device device) {
        constexpr double lr = 0.1;
        constexpr double wd = 0.2;
        std::mt19937 gen(0);
        std::vector<T> d_input = rand_vec<T>(4 * 4 * 4, gen);
        std::vector<T> d_weight = rand_vec<T>(4 * 4 * 4, gen);
        std::vector<T> d_target = rand_vec<T>(4 * 4 * 4, gen);

        Tensor x_input(d_input, {4, 4, 4}, device);
        Tensor x_weight(d_weight, {4, 4, 4}, device, true);
        Tensor x_target(d_target, {4, 4, 4}, device);
        Tensor x_weight_pre = x_weight.clone();
        optim::AdamW adamw({x_weight}, lr, {.weight_decay = wd, .use_amsgrad = true});
        Tensor x = relu(x_input * x_weight);
        Tensor x_loss = nn::mse_loss(x, x_target, nn::ReductionMode::mean);
        x_loss.backward();
        adamw.step();

        const auto options_f = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor t_input = torch::from_blob(d_input.data(), {4, 4, 4}, options_f);
        torch::Tensor t_weight = torch::from_blob(d_weight.data(), {4, 4, 4}, options_f.requires_grad(true));
        torch::Tensor t_target = torch::from_blob(d_target.data(), {4, 4, 4}, options_f);
        torch::optim::AdamW adamw_torch({t_weight}, torch::optim::AdamWOptions(lr).weight_decay(wd).amsgrad(true));
        torch::Tensor t = torch::nn::functional::relu(t_input * t_weight);
        auto torch_loss_options = torch::nn::functional::MSELossFuncOptions(torch::kMean);
        torch::Tensor t_loss = torch::nn::functional::mse_loss(t, t_target, torch_loss_options);
        t_loss.backward();
        adamw_torch.step();

        Tensor expected(tensor_to_vec<T>(t_weight), {4, 4, 4}, device);

        CHECK_FALSE(allclose(x_weight, x_weight_pre));
        CHECK(allclose(x_weight, expected));
    };
    runner_single_type<float>(test_ams);
}

// NOLINTNEXTLINE
TEST_CASE("Optimizer Adagrad") {
    auto test_vanilla = []<typename T>(Device device) {
        constexpr double lr = 0.1;
        std::mt19937 gen(0);
        std::vector<T> d_input = rand_vec<T>(4 * 4 * 4, gen);
        std::vector<T> d_weight = rand_vec<T>(4 * 4 * 4, gen);
        std::vector<T> d_target = rand_vec<T>(4 * 4 * 4, gen);

        Tensor x_input(d_input, {4, 4, 4}, device);
        Tensor x_weight(d_weight, {4, 4, 4}, device, true);
        Tensor x_target(d_target, {4, 4, 4}, device);
        Tensor x_weight_pre = x_weight.clone();
        optim::Adagrad adagrad({x_weight}, lr);
        Tensor x = relu(x_input * x_weight);
        Tensor x_loss = nn::mse_loss(x, x_target, nn::ReductionMode::mean);
        x_loss.backward();
        adagrad.step();

        const auto options_f = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor t_input = torch::from_blob(d_input.data(), {4, 4, 4}, options_f);
        torch::Tensor t_weight = torch::from_blob(d_weight.data(), {4, 4, 4}, options_f.requires_grad(true));
        torch::Tensor t_target = torch::from_blob(d_target.data(), {4, 4, 4}, options_f);
        torch::optim::Adagrad adagrad_torch({t_weight}, torch::optim::AdagradOptions(lr));
        torch::Tensor t = torch::nn::functional::relu(t_input * t_weight);
        auto torch_loss_options = torch::nn::functional::MSELossFuncOptions(torch::kMean);
        torch::Tensor t_loss = torch::nn::functional::mse_loss(t, t_target, torch_loss_options);
        t_loss.backward();
        adagrad_torch.step();

        Tensor expected(tensor_to_vec<T>(t_weight), {4, 4, 4}, device);

        CHECK_FALSE(allclose(x_weight, x_weight_pre));
        CHECK(allclose(x_weight, expected));
    };
    runner_single_type<float>(test_vanilla);

    auto test_weight_decay = []<typename T>(Device device) {
        constexpr double lr = 0.1;
        constexpr double wd = 0.2;
        std::mt19937 gen(0);
        std::vector<T> d_input = rand_vec<T>(4 * 4 * 4, gen);
        std::vector<T> d_weight = rand_vec<T>(4 * 4 * 4, gen);
        std::vector<T> d_target = rand_vec<T>(4 * 4 * 4, gen);

        Tensor x_input(d_input, {4, 4, 4}, device);
        Tensor x_weight(d_weight, {4, 4, 4}, device, true);
        Tensor x_target(d_target, {4, 4, 4}, device);
        Tensor x_weight_pre = x_weight.clone();
        optim::Adagrad adagrad({x_weight}, lr, {.weight_decay = wd});
        Tensor x = relu(x_input * x_weight);
        Tensor x_loss = nn::mse_loss(x, x_target, nn::ReductionMode::mean);
        x_loss.backward();
        adagrad.step();

        const auto options_f = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor t_input = torch::from_blob(d_input.data(), {4, 4, 4}, options_f);
        torch::Tensor t_weight = torch::from_blob(d_weight.data(), {4, 4, 4}, options_f.requires_grad(true));
        torch::Tensor t_target = torch::from_blob(d_target.data(), {4, 4, 4}, options_f);
        torch::optim::Adagrad adagrad_torch({t_weight}, torch::optim::AdagradOptions(lr).weight_decay(wd));
        torch::Tensor t = torch::nn::functional::relu(t_input * t_weight);
        auto torch_loss_options = torch::nn::functional::MSELossFuncOptions(torch::kMean);
        torch::Tensor t_loss = torch::nn::functional::mse_loss(t, t_target, torch_loss_options);
        t_loss.backward();
        adagrad_torch.step();

        Tensor expected(tensor_to_vec<T>(t_weight), {4, 4, 4}, device);

        CHECK_FALSE(allclose(x_weight, x_weight_pre));
        CHECK(allclose(x_weight, expected));
    };
    runner_single_type<float>(test_weight_decay);
}

#endif
