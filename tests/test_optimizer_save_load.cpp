// test_optimizer_save_load.cpp
// Test the optimizers saving state and loading

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

#include "doctest.h"
#include "test_util.h"

#include <cstddef>
#include <random>
#include <vector>

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

// NOLINTNEXTLINE
TEST_CASE("Optimizer Save Load SGD") {
    auto test = []<typename T>(Device device) {
        constexpr double lr = 0.1;
        constexpr double momentum = 0.2;
        std::mt19937 gen(0);
        std::vector<T> d_input = rand_vec<T>(4 * 4 * 4, gen);
        std::vector<T> d_weight = rand_vec<T>(4 * 4 * 4, gen);
        std::vector<T> d_target = rand_vec<T>(4 * 4 * 4, gen);

        Tensor input(d_input, {4, 4, 4}, device);
        Tensor weight(d_weight, {4, 4, 4}, device, true);
        Tensor target(d_target, {4, 4, 4}, device);
        optim::SGD sgd({weight}, lr, {.momentum = momentum, .use_nesterov = true});

        // Step 1
        Tensor x = relu(input * weight);
        Tensor loss1 = nn::mse_loss(x, target, nn::ReductionMode::mean);
        loss1.backward();
        sgd.step();

        sgd.zero_grad();

        // Save
        sgd.save("optimizer.pt");
        Tensor weight2 = weight.detach();
        weight2.set_requires_grad(true);

        // Step 2
        x = relu(input * weight);
        Tensor loss2 = nn::mse_loss(x, target, nn::ReductionMode::mean);
        loss2.backward();
        sgd.step();

        // Reload SGD
        optim::SGD sgd2({weight2}, lr, {.momentum = momentum, .use_nesterov = true});
        sgd2.load("optimizer.pt");

        // Step 2 from reload
        x = relu(input * weight2);
        Tensor loss3 = nn::mse_loss(x, target, nn::ReductionMode::mean);
        loss3.backward();
        sgd2.step();

        CHECK(allclose(weight, weight2));
    };
    runner_single_type<double>(test);
}

// NOLINTNEXTLINE
TEST_CASE("Optimizer Save Load RMSprop") {
    auto test = []<typename T>(Device device) {
        constexpr double lr = 0.1;
        constexpr double momentum = 0.2;
        std::mt19937 gen(0);
        std::vector<T> d_input = rand_vec<T>(4 * 4 * 4, gen);
        std::vector<T> d_weight = rand_vec<T>(4 * 4 * 4, gen);
        std::vector<T> d_target = rand_vec<T>(4 * 4 * 4, gen);

        Tensor input(d_input, {4, 4, 4}, device);
        Tensor weight(d_weight, {4, 4, 4}, device, true);
        Tensor target(d_target, {4, 4, 4}, device);
        optim::RMSprop rmsprop({weight}, lr, {.momentum = momentum, .center = true});

        // Step 1
        Tensor x = relu(input * weight);
        Tensor loss1 = nn::mse_loss(x, target, nn::ReductionMode::mean);
        loss1.backward();
        rmsprop.step();

        rmsprop.zero_grad();

        // Save
        rmsprop.save("optimizer.pt");
        Tensor weight2 = weight.detach();
        weight2.set_requires_grad(true);

        // Step 2
        x = relu(input * weight);
        Tensor loss2 = nn::mse_loss(x, target, nn::ReductionMode::mean);
        loss2.backward();
        rmsprop.step();

        // Reload SGD
        optim::RMSprop rmsprop2({weight2}, lr, {.momentum = momentum, .center = true});
        rmsprop2.load("optimizer.pt");

        // Step 2 from reload
        x = relu(input * weight2);
        Tensor loss3 = nn::mse_loss(x, target, nn::ReductionMode::mean);
        loss3.backward();
        rmsprop2.step();

        CHECK(allclose(weight, weight2));
    };
    runner_single_type<double>(test);
}

// NOLINTNEXTLINE
TEST_CASE("Optimizer Save Load Adam") {
    auto test = []<typename T>(Device device) {
        constexpr double lr = 0.1;
        std::mt19937 gen(0);
        std::vector<T> d_input = rand_vec<T>(4 * 4 * 4, gen);
        std::vector<T> d_weight = rand_vec<T>(4 * 4 * 4, gen);
        std::vector<T> d_target = rand_vec<T>(4 * 4 * 4, gen);

        Tensor input(d_input, {4, 4, 4}, device);
        Tensor weight(d_weight, {4, 4, 4}, device, true);
        Tensor target(d_target, {4, 4, 4}, device);
        optim::Adam adam({weight}, lr, {.weight_decay = 0.2, .use_amsgrad = true});

        // Step 1
        Tensor x = relu(input * weight);
        Tensor loss1 = nn::mse_loss(x, target, nn::ReductionMode::mean);
        loss1.backward();
        adam.step();

        adam.zero_grad();

        // Save
        adam.save("optimizer.pt");
        Tensor weight2 = weight.detach();
        weight2.set_requires_grad(true);

        // Step 2
        x = relu(input * weight);
        Tensor loss2 = nn::mse_loss(x, target, nn::ReductionMode::mean);
        loss2.backward();
        adam.step();

        // Reload SGD
        optim::Adam adam2({weight2}, lr, {.weight_decay = 0.2, .use_amsgrad = true});
        adam2.load("optimizer.pt");

        // Step 2 from reload
        x = relu(input * weight2);
        Tensor loss3 = nn::mse_loss(x, target, nn::ReductionMode::mean);
        loss3.backward();
        adam2.step();

        CHECK(allclose(weight, weight2));
    };
    runner_single_type<double>(test);
}

// NOLINTNEXTLINE
TEST_CASE("Optimizer Save Load AdamW") {
    auto test = []<typename T>(Device device) {
        constexpr double lr = 0.1;
        std::mt19937 gen(0);
        std::vector<T> d_input = rand_vec<T>(4 * 4 * 4, gen);
        std::vector<T> d_weight = rand_vec<T>(4 * 4 * 4, gen);
        std::vector<T> d_target = rand_vec<T>(4 * 4 * 4, gen);

        Tensor input(d_input, {4, 4, 4}, device);
        Tensor weight(d_weight, {4, 4, 4}, device, true);
        Tensor target(d_target, {4, 4, 4}, device);
        optim::AdamW adamw({weight}, lr, {.weight_decay = 0.2, .use_amsgrad = true});

        // Step 1
        Tensor x = relu(input * weight);
        Tensor loss1 = nn::mse_loss(x, target, nn::ReductionMode::mean);
        loss1.backward();
        adamw.step();

        adamw.zero_grad();

        // Save
        adamw.save("optimizer.pt");
        Tensor weight2 = weight.detach();
        weight2.set_requires_grad(true);

        // Step 2
        x = relu(input * weight);
        Tensor loss2 = nn::mse_loss(x, target, nn::ReductionMode::mean);
        loss2.backward();
        adamw.step();

        // Reload SGD
        optim::AdamW adamw2({weight2}, lr, {.weight_decay = 0.2, .use_amsgrad = true});
        adamw2.load("optimizer.pt");

        // Step 2 from reload
        x = relu(input * weight2);
        Tensor loss3 = nn::mse_loss(x, target, nn::ReductionMode::mean);
        loss3.backward();
        adamw2.step();

        CHECK(allclose(weight, weight2));
    };
    runner_single_type<double>(test);
}

// NOLINTNEXTLINE
TEST_CASE("Optimizer Save Load Adagrad") {
    auto test = []<typename T>(Device device) {
        constexpr double lr = 0.1;
        std::mt19937 gen(0);
        std::vector<T> d_input = rand_vec<T>(4 * 4 * 4, gen);
        std::vector<T> d_weight = rand_vec<T>(4 * 4 * 4, gen);
        std::vector<T> d_target = rand_vec<T>(4 * 4 * 4, gen);

        Tensor input(d_input, {4, 4, 4}, device);
        Tensor weight(d_weight, {4, 4, 4}, device, true);
        Tensor target(d_target, {4, 4, 4}, device);
        optim::Adagrad adagrad({weight}, lr, {.weight_decay = 0.2});

        // Step 1
        Tensor x = relu(input * weight);
        Tensor loss1 = nn::mse_loss(x, target, nn::ReductionMode::mean);
        loss1.backward();
        adagrad.step();

        adagrad.zero_grad();

        // Save
        adagrad.save("optimizer.pt");
        Tensor weight2 = weight.detach();
        weight2.set_requires_grad(true);

        // Step 2
        x = relu(input * weight);
        Tensor loss2 = nn::mse_loss(x, target, nn::ReductionMode::mean);
        loss2.backward();
        adagrad.step();

        // Reload SGD
        optim::Adagrad adagrad2({weight2}, lr, {.weight_decay = 0.2});
        adagrad2.load("optimizer.pt");

        // Step 2 from reload
        x = relu(input * weight2);
        Tensor loss3 = nn::mse_loss(x, target, nn::ReductionMode::mean);
        loss3.backward();
        adagrad2.step();

        CHECK(allclose(weight, weight2));
    };
    runner_single_type<double>(test);
}
