// mlp_example.cpp
// MLP learning example

#include <tt/data/dataset.h>
#include <tt/device.h>
#include <tt/grad_mode.h>
#include <tt/nn/linear.h>
#include <tt/nn/loss.h>
#include <tt/nn/module.h>
#include <tt/optim/adam.h>
#include <tt/random.h>
#include <tt/tensor.h>

#include <cstdint>
#include <format>
#include <iostream>
#include <ranges>
#include <string>
#include <tuple>
#include <utility>

using namespace tinytensor;

// Dataset representing sin(2*x) with noise
struct Dataset {
    using DataType = std::tuple<Tensor, Tensor>;

    Dataset(double start, double stop, int size, double noise_level, Device device)
        : N(size), x(linspace(start, stop, true, {size}, TensorOptions().device(device))), y(sin(2 * x)) {
        // Add noise to dataset
        double noise_coef = (y.max() - y.min()).item<double>() * noise_level;
        Tensor noise = zeros_like(y).normal_(0, 1) * noise_coef;
        y = y + noise;
    }

    [[nodiscard]] auto size() const -> int {
        return N;
    }
    [[nodiscard]] auto get(int idx) const -> DataType {
        return {x[idx], y[idx]};
    }

    int N;
    Tensor x;
    Tensor y;
};

// Neural Net
class Net : public nn::Module {
public:
    Net(int input_size, int output_size)
        : layer1(input_size, 32), layer2(32, output_size) {
        register_module(layer1);
        register_module(layer2);
    }

    [[nodiscard]] auto name() const -> std::string override {
        return "Net";
    }

    [[nodiscard]] auto forward(Tensor input) -> Tensor {
        Tensor result = layer1.forward(input);
        result = relu(result);
        result = layer2.forward(result);
        return result;
    }

private:
    nn::Linear layer1;
    nn::Linear layer2;
};

int main() {
    constexpr int dataset_size = 1000;
    constexpr double dataset_start = -10;
    constexpr double dataset_stop = 10;
    constexpr double noise_level = 0.1;
    constexpr double lr = 3e-4;
    constexpr int epochs = 100;
    constexpr int batch_size = 32;
    constexpr uint64_t seed = 0;
    constexpr Device device = kCPU;

    set_default_generator_seed(seed);

    // Dataset and splits
    Dataset dataset(dataset_start, dataset_stop, dataset_size, noise_level, device);
    auto [train_data, validate_data, test_data] = data::random_split(std::move(dataset), seed, 700, 200, 100);
    auto train_loader = data::DataLoader(train_data, batch_size, true, seed);
    auto validate_loader = data::DataLoader(validate_data, batch_size, false);
    auto test_loader = data::DataLoader(test_data, batch_size, false);

    Net net(1, 1);
    optim::Adam optim(net.parameters_for_optimizer(), lr);
    net.to(device);

    std::cout << "start" << std::endl;

    // Train/Validate
    for ([[maybe_unused]] int epoch : std::views::iota(0, epochs)) {
        // Train
        net.train();
        double epoch_train_loss = 0;
        for (auto [x, y] : train_loader) {
            x = x.to(device);
            y = y.to(device);
            optim.zero_grad();

            Tensor y_hat = net.forward(x);
            Tensor loss = nn::mse_loss(y_hat, y, nn::ReductionMode::mean);
            epoch_train_loss += loss.item<double>() / train_loader.size();
            loss.backward();

            optim.step();
        }

        // Validate
        net.eval();
        double epoch_validate_loss = 0;

        // Guard for no grads being tracked
        {
            const autograd::NoGradGuard guard;
            for (auto [x, y] : validate_loader) {
                x = x.to(device);
                y = y.to(device);
                Tensor y_hat = net.forward(x);
                Tensor loss = nn::mse_loss(y_hat, y, nn::ReductionMode::mean);
                epoch_validate_loss += loss.item<double>() / validate_loader.size();
            }
            std::cout << std::format(
                "Epoch: {:d}, Train Loss: {:f}, Validate Loss: {:f}",
                epoch,
                epoch_train_loss,
                epoch_validate_loss
            ) << std::endl;
        }
    }

    // Test net on held out set
    net.eval();
    double test_loss = 0;
    for (auto [x, y] : test_loader) {
        x = x.to(device);
        y = y.to(device);
        Tensor y_hat = net.forward(x);
        Tensor loss = nn::mse_loss(y_hat, y, nn::ReductionMode::mean);
        test_loss += loss.item<double>() / test_loader.size();
    }
    std::cout << std::format("Test Loss: {:f}", test_loss) << std::endl;

    return 0;
}
