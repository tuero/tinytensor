// mnist_cnn_example.cpp
// MNIST CNN learning example
// Download dataset from https://git-disl.github.io/GTDLBench/datasets/mnist_datasets/
// https://github.com/pytorch/examples/blob/main/mnist/main.py

#include <tt/data/dataset.h>
#include <tt/data/mnist.h>
#include <tt/device.h>
#include <tt/exception.h>
#include <tt/grad_mode.h>
#include <tt/nn/conv2d.h>
#include <tt/nn/dropout.h>
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
#include <utility>

using namespace tinytensor;

// Neural Net
// https://github.com/pytorch/examples/blob/main/mnist/main.py
class Net : public nn::Module {
public:
    Net()
        : conv1(1, 32, 3, 1, 0),
          conv2(32, 64, 3, 1, 0),
          linear1(9216, 128),
          linear2(128, 10),
          dropout1(0.25),
          dropout2(0.5) {
        register_module(conv1);
        register_module(conv2);
        register_module(linear1);
        register_module(linear2);
        register_module(dropout1);
        register_module(dropout2);
    }

    [[nodiscard]] auto name() const -> std::string override {
        return "Net";
    }

    [[nodiscard]] auto forward(Tensor input) -> Tensor {
        Tensor result = relu(conv1.forward(input));
        result = relu(conv2.forward(result));
        result = max_pool2d(result, 2, 2, 0);
        result = dropout1.forward(result);
        result = result.flatten(1);
        result = relu(linear1.forward(result));
        result = dropout2.forward(result);
        result = linear2.forward(result);
        return result;
    }

private:
    nn::Conv2d conv1;
    nn::Conv2d conv2;
    nn::Linear linear1;
    nn::Linear linear2;
    nn::Dropout dropout1;
    nn::Dropout dropout2;
};

#ifdef TT_CUDA
constexpr Device device = kCUDA;
#else
constexpr Device device = kCPU;
#endif

int main(int argc, char *argv[]) {
    if (argc != 5) {
        TT_EXCEPTION(
            "Usage: ./mnist_cnn_example train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte "
            "t10k-labels-idx1-ubyte"
        );
    }

    const std::string train_img_path = argv[1];      // NOLINT(*-pointer-arithmetic)
    const std::string train_label_path = argv[2];    // NOLINT(*-pointer-arithmetic)
    const std::string test_img_path = argv[3];       // NOLINT(*-pointer-arithmetic)
    const std::string test_label_path = argv[4];     // NOLINT(*-pointer-arithmetic)

    constexpr double train_val_ratio = 0.7;
    constexpr double lr = 3e-4;
    constexpr int epochs = 2;
    constexpr int batch_size = 256;
    constexpr uint64_t seed = 0;

    set_default_generator_seed(seed);

    // Dataset and splits
    data::MNISTDataset dataset_train(train_img_path, train_label_path, true);
    data::MNISTDataset dataset_test(test_img_path, test_label_path, true);
    int train_size = static_cast<int>(dataset_train.size() * train_val_ratio);
    int val_size = dataset_train.size() - train_size;
    auto [train_data, validate_data] = data::random_split(std::move(dataset_train), seed, train_size, val_size);
    auto test_data = data::DatasetView(std::move(dataset_test));
    auto train_loader = data::DataLoader(train_data, batch_size, true, seed);
    auto validate_loader = data::DataLoader(validate_data, batch_size, false);
    auto test_loader = data::DataLoader(test_data, batch_size, false);

    Net net;
    optim::Adam optim(net.parameters_for_optimizer(), lr);
    net.to(device);

    // Train/Validate
    for ([[maybe_unused]] int epoch : std::views::iota(0, epochs)) {
        // Train
        net.train();
        double epoch_train_loss = 0;
        int i = -1;
        for (auto [x, y] : train_loader) {
            x = x.to(device);
            y = y.to(device);

            optim.zero_grad();

            Tensor y_hat = net.forward(x);
            Tensor loss = nn::cross_entropy_loss(y_hat, y.flatten());
            auto loss_val = loss.item<double>();
            epoch_train_loss += loss_val / train_loader.size();
            loss.backward();

            optim.step();
            std::cout << std::format(
                "Epoch: {:d}, train step: {:d}/{:d}, loss: {:f}",
                epoch,
                ++i,
                train_loader.size(),
                loss_val
            ) << std::endl;
        }

        // Validate
        net.eval();
        double epoch_validate_loss = 0;
        i = -1;

        // Guard for no grads being tracked
        {
            const autograd::NoGradGuard guard;
            for (auto [x, y] : validate_loader) {
                x = x.to(device);
                y = y.to(device);
                Tensor y_hat = net.forward(x);
                Tensor loss = nn::cross_entropy_loss(y_hat, y.flatten());
                auto loss_val = loss.item<double>();
                epoch_validate_loss += loss_val / validate_loader.size();
                std::cout << std::format(
                    "Epoch: {:d}, val step step: {:d}/{:d}, loss: {:f}",
                    epoch,
                    ++i,
                    validate_loader.size(),
                    loss_val
                ) << std::endl;
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
    // Demonstrate how to save and load model
    net.save("model.pt");

    Net net2;
    net2.to(device);
    net2.load("model.pt");

    net2.eval();
    int num_correct = 0;
    int total_samples = 0;
    for (auto [x, y] : test_loader) {
        x = x.to(device);
        y = y.to(device);
        Tensor y_hat = net2.forward(x);
        Tensor pred_labels = y_hat.argmax(-1, true);
        num_correct += eq(pred_labels, y).sum().item<int>();
        total_samples += y.numel();
    }
    double accuracy = (static_cast<double>(num_correct) / total_samples) * 100;
    std::cout << std::format("Test Accuracy: {:d} / {:d} = {:.2f}%", num_correct, total_samples, accuracy) << std::endl;

    // Test net on held out set
    // Demonstrate how to serialize and deserialize model
    net.save("model.pt");

    Net net3;
    net3.to(device);

    net3.deserialize(net.serialize());

    net3.eval();
    num_correct = 0;
    total_samples = 0;
    for (auto [x, y] : test_loader) {
        x = x.to(device);
        y = y.to(device);
        Tensor y_hat = net3.forward(x);
        Tensor pred_labels = y_hat.argmax(-1, true);
        num_correct += eq(pred_labels, y).sum().item<int>();
        total_samples += y.numel();
    }
    accuracy = (static_cast<double>(num_correct) / total_samples) * 100;
    std::cout << std::format("Test Accuracy: {:d} / {:d} = {:.2f}%", num_correct, total_samples, accuracy) << std::endl;

    return 0;
}
