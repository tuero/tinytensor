// adagrad.cpp
// Adagrad optimizer
// https://jmlr.org/papers/v12/duchi11a.html

#include <tt/exception.h>
#include <tt/grad_mode.h>
#include <tt/optim/adagrad.h>
#include <tt/optim/optimizer.h>
#include <tt/tensor.h>

#include <nop/serializer.h>
#include <nop/utility/buffer_reader.h>
#include <nop/utility/buffer_writer.h>
#include <nop/utility/stream_reader.h>
#include <nop/utility/stream_writer.h>

#include <cmath>
#include <cstddef>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace tinytensor::optim {

Adagrad::Adagrad(
    const std::vector<std::reference_wrapper<Tensor>> &params,
    double learning_rate,
    const AdagradOptions &options
)
    : learning_rate_(learning_rate), options_(options) {
    add_parameters(params);
}

void Adagrad::add_parameters(const std::vector<std::reference_wrapper<Tensor>> &params) {
    params_.insert(params_.end(), params.begin(), params.end());
    for (const auto &p : params) {
        steps_.push_back(0);
        state_sums_.push_back(zeros_like(p));
    }
}

void Adagrad::save(const std::string &path) const {
    std::ofstream optimizer_file(path, std::ios::out | std::ios::binary);

    nop::Serializer<nop::StreamWriter<std::stringstream>> serializer;

    // State sums
    std::vector<std::vector<char>> data;
    for (const auto &state_sum : state_sums_) {
        data.emplace_back(state_sum.serialize());
    }
    serializer.Write(data);

    // Steps
    serializer.Write(steps_);

    auto &ss = serializer.writer().stream();
    // discover size of data in stream
    ss.seekg(0, std::ios::beg);
    auto bof = ss.tellg();
    ss.seekg(0, std::ios::end);
    auto stream_size = std::size_t(ss.tellg() - bof);
    ss.seekg(0, std::ios::beg);

    // Make vector long enough
    std::vector<char> byte_data(stream_size);

    // read directly in
    ss.read(byte_data.data(), std::streamsize(byte_data.size()));

    optimizer_file.write(&byte_data[0], static_cast<std::streamsize>(byte_data.size()));
    optimizer_file.close();
}

void Adagrad::load(const std::string &path) {
    std::ifstream optimizer_file(path, std::ios::in | std::ios::binary);

    // Get file size
    std::streampos fileSize;
    optimizer_file.seekg(0, std::ios::end);
    fileSize = optimizer_file.tellg();
    optimizer_file.seekg(0, std::ios::beg);

    // Read flat data
    std::vector<char> byte_data(static_cast<std::size_t>(fileSize));
    optimizer_file.read(&byte_data[0], fileSize);
    optimizer_file.close();

    std::stringstream ss;
    ss.write(byte_data.data(), std::streamsize(byte_data.size()));
    nop::Deserializer<nop::StreamReader<std::stringstream>> deserializer{std::move(ss)};

    // Deserialize read
    std::vector<std::vector<char>> data;
    deserializer.Read(&data);
    std::vector<int> steps;
    deserializer.Read(&steps);

    if (steps.size() != steps_.size() || data.size() != state_sums_.size()) {
        TT_EXCEPTION("Optimizer state being loaded has a different number of parameters as the current state");
    }

    // Deserialize into data structures
    for (std::size_t i = 0; i < state_sums_.size(); ++i) {
        state_sums_[i].deserialize(data[i]);
    }
    steps_ = steps;
}

void Adagrad::step() {
    autograd::NoGradGuard guard;
    for (std::size_t i = 0; i < params_.size(); ++i) {
        Tensor &param = params_[i].get();
        if (!param.grad()) {
            continue;
        }
        Tensor grad = param.grad()->clone();

        if (options_.maximize) {
            grad.mul_(-1);
        }

        int &step = steps_[i];
        ++step;

        double lr = learning_rate_ / (1.0 + (step - 1) * options_.learning_rate_decay);

        // Weight decay
        if (options_.weight_decay != 0) {
            switch (options_.regularization_mode) {
            case RegularizationMode::l1:
                grad.add_(options_.weight_decay * sign(param));
                break;
            case RegularizationMode::l2:
                grad.add_(options_.weight_decay * param);
                break;
            }
        }

        state_sums_[i] = state_sums_[i].to(param.device());
        Tensor &state_sum = state_sums_[i];
        state_sum.add_(grad * grad);

        param.sub_(lr * grad / (sqrt(state_sum) + options_.eps));
    }
}

}    // namespace tinytensor::optim
