// adam.cpp
// Adam optimizer
// https://arxiv.org/abs/1412.6980

#include <tt/exception.h>
#include <tt/grad_mode.h>
#include <tt/optim/optimizer.h>
#include <tt/optim/sgd.h>
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

SGD::SGD(const std::vector<std::reference_wrapper<Tensor>> &params, double learning_rate, const SGDOptions &options)
    : learning_rate_(learning_rate), options_(options) {
    add_parameters(params);
}

void SGD::add_parameters(const std::vector<std::reference_wrapper<Tensor>> &params) {
    params_.insert(params_.end(), params.begin(), params.end());
    if (options_.momentum > 0) {
        for (const auto &p : params) {
            velocities_.push_back(zeros_like(p));
        }
    }
}

void SGD::save(const std::string &path) const {
    std::ofstream optimizer_file(path, std::ios::out | std::ios::binary);

    nop::Serializer<nop::StreamWriter<std::stringstream>> serializer;

    // velocities
    std::vector<std::vector<char>> data;
    for (const auto &velocity : velocities_) {
        data.emplace_back(velocity.serialize());
    }
    serializer.Write(data);

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

void SGD::load(const std::string &path) {
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

    if (data.size() != velocities_.size()) {
        TT_EXCEPTION("Optimizer state being loaded has a different number of parameters as the current state");
    }

    // Deserialize into data structures
    for (std::size_t i = 0; i < velocities_.size(); ++i) {
        velocities_[i].deserialize(data[i]);
    }
}

// @note See https://github.com/pytorch/pytorch/issues/1099#issuecomment-289190614
void SGD::step() {
    autograd::NoGradGuard guard;
    for (std::size_t i = 0; i < params_.size(); ++i) {
        Tensor &param = params_[i].get();
        if (!param.grad()) {
            continue;
        }
        Tensor grad = param.grad()->clone();

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

        // Momentum
        if (options_.momentum != 0) {
            velocities_[i] = velocities_[i].to(param.device());
            Tensor &velocity = velocities_[i];
            velocity.mul_(options_.momentum).add_(grad);
            // Nesterov
            if (options_.use_nesterov) {
                grad = grad.add_(options_.momentum * velocity);
            } else {
                grad = velocity;
            }
        }

        // Update
        if (options_.maximize) {
            param.add_(learning_rate_ * grad);
        } else {
            param.sub_(learning_rate_ * grad);
        }
    }
}

}    // namespace tinytensor::optim
