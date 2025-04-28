// rmsprop.cpp
// RMSprop optimizer
// https://arxiv.org/abs/1308.0850

#include <tt/exception.h>
#include <tt/grad_mode.h>
#include <tt/optim/optimizer.h>
#include <tt/optim/rmsprop.h>
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

RMSprop::RMSprop(
    const std::vector<std::reference_wrapper<Tensor>> &params,
    double learning_rate,
    const RMSpropOptions &options
)
    : learning_rate_(learning_rate), options_(options) {
    add_parameters(params);
}

void RMSprop::add_parameters(const std::vector<std::reference_wrapper<Tensor>> &params) {
    params_.insert(params_.end(), params.begin(), params.end());
    for (const auto &p : params) {
        square_averages_.push_back(zeros_like(p));
        if (options_.momentum > 0) {
            velocities_.push_back(zeros_like(p));
        }
        if (options_.center) {
            centers_.push_back(zeros_like(p));
        }
    }
}

namespace {
auto serialize_tensor_list(const std::vector<Tensor> &tensors) -> std::vector<std::vector<char>> {
    std::vector<std::vector<char>> data;
    for (const auto &tensor : tensors) {
        data.emplace_back(tensor.serialize());
    }
    return data;
}
}    // namespace

void RMSprop::save(const std::string &path) const {
    std::ofstream optimizer_file(path, std::ios::out | std::ios::binary);

    nop::Serializer<nop::StreamWriter<std::stringstream>> serializer;

    // Moments
    std::vector<std::vector<char>> data_square_averages = serialize_tensor_list(square_averages_);
    serializer.Write(data_square_averages);
    std::vector<std::vector<char>> data_velocities = serialize_tensor_list(velocities_);
    serializer.Write(data_velocities);
    std::vector<std::vector<char>> data_centers = serialize_tensor_list(centers_);
    serializer.Write(data_centers);

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

void RMSprop::load(const std::string &path) {
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
    std::vector<std::vector<char>> data_square_averages;
    deserializer.Read(&data_square_averages);
    std::vector<std::vector<char>> data_velocities;
    deserializer.Read(&data_velocities);
    std::vector<std::vector<char>> data_centers;
    deserializer.Read(&data_centers);

    if (data_square_averages.size() != square_averages_.size() || data_velocities.size() != velocities_.size()
        || data_centers.size() != centers_.size())
    {
        TT_EXCEPTION("Optimizer state being loaded has a different number of parameters as the current state");
    }

    // Deserialize into data structures
    for (std::size_t i = 0; i < data_square_averages.size(); ++i) {
        square_averages_[i].deserialize(data_square_averages[i]);
    }
    for (std::size_t i = 0; i < data_velocities.size(); ++i) {
        velocities_[i].deserialize(data_velocities[i]);
    }
    for (std::size_t i = 0; i < data_centers.size(); ++i) {
        centers_[i].deserialize(data_centers[i]);
    }
}

void RMSprop::step() {
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

        square_averages_[i] = square_averages_[i].to(param.device());
        Tensor &square_average = square_averages_[i];

        square_average.mul_(options_.alpha).add_((1.0 - options_.alpha) * grad * grad);
        Tensor moving_average = square_average;

        // Centered
        if (options_.center) {
            centers_[i] = centers_[i].to(param.device());
            Tensor &center = centers_[i];
            center.mul_(options_.alpha).add_((1 - options_.alpha) * grad);
            moving_average = moving_average - (center * center);
        }

        // Update
        if (options_.momentum > 0) {
            velocities_[i] = velocities_[i].to(param.device());
            Tensor &velocity = velocities_[i];
            velocity.mul_(options_.momentum).add_(grad / (sqrt(moving_average) + options_.eps));
            param.sub_(learning_rate_ * velocity);
        } else {
            param.sub_(learning_rate_ * grad / (sqrt(moving_average) + options_.eps));
        }
    }
}

}    // namespace tinytensor::optim
