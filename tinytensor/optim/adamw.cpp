// adamw.cpp
// AdamWW optimizer
// https://arxiv.org/abs/1711.05101

#include <tt/exception.h>
#include <tt/grad_mode.h>
#include <tt/optim/adamw.h>
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

AdamW::AdamW(
    const std::vector<std::reference_wrapper<Tensor>> &params,
    double learning_rate,
    const AdamWOptions &options
)
    : learning_rate_(learning_rate), options_(options) {
    add_parameters(params);
}

void AdamW::add_parameters(const std::vector<std::reference_wrapper<Tensor>> &params) {
    params_.insert(params_.end(), params.begin(), params.end());
    for (const auto &p : params) {
        first_moments_.push_back(zeros_like(p));
        second_moments_.push_back(zeros_like(p));
        second_moments_max_.push_back(zeros_like(p));
        steps_.push_back(0);
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

void AdamW::save(const std::string &path) const {
    std::ofstream optimizer_file(path, std::ios::out | std::ios::binary);

    nop::Serializer<nop::StreamWriter<std::stringstream>> serializer;

    // Moments
    std::vector<std::vector<char>> data_first_moments = serialize_tensor_list(first_moments_);
    serializer.Write(data_first_moments);
    std::vector<std::vector<char>> data_second_moments = serialize_tensor_list(second_moments_);
    serializer.Write(data_second_moments);
    std::vector<std::vector<char>> data_second_moments_max = serialize_tensor_list(second_moments_max_);
    serializer.Write(data_second_moments_max);

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

void AdamW::load(const std::string &path) {
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
    std::vector<std::vector<char>> data_first_moments;
    deserializer.Read(&data_first_moments);
    std::vector<std::vector<char>> data_second_moments;
    deserializer.Read(&data_second_moments);
    std::vector<std::vector<char>> data_second_moments_max;
    deserializer.Read(&data_second_moments_max);
    std::vector<int> steps;
    deserializer.Read(&steps);

    if (steps.size() != steps_.size() || data_first_moments.size() != first_moments_.size()
        || data_second_moments.size() != second_moments_.size()
        || data_second_moments_max.size() != second_moments_max_.size())
    {
        TT_EXCEPTION("Optimizer state being loaded has a different number of parameters as the current state");
    }

    // Deserialize into data structures
    for (std::size_t i = 0; i < data_first_moments.size(); ++i) {
        first_moments_[i].deserialize(data_first_moments[i]);
    }
    for (std::size_t i = 0; i < data_second_moments.size(); ++i) {
        second_moments_[i].deserialize(data_second_moments[i]);
    }
    for (std::size_t i = 0; i < data_second_moments_max.size(); ++i) {
        second_moments_max_[i].deserialize(data_second_moments_max[i]);
    }
    steps_ = steps;
}

void AdamW::step() {
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
        // Modify parameters directly instead of gradient first
        if (options_.weight_decay != 0) {
            switch (options_.regularization_mode) {
            case RegularizationMode::l1:
                param.sub_(learning_rate_ * options_.weight_decay * sign(param));
                break;
            case RegularizationMode::l2:
                param.sub_(learning_rate_ * options_.weight_decay * param);
                break;
            }
        }

        auto beta1 = options_.betas.beta1;
        auto beta2 = options_.betas.beta2;
        first_moments_[i] = first_moments_[i].to(param.device());
        second_moments_[i] = second_moments_[i].to(param.device());
        second_moments_max_[i] = second_moments_max_[i].to(param.device());

        Tensor &first_moment = first_moments_[i];
        Tensor &second_moment = second_moments_[i];
        Tensor &second_moment_max = second_moments_max_[i];
        int &step = steps_[i];
        ++step;

        first_moment.mul_(beta1).add_((1 - beta1) * grad);
        second_moment.mul_(beta2).add_((1 - beta2) * grad * grad);

        const Tensor first_moment_hat = first_moment / (1 - std::pow(beta1, step));
        const Tensor second_moment_hat = second_moment / (1 - std::pow(beta2, step));
        if (options_.use_amsgrad) {
            second_moment_max = maximum(second_moment_max, second_moment_hat);
            param.sub_(learning_rate_ * first_moment_hat / (sqrt(second_moment_max) + options_.eps));
        } else {
            param.sub_(learning_rate_ * first_moment_hat / (sqrt(second_moment_hat) + options_.eps));
        }
    }
}

}    // namespace tinytensor::optim
