// module.cpp
// Base module

#include <tt/device.h>
#include <tt/exception.h>
#include <tt/nn/module.h>
#include <tt/tensor.h>

#include <nop/serializer.h>
#include <nop/utility/buffer_reader.h>
#include <nop/utility/buffer_writer.h>
#include <nop/utility/stream_reader.h>
#include <nop/utility/stream_writer.h>

#include <cstddef>
#include <cstdint>
#include <format>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace tinytensor::nn {

auto Module::parameters(bool recursive) const -> std::vector<Tensor> {
    std::vector<Tensor> params;
    get_params(params, recursive);
    return params;
}
void Module::get_params(std::vector<Tensor> &params, bool recursive) const {
    for (auto &p : params_) {
        params.push_back(*p);
    }
    if (recursive) {
        for (auto &[_, m] : modules_) {
            m.get().get_params(params, recursive);
        }
    }
}

auto Module::parameters_for_optimizer(bool recursive) const -> std::vector<std::reference_wrapper<Tensor>> {
    std::vector<std::reference_wrapper<Tensor>> params;
    get_params(params, recursive);
    return params;
}
void Module::get_params(std::vector<std::reference_wrapper<Tensor>> &params, bool recursive) const {
    for (auto &p : params_) {
        params.emplace_back(*p);
    }
    if (recursive) {
        for (auto &[_, m] : modules_) {
            m.get().get_params(params, recursive);
        }
    }
}

auto Module::num_params() const -> int64_t {
    int64_t count = 0;
    for (const auto &p : params_) {
        count += p->numel();
    }
    for (const auto &[_, m] : modules_) {
        count += m.get().num_params();
    }
    return count;
}

auto Module::serialize() const -> std::vector<std::vector<char>> {
    std::vector<std::vector<char>> data;
    for (const auto &param : parameters()) {
        data.emplace_back(param.serialize());
    }
    return data;
}
void Module::deserialize(const std::vector<std::vector<char>> &data) {
    std::vector<std::reference_wrapper<Tensor>> params;
    get_params(params, true);
    if (data.size() != params.size()) {
        TT_EXCEPTION(
            std::format(
                "Size of serialized data {:d} does not match number of parameters {:d}",
                data.size(),
                params.size()
            )
        );
    }
    for (std::size_t i = 0; i < params.size(); ++i) {
        params[i].get().deserialize(data[i]);
    }
}

void Module::save(const std::string &path) const {
    std::ofstream tensor_file(path, std::ios::out | std::ios::binary);

    nop::Serializer<nop::StreamWriter<std::stringstream>> serializer;
    std::vector<std::vector<char>> data = serialize();
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

    tensor_file.write(&byte_data[0], static_cast<std::streamsize>(byte_data.size()));
    tensor_file.close();
}

void Module::load(const std::string &path) {
    std::ifstream tensor_file(path, std::ios::in | std::ios::binary);

    // Get file size
    std::streampos fileSize;
    tensor_file.seekg(0, std::ios::end);
    fileSize = tensor_file.tellg();
    tensor_file.seekg(0, std::ios::beg);

    // Read
    std::vector<char> byte_data(static_cast<std::size_t>(fileSize));
    tensor_file.read(&byte_data[0], fileSize);
    tensor_file.close();

    std::stringstream ss;
    ss.write(byte_data.data(), std::streamsize(byte_data.size()));
    nop::Deserializer<nop::StreamReader<std::stringstream>> deserializer{std::move(ss)};

    std::vector<std::vector<char>> data;
    deserializer.Read(&data);
    deserialize(data);
}

void Module::zero_grad() {
    for (auto &p : params_) {
        p->clear_grad();
    }
    for (auto &[_, m] : modules_) {
        m.get().zero_grad();
    }
}

void Module::to(Device device) {
    for (auto &p : params_) {
        p->set_from(p->to(device));
    }
    for (auto &[_, m] : modules_) {
        m.get().to(device);
    }
}

void Module::apply(const std::function<void(Module &)> &func, bool recursive) {
    func(*this);
    if (recursive) {
        for (auto &[_, m] : modules_) {
            m.get().apply(func);
        }
    }
}

void Module::register_param(std::shared_ptr<Tensor> param) {
    params_.push_back(std::move(param));
}
void Module::register_module(Module &module) {
    modules_.emplace_back(std::to_string(modules_.size()), module);
}
void Module::register_module(Module &module, const std::string &name) {
    modules_.emplace_back(name, module);
}

void Module::train(bool is_train) {
    is_train_ = is_train;
    for (auto &[_, m] : modules_) {
        m.get().train(is_train);
    }
}
void Module::eval() {
    train(false);
}

void Module::pretty_print(std::ostream &os) const {
    os << name();
}

// This follows the libtorch implementation
void Module::pretty_print_recursive(std::ostream &os, const std::string &indentation) const {
    pretty_print(os);
    if (!modules_.empty()) {
        os << "(\n";
        const std::string next_indentation = indentation + "  ";
        for (const auto &[name, m] : modules_) {
            os << next_indentation << "(" << name << "): ";
            m.get().pretty_print_recursive(os, next_indentation);
            os << "\n";
        }
        os << indentation << ")";
    }
}

std::ostream &operator<<(std::ostream &os, Module &module) {
    module.pretty_print_recursive(os, "");
    return os;
}

}    // namespace tinytensor::nn
