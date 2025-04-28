// mnist.cpp
// MNIST dataset

#include <tt/data/mnist.h>
#include <tt/device.h>
#include <tt/exception.h>
#include <tt/shape.h>
#include <tt/tensor.h>

#include <cstdint>
#include <format>
#include <fstream>
#include <iostream>
#include <ranges>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace tinytensor::data {

namespace {
auto swap_endian(uint32_t val) -> uint32_t {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

auto read_mnist(const std::string &image_path, const std::string &label_path) -> std::tuple<Tensor, Tensor, int> {
    constexpr uint64_t MAGIC_VAL1 = 2051;
    constexpr uint64_t MAGIC_VAL2 = 2049;
    std::ifstream image_file(image_path, std::ios::in | std::ios::binary);
    std::ifstream label_file(label_path, std::ios::in | std::ios::binary);
    if (image_file.fail()) {
        TT_EXCEPTION(std::format("Error trying to read image file {:s}", image_path));
    }
    if (label_file.fail()) {
        TT_EXCEPTION(std::format("Error trying to read label file {:s}", label_path));
    }

    // Read the magic and the meta data
    uint32_t magic{};
    uint32_t num_images{};
    uint32_t num_labels{};
    uint32_t rows{};
    uint32_t cols{};

    image_file.read(reinterpret_cast<char *>(&magic), 4);    // NOLINT(*-reinterpret-cast)
    magic = swap_endian(magic);
    if (magic != 2051) {
        TT_EXCEPTION(std::format("Incorrect image file magic value {:d}, expected {:d}", magic, MAGIC_VAL1));
    }

    label_file.read(reinterpret_cast<char *>(&magic), 4);    // NOLINT(*-reinterpret-cast)
    magic = swap_endian(magic);
    if (magic != MAGIC_VAL2) {
        TT_EXCEPTION(std::format("Incorrect image file magic value {:d}, expected {:d}", magic, MAGIC_VAL2));
    }

    image_file.read(reinterpret_cast<char *>(&num_images), 4);    // NOLINT(*-reinterpret-cast)
    num_images = swap_endian(num_images);
    label_file.read(reinterpret_cast<char *>(&num_labels), 4);    // NOLINT(*-reinterpret-cast)
    num_labels = swap_endian(num_labels);
    if (num_images != num_labels) {
        TT_EXCEPTION(
            std::format("Number of images ({:d}) does not equal number of labels ({:d})", num_images, num_labels)
        );
    }

    image_file.read(reinterpret_cast<char *>(&rows), 4);    // NOLINT(*-reinterpret-cast)
    rows = swap_endian(rows);
    image_file.read(reinterpret_cast<char *>(&cols), 4);    // NOLINT(*-reinterpret-cast)
    cols = swap_endian(cols);

    char label{};

    std::vector<int> labels;
    std::vector<float> images;
    std::vector<char> pixels(rows * cols, 0);

    for ([[maybe_unused]] int _ : std::views::iota(0, static_cast<int>(num_images))) {
        image_file.read(pixels.data(), rows * cols);
        for (const auto p : pixels) {
            images.push_back(static_cast<float>(p));
        }
        label_file.read(&label, 1);
        labels.push_back(static_cast<int>(label));
    }

    auto b = static_cast<int>(num_images);
    auto r = static_cast<int>(rows);
    auto c = static_cast<int>(cols);
    Tensor images_tensor(images, {b, 1, r, c}, kCPU);
    Tensor labels_tensor(labels, kCPU);
    return {images_tensor / 255, labels_tensor, b};
}
}    // namespace

MNISTDataset::MNISTDataset(const std::tuple<Tensor, Tensor, int> &data)
    : images(std::move(std::get<0>(data))), labels(std::move(std::get<1>(data))), N(std::move(std::get<2>(data))) {}

MNISTDataset::MNISTDataset(const std::string &img_path, const std::string &label_path, bool normalize)
    : MNISTDataset(read_mnist(img_path, label_path)) {
    if (normalize) {
        images = (images - images.mean().expand(images.shape())) / images.var().sqrt_().expand(images.shape());
    }
}

auto MNISTDataset::img_shape() const -> Shape {
    return images[0].shape();
}

auto MNISTDataset::size() const -> int {
    return N;
}

auto MNISTDataset::get(int idx) const -> DataType {
    return {images[idx], labels[idx]};
}

}    // namespace tinytensor::data
