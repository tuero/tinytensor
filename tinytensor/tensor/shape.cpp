// shape.cpp
// Array shape type definition

#include <tt/exception.h>
#include <tt/shape.h>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <numeric>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace tinytensor {

namespace {
std::string get_error_message(int index, const size_t num_dims) {
    return "Shape index " + std::to_string(index) + " out of bounds for shape with " + std::to_string(num_dims)
           + " dimensions";
}
}    // namespace

Shape::Shape(const std::vector<int> &_dims)
    : dims(_dims) {};
Shape::Shape(std::vector<int> &&_dims)
    : dims(std::move(_dims)) {};
Shape::Shape(std::initializer_list<int> _dims)
    : dims(_dims) {};

auto Shape::numel() const noexcept -> int {
    return std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<>());
}

auto Shape::ndim() const noexcept -> int {
    return static_cast<int>(dims.size());
}
auto Shape::size() const noexcept -> int {
    return static_cast<int>(dims.size());
}

auto Shape::operator[](int dim) -> int & {
    int N = size();
    dim = (dim < 0) ? (dim + N) % N : dim;
    if (dim < 0 || dim >= static_cast<int>(dims.size())) {
        TT_EXCEPTION(get_error_message(dim, dims.size()));
    }
    return dims[static_cast<std::size_t>(dim)];
}

auto Shape::operator[](int dim) const -> int {
    int N = size();
    dim = (dim < 0) ? (dim + N) % N : dim;
    if (dim < 0 || dim >= static_cast<int>(dims.size())) {
        TT_EXCEPTION(get_error_message(dim, dims.size()));
    }
    return dims[static_cast<std::size_t>(dim)];
}

auto Shape::operator==(const Shape &other) const noexcept -> bool {
    return dims == other.dims;
}

auto Shape::operator!=(const Shape &other) const noexcept -> bool {
    return dims != other.dims;
}

auto Shape::operator==(const std::initializer_list<int> &other) const noexcept -> bool {
    return dims.size() == other.size() && std::equal(std::begin(dims), std::end(dims), std::begin(other));
}

auto Shape::operator!=(const std::initializer_list<int> &other) const noexcept -> bool {
    return !(this->operator==(other));
}

auto Shape::pop(int dim) -> int {
    int N = size();
    dim = (dim < 0) ? (dim + N) % N : dim;
    if (dim < 0 || dim >= static_cast<int>(dims.size())) {
        TT_EXCEPTION(get_error_message(dim, dims.size()));
    }
    int size = dims[static_cast<std::size_t>(dim)];
    dims.erase(dims.begin() + dim);
    return size;
}

void Shape::insert(int size, int dim) {
    int N = this->size();
    dim = (dim < 0) ? (dim + N + 1) % (N + 1) : dim;
    if (dim < 0 || dim > static_cast<int>(dims.size())) {
        TT_EXCEPTION(get_error_message(dim, dims.size()));
    }
    dims.insert(dims.begin() + dim, size);
}

// Convert Shape to string
auto Shape::to_string() const -> std::string {
    std::string str = "(";
    for (int i = 0; i < ndim(); ++i) {
        str += std::to_string(dims[static_cast<std::size_t>(i)]);
        str += (i == ndim() - 1) ? "" : ", ";
    }
    str += ")";
    return str;
}

auto Shape::to_stride() const -> Shape {
    std::vector<int> stride(static_cast<std::size_t>(size()), 1);
    for (int i = size() - 2; i >= 0; --i) {
        stride[static_cast<std::size_t>(i)] = stride[static_cast<std::size_t>(i + 1)] * this->operator[](i + 1);
    }
    return Shape(std::move(stride));
}

auto Shape::to_vec() const -> const std::vector<int> & {
    return dims;
}

auto operator<<(std::ostream &os, const Shape &shape) -> std::ostream & {
    os << shape.to_string();
    return os;
}

auto to_string(const Shape &shape) -> std::string {
    return shape.to_string();
}

}    // namespace tinytensor
