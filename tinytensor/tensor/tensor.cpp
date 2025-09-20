// tensor.cpp
// Multi-dimensional tensor class

#include <tt/autograd.h>
#include <tt/concepts.h>
#include <tt/device.h>
#include <tt/exception.h>
#include <tt/grad_mode.h>
#include <tt/index.h>
#include <tt/random.h>
#include <tt/scalar.h>
#include <tt/shape.h>
#include <tt/tensor.h>
#include <tt/util.h>

#include "autograd/backward.h"
#include "autograd/binary_ops.h"
#include "autograd/clamp_ops.h"
#include "autograd/conv_ops.h"
#include "autograd/index_ops.h"
#include "autograd/joining_ops.h"
#include "autograd/misc_ops.h"
#include "autograd/reduce_ops.h"
#include "autograd/shape_ops.h"
#include "autograd/unary_ops.h"
#include "tensor/backend/common/dispatch.h"
#include "tensor/backend/common/unary.h"
#include "tensor/backend_register.h"

#include <nop/serializer.h>
#include <nop/utility/buffer_reader.h>
#include <nop/utility/buffer_writer.h>
#include <nop/utility/stream_reader.h>
#include <nop/utility/stream_writer.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <format>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <ranges>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace tinytensor {

// ------------------------------------------------
// Utility Helpers
// ------------------------------------------------
namespace {

// NOLINTNEXTLINE(*-macro-usage)
#define CHECK_VALID_TENSOR(t)                         \
    if (!(t).has_storage()) {                         \
        TT_EXCEPTION("Detected unitialized Tensor."); \
    }

// Check and convert inclusing indexing with negative indexing
// NOLINTNEXTLINE(*-macro-usage)
#define CHECK_VALID_INDEX_INCLUSIVE(dim, N)                                                   \
    if (dim > N || dim < -N - 1) {                                                            \
        TT_EXCEPTION(std::format("Invalid dim, expected to be in range[{}, {}]", -N - 1, N)); \
    }
inline auto indexing_inclusive(int dim, const Shape &shape) -> int {
    const auto N = shape.ndim();
    return (dim < 0) ? (dim + N + 1) % (N + 1) : dim;
}

// Check and convert exclusive indexing with negative indexing
// NOLINTNEXTLINE(*-macro-usage)
#define CHECK_VALID_INDEX_EXCLUSIVE(dim, N)                                                   \
    if (dim >= N || dim < -N) {                                                               \
        TT_EXCEPTION(std::format("Invalid dim, expected to be in range[{}, {}]", -N, N - 1)); \
    }
inline auto indexing_exclusive(int dim, const Shape &shape) -> int {
    const auto N = shape.ndim();
    return (dim < 0) ? (dim + N) % N : dim;
}

// NOLINTNEXTLINE(*-macro-usage)
#define CHECK_EMPTY_SHAPE(shape)            \
    if (shape.numel() == 0) {               \
        TT_EXCEPTION("Given empty shape."); \
    }

// Check if all tensors have the same of various properties
template <typename... Args>
    requires IsAllOf<Tensor, Args...>
auto is_same_device(const Tensor &a0, const Args &...args) -> bool {
    return ((args.device() == a0.device()) && ... && true);
}

template <typename... Args>
    requires IsAllOf<Tensor, Args...>
auto is_same_shape(const Tensor &a0, const Args &...args) -> bool {
    return ((args.shape() == a0.shape()) && ... && true);
}

template <typename... Args>
    requires IsAllOf<Tensor, Args...>
auto is_same_type(const Tensor &a0, const Args &...args) -> bool {
    return ((args.dtype() == a0.dtype()) && ... && true);
}

// String concatenation of tensor properties for error msg
template <typename... Args>
    requires IsAllOf<Tensor, Args...>
auto concat_device_str(const Tensor &a0, const Args &...args) -> std::string {
    std::ostringstream stream;
    stream << a0.device();
    ((stream << ", " << args.device()), ...);
    return stream.str();
}
auto concat_device_str(const TensorList &tensors) -> std::string {
    std::ostringstream ss;
    std::string delim = "";
    for (const auto &tensor : tensors) {
        ss << delim;
        ss << tensor.device();
        delim = ", ";
    }
    return ss.str();
}

template <typename... Args>
    requires IsAllOf<Tensor, Args...>
auto concat_shape_str(const Tensor &a0, const Args &...args) -> std::string {
    std::ostringstream stream;
    stream << a0.shape();
    ((stream << ", " << args.shape()), ...);
    return stream.str();
}
auto concat_shape_str(const TensorList &tensors) -> std::string {
    std::ostringstream ss;
    std::string delim = "";
    for (const auto &tensor : tensors) {
        ss << delim;
        ss << tensor.shape();
        delim = ", ";
    }
    return ss.str();
}

template <typename... Args>
    requires IsAllOf<Tensor, Args...>
auto concat_dtype_str(const Tensor &a0, const Args &...args) -> std::string {
    std::ostringstream stream;
    stream << a0.dtype();
    ((stream << ", " << args.dtype()), ...);
    return stream.str();
}

template <typename T>
auto vec_to_str(const std::vector<T> &vec) -> std::string {
    std::ostringstream ss;
    std::string delim = "";
    ss << "(";
    for (const auto &v : vec) {
        ss << delim;
        ss << v;
        delim = ", ";
    }
    ss << ")";
    return ss.str();
}

// NOLINTNEXTLINE*(-macro-usage)
#define CHECK_SAME_DEVICE(...)                                                                                      \
    if (!is_same_device(__VA_ARGS__)) {                                                                             \
        TT_EXCEPTION(                                                                                               \
            std::format("Expected inputs to be on same device, given devices {:s}", concat_device_str(__VA_ARGS__)) \
        );                                                                                                          \
    }

// NOLINTNEXTLINE*(-macro-usage)
#define CHECK_SAME_SHAPE(...)                                                                                     \
    if (!is_same_shape(__VA_ARGS__)) {                                                                            \
        TT_EXCEPTION(                                                                                             \
            std::format("Expected inputs to be the same shape, given shapes {:s}", concat_shape_str(__VA_ARGS__)) \
        );                                                                                                        \
    }

// NOLINTNEXTLINE*(-macro-usage)
#define CHECK_SAME_TYPE(...)                                                                                      \
    if (!is_same_type(__VA_ARGS__)) {                                                                             \
        TT_EXCEPTION(                                                                                             \
            std::format("Expected inputs to be the same dtype, given dtypes {:s}", concat_dtype_str(__VA_ARGS__)) \
        );                                                                                                        \
    }

// NOLINTNEXTLINE*(-macro-usage)
#define CHECK_SELF_FLOAT()                                                                                 \
    if (!is_float_dtype(scalar_type_)) {                                                                   \
        TT_EXCEPTION(std::format("Expected floating point self dtype. Self dtype is {:s}", scalar_type_)); \
    }

// NOLINTNEXTLINE*(-macro-usage)
#define CHECK_INPLACE_AUTOGRAD(t)                                                                         \
    if (autograd::GradMode::is_enabled() && (t).requires_grad()) {                                        \
        TT_EXCEPTION("Performing inplace operation on tensor which requires grad is not yet supported."); \
    }

// NOLINTNEXTLINE*(-macro-usage)
#define DISABLE_BOOL(t, OP)                                                          \
    if ((t).dtype() == kBool) {                                                      \
        TT_EXCEPTION(std::format("{:s} on boolean tensors is not supported.", #OP)); \
    }

// NOLINTNEXTLINE*(-macro-usage)
#define CHECK_REQUIRES_GRAD(dtype, requires_grad)                                                                  \
    if (requires_grad && !is_float_dtype(dtype)) {                                                                 \
        TT_EXCEPTION(                                                                                              \
            std::format("Only tensors of floating point type can have their gradient required, given {:s}", dtype) \
        );                                                                                                         \
    }

inline auto check_promote(const Tensor &input, ScalarType promote_type) -> Tensor {
    return (input.dtype() == promote_type) ? input : input.to(promote_type);
}

}    // namespace

// ------------------------------------------------
// Constructors
// ------------------------------------------------

template <typename T>
    requires(IsScalarType<T> || std::is_same_v<T, bool>)
Tensor::Tensor(const std::vector<T> &data, Shape shape, Device device, bool requires_grad)
    : device_(device),
      scalar_type_(to_scalar<T>::type),
      offset_(0),
      shape_(std::move(shape)),
      stride_(shape_.to_stride()),
      storage_(get_backend(device)->from_vec(data, device.id)),
      ctx_(std::make_shared<autograd::SharedGrad>()) {
    if (data.empty()) {
        TT_EXCEPTION("Cannot make empty tensor");
    }
    if (shape_.numel() != static_cast<int>(data.size())) {
        TT_EXCEPTION(
            std::format(
                "Number of elements the given shape represents ({:d}) does not match size of input "
                "data ({})",
                shape_.numel(),
                data.size()
            )
        );
    }
    CHECK_REQUIRES_GRAD(scalar_type_, requires_grad);
    set_requires_grad(requires_grad);
}

template Tensor::Tensor(const std::vector<bool> &, Shape shape, Device device, bool requires_grad);
template Tensor::Tensor(const std::vector<to_ctype_t<kU8>> &, Shape shape, Device device, bool requires_grad);
template Tensor::Tensor(const std::vector<to_ctype_t<kI16>> &, Shape shape, Device device, bool requires_grad);
template Tensor::Tensor(const std::vector<to_ctype_t<kI32>> &, Shape shape, Device device, bool requires_grad);
template Tensor::Tensor(const std::vector<to_ctype_t<kI64>> &, Shape shape, Device device, bool requires_grad);
template Tensor::Tensor(const std::vector<to_ctype_t<kF32>> &, Shape shape, Device device, bool requires_grad);
template Tensor::Tensor(const std::vector<to_ctype_t<kF64>> &, Shape shape, Device device, bool requires_grad);

template <typename T>
    requires(IsScalarType<T> || std::is_same_v<T, bool>)
Tensor::Tensor(std::vector<T> &&data, Shape shape, Device device, bool requires_grad)
    : device_(device),
      scalar_type_(to_scalar<T>::type),
      offset_(0),
      shape_({static_cast<int>(data.size())}),
      stride_(shape_.to_stride()),
      storage_(get_backend(device)->from_vec(std::move(data), device.id)),
      ctx_(std::make_shared<autograd::SharedGrad>()) {
    // shape_ is the moved from vector size, need this temporary since we cannot refer back to moved from data
    if (shape.size() <= 0) {
        TT_EXCEPTION("Cannot make empty tensor");
    }
    if (shape.numel() != shape_.numel()) {
        TT_EXCEPTION(
            std::format(
                "Number of elements the given shape represents ({:d}) does not match size of input "
                "data ({})",
                shape.numel(),
                shape_.numel()
            )
        );
    }
    // Set shape from passed value
    shape_ = shape;
    stride_ = shape_.to_stride();
    CHECK_REQUIRES_GRAD(scalar_type_, requires_grad);
    set_requires_grad(requires_grad);
}

template Tensor::Tensor(std::vector<bool> &&, Shape shape, Device device, bool requires_grad);
template Tensor::Tensor(std::vector<to_ctype_t<kU8>> &&, Shape shape, Device device, bool requires_grad);
template Tensor::Tensor(std::vector<to_ctype_t<kI16>> &&, Shape shape, Device device, bool requires_grad);
template Tensor::Tensor(std::vector<to_ctype_t<kI32>> &&, Shape shape, Device device, bool requires_grad);
template Tensor::Tensor(std::vector<to_ctype_t<kI64>> &&, Shape shape, Device device, bool requires_grad);
template Tensor::Tensor(std::vector<to_ctype_t<kF32>> &&, Shape shape, Device device, bool requires_grad);
template Tensor::Tensor(std::vector<to_ctype_t<kF64>> &&, Shape shape, Device device, bool requires_grad);

// Can't reuse other constructor since we need to get data size after moving it
template <typename T>
    requires(IsScalarType<T> || std::is_same_v<T, bool>)
Tensor::Tensor(std::vector<T> &&data, Device device, bool requires_grad)
    : device_(device),
      scalar_type_(to_scalar<T>::type),
      offset_(0),
      shape_({static_cast<int>(data.size())}),
      stride_(shape_.to_stride()),
      storage_(get_backend(device)->from_vec(std::move(data), device.id)),
      ctx_(std::make_shared<autograd::SharedGrad>()) {
    if (shape_.size() <= 0) {
        TT_EXCEPTION("Cannot make empty tensor");
    }
    CHECK_REQUIRES_GRAD(scalar_type_, requires_grad);
    set_requires_grad(requires_grad);
}

template Tensor::Tensor(std::vector<bool> &&, Device device, bool requires_grad);
template Tensor::Tensor(std::vector<to_ctype_t<kU8>> &&, Device device, bool requires_grad);
template Tensor::Tensor(std::vector<to_ctype_t<kI16>> &&, Device device, bool requires_grad);
template Tensor::Tensor(std::vector<to_ctype_t<kI32>> &&, Device device, bool requires_grad);
template Tensor::Tensor(std::vector<to_ctype_t<kI64>> &&, Device device, bool requires_grad);
template Tensor::Tensor(std::vector<to_ctype_t<kF32>> &&, Device device, bool requires_grad);
template Tensor::Tensor(std::vector<to_ctype_t<kF64>> &&, Device device, bool requires_grad);

Tensor::Tensor(Scalar scalar, Device device, bool requires_grad)
    : device_(device),
      scalar_type_(scalar.dtype()),
      offset_(0),
      shape_({1}),
      stride_(shape_.to_stride()),
      storage_(get_backend(device)->from_scalar(scalar, device.id)),
      ctx_(std::make_shared<autograd::SharedGrad>()) {
    CHECK_REQUIRES_GRAD(scalar_type_, requires_grad);
}

// Caller of this from internal autograd engine will set ctx
Tensor::Tensor(std::shared_ptr<StorageBase> _storage, ScalarType scalar_type, Shape shape, Device device)
    : device_(device),
      scalar_type_(scalar_type),
      offset_(0),
      shape_(std::move(shape)),
      stride_(shape_.to_stride()),
      storage_(std::move(_storage)),
      ctx_(std::make_shared<autograd::SharedGrad>()) {}

void Tensor::set_from(const Tensor &other) {
    device_ = other.device_;
    scalar_type_ = other.scalar_type_;
    offset_ = other.offset_;
    shape_ = other.shape_;
    stride_ = other.stride_;
    storage_ = other.storage_;
}

auto Tensor::serialize() const -> std::vector<char> {
    CHECK_VALID_TENSOR(*this);
    autograd::NoGradGuard guard;
    Tensor t = contiguous();

    nop::Serializer<nop::StreamWriter<std::stringstream>> serializer;
    serializer.Write(t.shape().to_vec());
    serializer.Write(static_cast<int>(t.dtype()));
    serializer.Write(t.requires_grad());
    DISPATCH_ALL_TYPES(scalar_type_, "tensor::serialize", [&]() { serializer.Write(t.to_vec<scalar_t>()); });

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
    return byte_data;
}

namespace {

auto deserialize(const std::vector<char> &serialized_data) -> Tensor {
    std::stringstream ss;
    ss.write(serialized_data.data(), std::streamsize(serialized_data.size()));
    nop::Deserializer<nop::StreamReader<std::stringstream>> deserializer{std::move(ss)};

    // Read from archive
    std::vector<int> _shape;
    int _dtype{};
    bool _requires_grad{};
    deserializer.Read(&_shape);
    deserializer.Read(&_dtype);
    deserializer.Read(&_requires_grad);
    Shape shape(_shape);
    auto dtype = static_cast<ScalarType>(_dtype);
    // Create temporary tensor and set from it
    return DISPATCH_ALL_TYPES(dtype, "tensor::deserialize", [&]() -> Tensor {
        std::vector<scalar_t> data;
        deserializer.Read(&data);
        return Tensor(data, shape, kCPU, false);
    });
}
}    // namespace

void Tensor::deserialize(const std::vector<char> &serialized_data) {
    CHECK_VALID_TENSOR(*this);
    autograd::NoGradGuard guard;

    std::stringstream ss;
    ss.write(serialized_data.data(), std::streamsize(serialized_data.size()));
    nop::Deserializer<nop::StreamReader<std::stringstream>> deserializer{std::move(ss)};

    // Read from archive
    std::vector<int> _shape;
    int _dtype{};
    bool _requires_grad{};
    deserializer.Read(&_shape);
    deserializer.Read(&_dtype);
    deserializer.Read(&_requires_grad);
    Shape shape(_shape);
    auto dtype = static_cast<ScalarType>(_dtype);
    // Create temporary tensor and set from it
    DISPATCH_ALL_TYPES(dtype, "tensor::deserialize", [&]() {
        std::vector<scalar_t> data;
        deserializer.Read(&data);
        Tensor t(data, shape, device(), _requires_grad);
        set_from(t);
    });
}

void save(const std::string &path, const Tensor &tensor) {
    std::ofstream tensor_file(path, std::ios::out | std::ios::binary);
    std::vector<char> data = tensor.serialize();
    tensor_file.write(&data[0], static_cast<std::streamsize>(data.size()));
    tensor_file.close();
}

Tensor load(const std::string &path) {
    std::ifstream tensor_file(path, std::ios::in | std::ios::binary);

    // Get file size
    std::streampos fileSize;
    tensor_file.seekg(0, std::ios::end);
    fileSize = tensor_file.tellg();
    tensor_file.seekg(0, std::ios::beg);

    // Read
    std::vector<char> data(static_cast<std::size_t>(fileSize));
    tensor_file.read(&data[0], fileSize);
    tensor_file.close();
    return deserialize(data);
}

// Getters
auto Tensor::has_storage() const -> bool {
    return storage_ != nullptr;
}

auto Tensor::dtype() const -> ScalarType {
    CHECK_VALID_TENSOR(*this);
    return scalar_type_;
}

auto Tensor::device() const -> Device {
    CHECK_VALID_TENSOR(*this);
    return device_;
}

auto Tensor::offset() const -> int {
    CHECK_VALID_TENSOR(*this);
    return offset_;
}

auto Tensor::shape() const -> const Shape & {
    CHECK_VALID_TENSOR(*this);
    return shape_;
}

auto Tensor::stride() const -> const Shape & {
    CHECK_VALID_TENSOR(*this);
    return stride_;
}

auto Tensor::numel() const -> int {
    CHECK_VALID_TENSOR(*this);
    return shape_.numel();
}

auto Tensor::dim() const -> int {
    CHECK_VALID_TENSOR(*this);
    return shape_.size();
}

auto Tensor::size(int dim) const -> int {
    CHECK_VALID_TENSOR(*this);
    CHECK_VALID_INDEX_EXCLUSIVE(dim, shape_.ndim());
    dim = indexing_exclusive(dim, shape_);
    return shape_[dim];
}
auto Tensor::is_contiguous() const -> bool {
    CHECK_VALID_TENSOR(*this);
    // Check if stride of tensor matches expected stride of the shape
    int accum = 1;
    for (int i = stride_.size() - 1; i >= 0; --i) {
        if (accum != stride_[i]) {
            return false;
        }
        accum *= shape_[i];
    }
    return true;
}

auto Tensor::item() const -> Scalar {
    CHECK_VALID_TENSOR(*this);
    return get_backend(device_)->item(*this);
}

auto Tensor::to(ScalarType dtype) const -> Tensor {
    CHECK_VALID_TENSOR(*this);
    if (dtype == scalar_type_) {
        return *this;
    }
    return autograd::TensorToScalarType::apply(*this, dtype);
}

auto Tensor::to(Device device) const -> Tensor {
    CHECK_VALID_TENSOR(*this);
    if (device == device_) {
        return *this;
    }
    return autograd::TensorToDevice::apply(*this, device);
}

template <typename T>
    requires(IsScalarType<T> || std::is_same_v<T, bool>)
auto Tensor::to_vec() const -> std::vector<T> {
    CHECK_VALID_TENSOR(*this);
    // cast if underlying type does not match output type
    Tensor res = scalar_type_ == to_scalar<T>::type ? clone() : to(to_scalar<T>::type);
    std::vector<T> data_out;
    // Specialization for bool
    if constexpr (std::is_same_v<T, bool>) {
        std::vector<uint8_t> _data_out;
        get_backend(device_)->to_vec(res, _data_out);
        data_out.reserve(_data_out.size());
        for (const auto &v : _data_out) {
            data_out.push_back(static_cast<bool>(v));
        }
    } else {
        get_backend(device_)->to_vec(res, data_out);
    }
    return data_out;
}
template auto Tensor::to_vec() const -> std::vector<bool>;
template auto Tensor::to_vec() const -> std::vector<to_ctype_t<kU8>>;
template auto Tensor::to_vec() const -> std::vector<to_ctype_t<kI16>>;
template auto Tensor::to_vec() const -> std::vector<to_ctype_t<kI32>>;
template auto Tensor::to_vec() const -> std::vector<to_ctype_t<kI64>>;
template auto Tensor::to_vec() const -> std::vector<to_ctype_t<kF32>>;
template auto Tensor::to_vec() const -> std::vector<to_ctype_t<kF64>>;

std::ostream &operator<<(std::ostream &os, const Tensor &tensor) {
    CHECK_VALID_TENSOR(tensor);
    return get_backend(tensor.device_)->print(os, tensor);
}

auto Tensor::contiguous() const -> Tensor {
    CHECK_VALID_TENSOR(*this);
    return is_contiguous() ? *this : clone();
}

auto Tensor::clone() const -> Tensor {
    CHECK_VALID_TENSOR(*this);
    return autograd::TensorClone::apply(*this);
}

auto Tensor::detach() const -> Tensor {
    CHECK_VALID_TENSOR(*this);
    return get_backend(this->device())->identity(*this);
}

auto Tensor::data_ptr() const -> uintptr_t {
    CHECK_VALID_TENSOR(*this);
    return get_backend(device_)->data_ptr(*this);
}

auto Tensor::version_count() const -> int {
    CHECK_VALID_TENSOR(*this);
    return version_count_;
}

// Assignment
auto Tensor::operator=(const Scalar &rhs) -> Tensor & {
    CHECK_VALID_TENSOR(*this);
    CHECK_INPLACE_AUTOGRAD(*this);
    Tensor _rhs(rhs.to(scalar_type_), device_);
    get_backend(device_)->assign(*this, _rhs.expand(shape_));
    return *this;
}

auto Tensor::fill_(const Scalar &rhs) -> Tensor & {
    CHECK_VALID_TENSOR(*this);
    CHECK_INPLACE_AUTOGRAD(*this);
    Tensor _rhs(rhs.to(scalar_type_), device_);
    get_backend(device_)->assign(*this, _rhs.expand(shape_));
    return *this;
}

auto Tensor::operator=(const Tensor &rhs) & -> Tensor & {
    CHECK_VALID_TENSOR(rhs);
    if (this != &rhs) {
        device_ = rhs.device_;
        scalar_type_ = rhs.scalar_type_;
        offset_ = rhs.offset_;
        shape_ = rhs.shape_;
        stride_ = rhs.stride_;
        storage_ = rhs.storage_;
        ctx_ = rhs.ctx_;
    }
    return *this;
}

auto Tensor::operator=(Tensor &&rhs) & -> Tensor & {
    CHECK_VALID_TENSOR(rhs);
    if (this != &rhs) {
        device_ = rhs.device_;
        scalar_type_ = rhs.scalar_type_;
        offset_ = rhs.offset_;
        shape_ = rhs.shape_;
        stride_ = rhs.stride_;
        storage_ = rhs.storage_;
        ctx_ = rhs.ctx_;
    }
    return *this;
}

// Assignment on rvalues disabled in grad contexts, so we can ignore fixing ctx
auto Tensor::operator=(const Tensor &rhs) && -> Tensor & {
    CHECK_VALID_TENSOR(*this);
    CHECK_VALID_TENSOR(rhs);
    CHECK_SAME_DEVICE(*this, rhs);
    CHECK_SAME_SHAPE(*this, rhs);
    CHECK_INPLACE_AUTOGRAD(*this);
    get_backend(device_)->assign(*this, rhs.to(scalar_type_));
    return *this;
}
auto Tensor::operator=(Tensor &&rhs) && -> Tensor & {
    CHECK_VALID_TENSOR(*this);
    CHECK_VALID_TENSOR(rhs);
    CHECK_SAME_DEVICE(*this, rhs);
    CHECK_SAME_SHAPE(*this, rhs);
    CHECK_INPLACE_AUTOGRAD(*this);
    get_backend(device_)->assign(*this, rhs.to(scalar_type_));
    return *this;
}

// Element-wise inplace arithmetic
auto Tensor::operator+=(const Scalar &rhs) -> Tensor & {
    CHECK_VALID_TENSOR(*this);
    return this->operator+=(Tensor(rhs, device_).expand(shape_));
}
auto Tensor::operator+=(const Tensor &rhs) -> Tensor & {
    CHECK_VALID_TENSOR(*this);
    CHECK_VALID_TENSOR(rhs);
    CHECK_SAME_DEVICE(*this, rhs);
    CHECK_SAME_SHAPE(*this, rhs);
    CHECK_INPLACE_AUTOGRAD(*this);
    DISABLE_BOOL(*this, operator+=);
    get_backend(device_)->add_inplace(*this, rhs.to(scalar_type_));
    return *this;
}
auto Tensor::add_(const Scalar &rhs) -> Tensor & {
    return this->operator+=(rhs);
}
auto Tensor::add_(const Tensor &rhs) -> Tensor & {
    CHECK_VALID_TENSOR(*this);
    return this->operator+=(rhs);
}

auto Tensor::operator-=(const Scalar &rhs) -> Tensor & {
    CHECK_VALID_TENSOR(*this);
    return this->operator-=(Tensor(rhs, device_).expand(shape_));
}
auto Tensor::operator-=(const Tensor &rhs) -> Tensor & {
    CHECK_VALID_TENSOR(*this);
    CHECK_VALID_TENSOR(rhs);
    CHECK_SAME_DEVICE(*this, rhs);
    CHECK_SAME_SHAPE(*this, rhs);
    CHECK_INPLACE_AUTOGRAD(*this);
    DISABLE_BOOL(*this, operator-=);
    get_backend(device_)->sub_inplace(*this, rhs.to(scalar_type_));
    return *this;
}
auto Tensor::sub_(const Scalar &rhs) -> Tensor & {
    return this->operator-=(rhs);
}
auto Tensor::sub_(const Tensor &rhs) -> Tensor & {
    CHECK_VALID_TENSOR(*this);
    return this->operator-=(rhs);
}

auto Tensor::operator*=(const Scalar &rhs) -> Tensor & {
    CHECK_VALID_TENSOR(*this);
    return this->operator*=(Tensor(rhs, device_).expand(shape_));
}
auto Tensor::operator*=(const Tensor &rhs) -> Tensor & {
    CHECK_VALID_TENSOR(*this);
    CHECK_VALID_TENSOR(rhs);
    CHECK_SAME_DEVICE(*this, rhs);
    CHECK_SAME_SHAPE(*this, rhs);
    CHECK_INPLACE_AUTOGRAD(*this);
    DISABLE_BOOL(*this, operator*=);
    get_backend(device_)->mul_inplace(*this, rhs.to(scalar_type_));
    return *this;
}
auto Tensor::mul_(const Scalar &rhs) -> Tensor & {
    return this->operator*=(rhs);
}
auto Tensor::mul_(const Tensor &rhs) -> Tensor & {
    CHECK_VALID_TENSOR(*this);
    return this->operator*=(rhs);
}

auto Tensor::operator/=(const Scalar &rhs) -> Tensor & {
    CHECK_VALID_TENSOR(*this);
    return this->operator/=(Tensor(rhs, device_).expand(shape_));
}
auto Tensor::operator/=(const Tensor &rhs) -> Tensor & {
    CHECK_VALID_TENSOR(*this);
    CHECK_VALID_TENSOR(rhs);
    CHECK_SAME_DEVICE(*this, rhs);
    CHECK_SAME_SHAPE(*this, rhs);
    CHECK_INPLACE_AUTOGRAD(*this);
    DISABLE_BOOL(*this, operator/=);
    get_backend(device_)->div_inplace(*this, rhs.to(scalar_type_));
    return *this;
}
auto Tensor::div_(const Scalar &rhs) -> Tensor & {
    return this->operator/=(rhs);
}
auto Tensor::div_(const Tensor &rhs) -> Tensor & {
    CHECK_VALID_TENSOR(*this);
    return this->operator/=(rhs);
}

// Tensor indexing
auto Tensor::operator[](const Tensor &mask) const -> Tensor {
    CHECK_VALID_TENSOR(*this);
    CHECK_VALID_TENSOR(mask);
    if (mask.scalar_type_ != kBool) {
        TT_EXCEPTION(
            std::format("Indexing operator with Tensor only supports kBool scalar types. Given {:s}", mask.dtype())
        );
    }
    return index(*this, mask);
}

auto Tensor::operator[](const indexing::Index &index) const -> Tensor {
    CHECK_VALID_TENSOR(*this);
    return (*this)[std::vector<indexing::Index>{index}];
}

auto Tensor::operator[](const std::vector<indexing::Index> &indices) const -> Tensor {
    CHECK_VALID_TENSOR(*this);
    if (indices.empty()) {
        TT_EXCEPTION("Indexing with an empty list");
    }
    if (static_cast<int>(indices.size()) > shape_.size()) {
        TT_EXCEPTION(
            std::format(
                "Index list {:s} of size {:d} is too large for tensor with {:d} dimensions",
                indexing::index_list_to_string(indices),
                indices.size(),
                shape_.ndim()
            )
        );
    }
    // Check valid indexing
    for (int i = 0; i < static_cast<int>(indices.size()); ++i) {
        std::visit(
            overloaded{
                // int index will move offset forward, remove shape/stride at that index
                [&](int index) {
                    if (index < 0 || index >= shape_[i]) {
                        TT_EXCEPTION(
                            std::format(
                                "Indexing with {:s} at dim {:d} is out of bounds for shape {:s}",
                                indexing::index_list_to_string(indices),
                                i,
                                shape_.to_string()
                            )
                        );
                    }
                },
                [&](const indexing::Slice &slice) {
                    if (slice.start() < 0 || slice.start() >= shape_[i]) {
                        TT_EXCEPTION(
                            std::format(
                                "Indexing with {:s} start at dim {:d} is out of range for shape {:s}.",
                                indexing::index_list_to_string(indices),
                                i,
                                shape_.to_string()
                            )
                        );
                    }
                    // Slice end could be INT_MAX if no end given,
                    // Clamp to dim if no end provided, otherwise use provided value and check for
                    // bounds
                    if (slice.end(shape_[i]) < 0 || slice.end(shape_[i]) > shape_[i]) {
                        TT_EXCEPTION(
                            std::format(
                                "Indexing with {:s} end at dim {:d} is out of range for shape {:s}.",
                                indexing::index_list_to_string(indices),
                                i,
                                shape_.to_string()
                            )
                        );
                    }
                    if (slice.start() >= slice.end()) {
                        TT_EXCEPTION(
                            std::format(
                                "Indexing with {:s} at dim {:d} has start >= end.",
                                indexing::index_list_to_string(indices),
                                i,
                                shape_.to_string()
                            )
                        );
                    }
                },
            },
            indices[static_cast<std::size_t>(i)].get_index()
        );
    }
    return autograd::TensorIndex::apply(*this, indices);
}

// index runners
namespace {
auto index_mask(const Tensor &self, const Tensor &mask) -> Tensor {
    CHECK_SAME_SHAPE(self, mask);
    // Find number of elements mask will select
    int mask_count = mask.sum().item<int>();
    if (mask_count == 0) {
        TT_EXCEPTION("Indexing Tensor with false mask will result in empty Tensor");
    }
    return autograd::TensorIndexMask::apply(self, mask, mask_count);
}

auto index_indices(const Tensor &self, const Tensor &indices) -> Tensor {
    const auto max = indices.max().item<kDefaultInt>();
    const auto min = indices.min().item<kDefaultInt>();
    if (min < 0 || max >= self.numel()) {
        TT_EXCEPTION(
            std::format("Given indices contains values outside range, expected to be within [0, {:d})", self.numel())
        );
    }
    return autograd::TensorIndexIndices::apply(self, indices);
}
}    // namespace

auto index(const Tensor &tensor, const Tensor &rhs) -> Tensor {
    CHECK_VALID_TENSOR(tensor);
    CHECK_VALID_TENSOR(rhs);
    CHECK_SAME_DEVICE(tensor, rhs);
    if (rhs.dtype() == kBool) {
        return index_mask(tensor, rhs);
    } else if (is_integral_dtype(rhs.dtype())) {
        return index_indices(tensor, rhs.to(kDefaultInt));
    } else {
        TT_EXCEPTION(
            std::format(
                "Invalid rhs dtype ({:s}) to index. Must be kBool for masking or kDefaultInt for indexing.",
                rhs.dtype()
            )
        );
    }
}

auto index_put(const Tensor &self, const Tensor &rhs, const Scalar &value) -> Tensor {
    CHECK_VALID_TENSOR(self);
    CHECK_VALID_TENSOR(rhs);
    CHECK_SAME_DEVICE(self, rhs);

    Tensor values(value.to(self.dtype()), self.device());
    if (rhs.dtype() == kBool) {
        int mask_count = rhs.sum().item<int>();
        if (mask_count == 0) {
            TT_EXCEPTION("Indexing Tensor with false mask will result in empty Tensor");
        }
        values = values.expand({mask_count});
        return autograd::TensorIndexPutMask::apply(self, rhs, values);
    } else if (is_integral_dtype(rhs.dtype())) {
        values = values.expand(rhs.shape());
        return autograd::TensorIndexPutIndices::apply(self, rhs.to(kDefaultInt), values);
    } else {
        TT_EXCEPTION(
            std::format(
                "Invalid rhs dtype ({:s}) to index. Must be kBool for masking or kDefaultInt for indexing.",
                rhs.dtype()
            )
        );
    }
}

auto index_put(const Tensor &self, const Tensor &rhs, const Tensor &values) -> Tensor {
    CHECK_VALID_TENSOR(self);
    CHECK_VALID_TENSOR(rhs);
    CHECK_VALID_TENSOR(values);
    CHECK_SAME_DEVICE(self, rhs, values);

    if (rhs.dtype() == kBool) {
        int mask_count = rhs.sum().item<int>();
        if (mask_count == 0) {
            TT_EXCEPTION("Indexing Tensor with false mask will result in empty Tensor");
        }
        return autograd::TensorIndexPutMask::apply(self, rhs, values);
    } else if (is_integral_dtype(rhs.dtype())) {
        return autograd::TensorIndexPutIndices::apply(self, rhs.to(kDefaultInt), values);
    } else {
        TT_EXCEPTION(
            std::format(
                "Invalid rhs dtype ({:s}) to index. Must be kBool for masking or kDefaultInt for indexing.",
                rhs.dtype()
            )
        );
    }
}

auto index_select(const Tensor &tensor, const std::vector<int> &indices, int dim) -> Tensor {
    CHECK_VALID_TENSOR(tensor);
    if (indices.empty()) {
        TT_EXCEPTION("Cannot index with empty indices list.");
    }

    int N = tensor.dim();
    CHECK_VALID_INDEX_EXCLUSIVE(dim, N);
    dim = indexing_exclusive(dim, tensor.shape());

    // Check that all indices are within range
    for (const auto &idx : indices) {
        if (idx < 0 || idx > tensor.size(dim)) {
            TT_EXCEPTION(
                std::format(
                    "Given indices contains values outside range, expected to be within [0, {:d})",
                    tensor.size(dim)
                )
            );
        }
    }

    return autograd::TensorIndexSelect::apply(tensor, indices, dim);
}

auto index_select(const Tensor &tensor, const Tensor &indices, int dim) -> Tensor {
    CHECK_VALID_TENSOR(tensor);
    CHECK_VALID_TENSOR(indices);
    CHECK_SAME_DEVICE(tensor, indices);
    if (!is_integral_dtype(indices.dtype())) {
        TT_EXCEPTION(std::format("Indices must be an integral type, given dtype = {:s}", indices.dtype()));
    }
    return index_select(tensor, indices.to_vec<int>(), dim);
}

auto repeat_interleave(const Tensor &tensor, int repeats, int dim) -> Tensor {
    CHECK_VALID_TENSOR(tensor);
    int N = tensor.dim();
    CHECK_VALID_INDEX_EXCLUSIVE(dim, N);
    dim = indexing_exclusive(dim, tensor.shape());
    if (repeats < 1) {
        TT_EXCEPTION(std::format("Expected repeats > 0, given {:d}", repeats));
    }
    std::vector<int> indices;
    indices.reserve(static_cast<std::size_t>(repeats * tensor.size(dim)));
    for (int i = 0; i < tensor.size(dim); ++i) {
        for (int j = 0; j < repeats; ++j) {
            indices.push_back(i);
        }
    }
    return index_select(tensor, indices, dim);
}

auto repeat(const Tensor &tensor, const Tensor &repeats) -> Tensor {
    CHECK_VALID_TENSOR(tensor);
    CHECK_VALID_TENSOR(repeats);
    CHECK_SAME_DEVICE(tensor, repeats);
    if (!is_integral_dtype(repeats.dtype())) {
        TT_EXCEPTION(std::format("Indices must be an integral type, given dtype = {:s}", repeats.dtype()));
    }
    const std::vector<int> _repeats = repeats.to_vec<int>();
    return repeat(tensor, _repeats);
}

auto repeat(const Tensor &tensor, const std::vector<int> &repeats) -> Tensor {
    CHECK_VALID_TENSOR(tensor);
    if (tensor.dim() != static_cast<int>(repeats.size())) {
        TT_EXCEPTION(
            std::format(
                "Dimension of given repeats ({:d}) does not match the number of dimensions of the given tensor ({:d})",
                repeats.size(),
                tensor.dim()
            )
        );
    }

    // Check for non-positive repeats
    for (int i = 0; i < tensor.dim(); ++i) {
        int num_repeats = repeats[static_cast<std::size_t>(i)];
        if (repeats[static_cast<std::size_t>(i)] <= 0) {
            TT_EXCEPTION(
                std::format("Given a non-positive number of repeats ({:d}) at dimension ({:d})", num_repeats, i)
            );
        }
    }

    return autograd::TensorRepeat::apply(tensor, repeats);
}

auto gather(const Tensor &tensor, const Tensor &_indices, int dim) -> Tensor {
    CHECK_VALID_TENSOR(tensor);
    CHECK_VALID_TENSOR(_indices);
    CHECK_SAME_DEVICE(tensor, _indices);
    if (tensor.dim() != _indices.dim()) {
        TT_EXCEPTION(
            std::format(
                "The number of dimensions in the input ({:d}) does not match the number of dimensions in the given "
                "indices "
                "({:d})",
                tensor.dim(),
                _indices.dim()
            )
        );
    }
    if (!is_integral_dtype(_indices.dtype())) {
        TT_EXCEPTION(std::format("indices scalar type should be integral, given {:s}", _indices.dtype()));
    }
    auto indices = check_promote(_indices, kDefaultInt);

    int N = tensor.dim();
    CHECK_VALID_INDEX_EXCLUSIVE(dim, N);
    dim = indexing_exclusive(dim, tensor.shape());

    // Indices shape = result shape except for 1 along selected dim
    auto result_shape = tensor.shape();
    result_shape[dim] = 1;
    if (result_shape != indices.shape()) {
        TT_EXCEPTION(
            std::format(
                "Expected indices shape {:s} to match {:s} for input tensor shape {:s} and dim={:d}",
                indices.shape(),
                result_shape,
                tensor.shape(),
                dim
            )
        );
    }

    const auto max = indices.max().item<kDefaultInt>();
    const auto min = indices.min().item<kDefaultInt>();
    if (min < 0 || max >= tensor.size(dim)) {
        TT_EXCEPTION(
            std::format(
                "Given indices contains values outside range, expected to be within [0, {:d})",
                tensor.size(dim)
            )
        );
    }
    return autograd::TensorGather::apply(tensor, indices, dim);
}

// ------------------------------------------------
// Range based loop iterator support
// ------------------------------------------------

auto Tensor::Iterator::operator!=(const Tensor::Iterator &other) const -> bool {
    return idx_ != other.idx_;
}

auto Tensor::Iterator::operator++() -> Tensor::Iterator {
    if (idx_ > tensor_.size(0)) {
        TT_ERROR(
            "Internal iterator is already passed the end. Tensor::Iterator should only be used implicitly "
            "in range-based for loops"
        );
    }
    ++idx_;
    return *this;
}

auto Tensor::Iterator::operator*() const -> Tensor {
    if (idx_ >= tensor_.size(0)) {
        TT_ERROR(
            "Internal iterator is already passed the end. Tensor::Iterator should only be used implicitly "
            "in range-based for loops"
        );
    }
    return tensor_[idx_];
}

auto Tensor::begin() -> Tensor::Iterator {
    return {*this, 0};
}

auto Tensor::end() -> Tensor::Iterator {
    return {*this, size(0)};
}

// ------------------------------------------------
// Autograd related operations
// ------------------------------------------------

void Tensor::clear_grad() {
    CHECK_VALID_TENSOR(*this);
    ctx_->grad.reset();
}

auto Tensor::grad() const -> const std::optional<Tensor> & {
    CHECK_VALID_TENSOR(*this);
    return ctx_->grad;
}

void Tensor::add_grad(const Tensor &grad) {
    CHECK_VALID_TENSOR(*this);
    CHECK_VALID_TENSOR(grad);
    if (requires_grad()) {
        if (grad.device() != device_) {
            TT_ERROR(
                std::format("Grad device ({:s}) does not match the tensor device ({:s}).", grad.device(), device_)
            );
        }
        if (grad.dtype() != scalar_type_) {
            TT_ERROR(
                std::format("Grad dtype ({:s}) does not match the tensor dtype ({:s}).", grad.dtype(), scalar_type_)
            );
        }
        if (grad.shape() != shape_) {
            TT_ERROR(std::format("Grad shape ({:s}) does not match the tensor shape ({:s}).", grad.shape(), shape_));
        }
        if (ctx_->grad) {
            *ctx_->grad = *(ctx_->grad) + grad;
        } else {
            ctx_->grad = grad;
        }
    }
}

auto Tensor::requires_grad() const -> bool {
    CHECK_VALID_TENSOR(*this);
    return ctx_->requires_grad;
}

void Tensor::set_requires_grad(bool set_grad) {
    CHECK_VALID_TENSOR(*this);
    if (set_grad && !is_float_dtype(scalar_type_)) {
        TT_EXCEPTION(
            std::format("Only floating point Tensors support gradient tracking. Dtype of Tensor is {:s}", scalar_type_)
        );
    }
    ctx_->requires_grad = set_grad;
}

void Tensor::register_hook(const autograd::GradHook &hook) {
    CHECK_VALID_TENSOR(*this);
    ctx_->grad_hook = hook;
}

void Tensor::apply_grad_hook() {
    CHECK_VALID_TENSOR(*this);
    if (ctx_->grad_hook) {
        if (!ctx_->grad) {
            TT_EXCEPTION("Grad hook called on tensor without gradient set.");
        }
        ctx_->grad_hook(*ctx_->grad);
    }
}

void Tensor::backward(const std::optional<Tensor> &grad, bool retain_graph) {
    CHECK_VALID_TENSOR(*this);
    if (!is_float_dtype(scalar_type_)) {
        TT_EXCEPTION(std::format("Can only call backward on floating point tensors. Given dtype %s", scalar_type_));
    }
    if (!ctx_->requires_grad) {
        TT_EXCEPTION("Calling backward on tensor which does not require a gradient");
    }
    if (!ctx_->grad_func) {
        TT_EXCEPTION("Calling backward on tensor from non-differentiable operation");
    }
    if (grad.has_value()) {
        CHECK_SAME_TYPE(*this, *grad);
        CHECK_SAME_SHAPE(*this, *grad);
        CHECK_SAME_DEVICE(*this, *grad);
    }
    autograd::backward(*this, grad.value_or(ones_like(*this)), retain_graph);
}

auto Tensor::is_leaf() const -> bool {
    CHECK_VALID_TENSOR(*this);
    return !ctx_->requires_grad || !ctx_->grad_func;
}

// ------------------------------------------------
// Shape Modification operations
// ------------------------------------------------

// If number of dimensions not equal, prepend 1s until same length
// For each resulting dim, set max of the sizes
auto broadcast_result_shape(const Shape lhs, const Shape rhs) -> Shape {
    if (lhs.numel() == 0 || rhs.numel() == 0) {
        TT_EXCEPTION(
            std::format("Cannot broadcast empty shapes, given lhs={:s}, rhs={:s}", lhs.to_string(), rhs.to_string())
        );
    }
    if (!are_broadcastable(lhs, rhs)) {
        TT_EXCEPTION(std::format("Shape {:s} and shape {:s} are not broadcastable.", lhs.to_string(), rhs.to_string()));
    }
    const int result_dim = std::max(lhs.ndim(), rhs.ndim());
    Shape result_shape(std::vector<int>(static_cast<std::size_t>(result_dim), 1));
    for (int i = 0; i < result_dim; ++i) {
        const int dim_lhs = (i < lhs.ndim()) ? lhs[lhs.ndim() - 1 - i] : 1;
        const int dim_rhs = (i < rhs.ndim()) ? rhs[rhs.ndim() - 1 - i] : 1;
        result_shape[result_dim - i - 1] = std::max(dim_lhs, dim_rhs);
    }
    return result_shape;
}

auto can_broadcast_to(const Shape &shape, const Shape &target_shape) -> bool {
    if (shape.ndim() == 0 || target_shape.ndim() == 0 || shape.numel() == 0 || target_shape.numel() == 0) {
        return false;
    }
    const int result_dim = std::max(shape.ndim(), target_shape.ndim());
    for (int i = 0; i < result_dim; ++i) {
        const int dim_lhs = (i < shape.ndim()) ? shape[shape.ndim() - 1 - i] : 1;
        const int dim_rhs = (i < target_shape.ndim()) ? target_shape[target_shape.ndim() - 1 - i] : 1;
        if (dim_lhs > 1 && dim_lhs != dim_rhs) {
            return false;
        }
    }
    return true;
}

// https://pytorch.org/docs/stable/notes/broadcasting.html
// Broadcastable IF
// 1. Each shape has at least 1 dimension
// 2. From trailing dims, must be: equal, one of them is 1, or doesn't exist
auto are_broadcastable(const Shape &lhs, const Shape &rhs) -> bool {
    if (lhs.numel() == 0 || rhs.numel() == 0) {
        TT_EXCEPTION(
            std::format("Cannot broadcast empty shapes, given lhs={:s}, rhs={:s}", lhs.to_string(), rhs.to_string())
        );
    }
    if (lhs.ndim() < 1 || rhs.ndim() < 1) {
        return false;
    }
    for (int i = 0; i < std::min(lhs.ndim(), rhs.ndim()); ++i) {
        const int idx_lhs = lhs.ndim() - i - 1;
        const int idx_rhs = rhs.ndim() - i - 1;
        if (lhs[idx_lhs] != rhs[idx_rhs] && lhs[idx_lhs] != 1 && rhs[idx_rhs] != 1) {
            return false;
        }
    }
    return true;
}

// Broadcast
auto Tensor::broadcast_to(const Shape &shape) const -> Tensor {
    CHECK_VALID_TENSOR(*this);
    if (shape.ndim() == 0 || shape.numel() == 0) {
        TT_EXCEPTION("Cannot broadcast to empty shape");
    }
    if (shape_.ndim() > shape.ndim()) {
        TT_EXCEPTION(
            std::format(
                "Cannot broadcast shape {:s} to shape {:s}."
                "\n\tThe number of dimensions in the target shape ({:d}) must be greater "
                "than or equal to the number of dimensions in the current tensor ({:d}).",
                shape_.to_string(),
                shape.to_string(),
                shape.size(),
                shape_.size()
            )
        );
    }
    // Check if given broadcast to matches rules
    for (int i = 0; i < std::max(shape_.ndim(), shape.ndim()); ++i) {
        const int dim_lhs = (i < shape_.ndim()) ? shape_[shape_.ndim() - 1 - i] : 1;
        const int dim_rhs = (i < shape.ndim()) ? shape[shape.ndim() - 1 - i] : 1;
        if (dim_lhs > 1 && dim_lhs != dim_rhs) {
            TT_EXCEPTION(
                std::format(
                    "Cannot broadcast shape {:s} to shape {:s}."
                    "\n\tAt non-singleton index {:d}, the expanded-to size ({:d}) must match the "
                    "current size ({:d}).",
                    shape_.to_string(),
                    shape.to_string(),
                    shape.ndim() - 1 - i,
                    dim_rhs,
                    dim_lhs
                )
            );
        }
    }
    return autograd::TensorBroadcast::apply(*this, shape);
}
auto broadcast_to(const Tensor &tensor, const Shape &shape) -> Tensor {
    CHECK_VALID_TENSOR(tensor);
    return tensor.broadcast_to(shape);
}

// Expand
auto Tensor::expand(const Shape &shape) const -> Tensor {
    CHECK_VALID_TENSOR(*this);
    return broadcast_to(shape);
}
auto expand(const Tensor &tensor, const Shape &shape) -> Tensor {
    CHECK_VALID_TENSOR(tensor);
    return tensor.expand(shape);
}

// Squeeze
auto Tensor::squeeze(int dim) const -> Tensor {
    CHECK_VALID_TENSOR(*this);
    const auto N = shape_.ndim();
    CHECK_VALID_INDEX_EXCLUSIVE(dim, N);
    dim = indexing_exclusive(dim, shape_);

    // Return self shape if already single dim or squeezed dim is not 1
    auto result_shape = shape_;
    if (!(N == 1 || shape_[dim] != 1)) {
        result_shape.pop(dim);
    }
    return this->reshape(result_shape);
}
auto squeeze(const Tensor &tensor, int dim) -> Tensor {
    CHECK_VALID_TENSOR(tensor);
    return tensor.squeeze(dim);
}

// Unsqueeze
auto Tensor::unsqueeze(int dim) const -> Tensor {
    CHECK_VALID_TENSOR(*this);
    CHECK_VALID_INDEX_INCLUSIVE(dim, shape_.ndim());
    dim = indexing_inclusive(dim, shape_);

    // Insert new dim of size 1
    std::vector<int> new_shape = shape_.to_vec();
    new_shape.insert(new_shape.begin() + dim, 1);
    return this->reshape(Shape{new_shape});
}
auto unsqueeze(const Tensor &tensor, int dim) -> Tensor {
    CHECK_VALID_TENSOR(tensor);
    return tensor.unsqueeze(dim);
}

// Reshape
// @TODO: View can be returned on some non-contiguous tensors, but this is a bit complicated
auto Tensor::reshape(const Shape &shape) const -> Tensor {
    CHECK_VALID_TENSOR(*this);
    if (shape.numel() != shape_.numel()) {
        TT_EXCEPTION(
            std::format(
                "Tensor shape {:s} with {:d} elements cannot be reshaped to shape {:s} with {:d} elements, total "
                "size must match.",
                shape_,
                shape_.numel(),
                shape,
                shape.numel()
            )
        );
    }
    // view returned if contiguous
    return autograd::TensorReshape::apply(is_contiguous() ? *this : clone(), shape);
}
auto reshape(const Tensor &tensor, const Shape &shape) -> Tensor {
    CHECK_VALID_TENSOR(tensor);
    return tensor.reshape(shape);
}

// Flatten
auto Tensor::flatten(int start_dim, int end_dim) const -> Tensor {
    CHECK_VALID_TENSOR(*this);
    CHECK_VALID_INDEX_EXCLUSIVE(start_dim, shape_.size());
    CHECK_VALID_INDEX_EXCLUSIVE(end_dim, shape_.size());
    start_dim = indexing_exclusive(start_dim, shape_);
    end_dim = indexing_exclusive(end_dim, shape_);
    if (end_dim < start_dim) {
        TT_EXCEPTION(
            std::format("Expected start_dim {:d} to be less than or equal to end_dim {:d}", start_dim, end_dim)
        );
    }
    std::vector<int> new_shape;
    int flattened_shape = 1;
    for (int i : std::views::iota(0, shape_.size())) {
        if (i >= start_dim && i <= end_dim) {
            flattened_shape *= shape_[i];
            if (i == end_dim) {
                new_shape.push_back(flattened_shape);
            }
        } else {
            new_shape.push_back(shape_[i]);
        }
    }

    // flatten is just a special reshape
    return reshape(Shape(std::move(new_shape)));
}
auto flatten(const Tensor &tensor, int start_dim, int end_dim) -> Tensor {
    CHECK_VALID_TENSOR(tensor);
    return tensor.flatten(start_dim, end_dim);
}

auto Tensor::permute(const std::vector<int> &dims) const -> Tensor {
    CHECK_VALID_TENSOR(*this);
    // Check that dims contains permutation of 0..size()-1
    std::vector<int> sorted_dims(static_cast<std::size_t>(dim()));
    std::iota(sorted_dims.begin(), sorted_dims.end(), 0);
    if (dims.size() != sorted_dims.size() || !std::is_permutation(dims.begin(), dims.end(), sorted_dims.begin())) {
        TT_EXCEPTION(
            std::format("Input dims {:s} is not a permutation of {:s}", vec_to_str(dims), vec_to_str(sorted_dims))
        );
    }
    // Result is a shallow copy, just with a new view of the same underlying data
    return autograd::TensorPermute::apply(*this, dims);
}

auto permute(const Tensor &tensor, const std::vector<int> &dims) -> Tensor {
    CHECK_VALID_TENSOR(tensor);
    return tensor.permute(dims);
}

// ------------------------------------------------
// Tensor Creation
// ------------------------------------------------
auto full(Scalar value, Shape shape, Device device, bool requires_grad) -> Tensor {
    CHECK_EMPTY_SHAPE(shape);
    CHECK_REQUIRES_GRAD(value.dtype(), requires_grad);
    const auto N = static_cast<std::size_t>(shape.numel());
    Tensor result{get_backend(device)->full(value, N, device.id), value.dtype(), shape, device};
    result.set_requires_grad(requires_grad);
    return result;
}

auto zeros(Shape shape, ScalarType dtype, Device device, bool requires_grad) -> Tensor {
    CHECK_EMPTY_SHAPE(shape);
    CHECK_REQUIRES_GRAD(dtype, requires_grad);
    return full(Scalar(0, dtype), shape, device, requires_grad);
}
auto zeros(Shape shape, const TensorOptions &options) -> Tensor {
    return zeros(shape, options.dtype(), options.device(), options.requires_grad());
}
auto zeros_like(const Tensor &tensor, bool requires_grad) -> Tensor {
    return zeros(tensor.shape(), tensor.dtype(), tensor.device(), requires_grad);
}

auto ones(Shape shape, ScalarType dtype, Device device, bool requires_grad) -> Tensor {
    CHECK_EMPTY_SHAPE(shape);
    CHECK_REQUIRES_GRAD(dtype, requires_grad);
    return full(Scalar(1, dtype), shape, device);
}
auto ones(Shape shape, const TensorOptions &options) -> Tensor {
    return ones(shape, options.dtype(), options.device(), options.requires_grad());
}
auto ones_like(const Tensor &tensor, bool requires_grad) -> Tensor {
    return ones(tensor.shape(), tensor.dtype(), tensor.device(), requires_grad);
}

auto arange(Shape shape, ScalarType dtype, Device device, bool requires_grad) -> Tensor {
    CHECK_EMPTY_SHAPE(shape);
    if (dtype == kBool) {
        TT_EXCEPTION("arange not defined for boolean scalar type.");
    }
    CHECK_REQUIRES_GRAD(dtype, requires_grad);
    const auto N = static_cast<std::size_t>(shape.numel());
    Tensor result{get_backend(device)->arange(N, dtype, device.id), dtype, shape, device};
    result.set_requires_grad(requires_grad);
    return result;
}
auto arange(Shape shape, const TensorOptions &options) -> Tensor {
    return arange(shape, options.dtype(), options.device(), options.requires_grad());
}

auto linspace(
    double start,
    double stop,
    bool endpoint,
    Shape shape,
    ScalarType dtype,
    Device device,
    bool requires_grad
) -> Tensor {
    CHECK_EMPTY_SHAPE(shape);
    if (dtype == kBool) {
        TT_EXCEPTION("linspace not defined for boolean scalar type.");
    }
    CHECK_REQUIRES_GRAD(dtype, requires_grad);
    const auto N = static_cast<std::size_t>(shape.numel());
    ScalarType inter_type = is_float_dtype(dtype) ? dtype : kDefaultFloat;
    Tensor result{get_backend(device)->arange(N, inter_type, device.id), inter_type, shape, device};

    // Rescale
    auto num = static_cast<double>(shape.numel());
    num = endpoint ? num - 1 : num;
    double step = (stop - start) / num;
    result = result * step + start;

    result = result.to(dtype);
    result.set_requires_grad(requires_grad);
    return result;
}
auto linspace(double start, double stop, bool endpoint, Shape shape, const TensorOptions &options) -> Tensor {
    return linspace(start, stop, endpoint, shape, options.dtype(), options.device(), options.requires_grad());
}

auto eye(int rows, int cols, ScalarType dtype, Device device, bool requires_grad) -> Tensor {
    if (rows <= 0 || cols <= 0) {
        TT_EXCEPTION(
            std::format("Expected rows and columns to be greater than 0, given rows={:d}, cols={:d}", rows, cols)
        );
    }
    if (dtype == kBool) {
        TT_EXCEPTION("eye not defined for boolean scalar type.");
    }
    CHECK_REQUIRES_GRAD(dtype, requires_grad);
    autograd::NoGradGuard guard;
    std::vector<int> indices;
    for (int i = 0; i < std::min(rows, cols); ++i) {
        indices.push_back(i * cols + i);
    }
    Tensor base = zeros({rows, cols}, dtype, device, requires_grad);
    return index_put(base, Tensor(indices, device), 1);
}
auto eye(int rows, int cols, const TensorOptions &options) -> Tensor {
    return eye(rows, cols, options.dtype(), options.device(), options.requires_grad());
}

auto one_hot(Tensor indices, int num_classes) -> Tensor {
    CHECK_VALID_TENSOR(indices);
    if (num_classes <= 0 && num_classes != -1) {
        TT_EXCEPTION(
            std::format(
                "num_classes must be positive or -1 to compute from the passed indices, given num_classes={:d}",
                num_classes
            )
        );
    }
    if (!is_integral_dtype(indices.dtype())) {
        TT_EXCEPTION(std::format("Expected indices to be of integral dtype, given {:s}", indices.dtype()));
    }
    autograd::NoGradGuard guard;
    int indices_min = indices.min().item<int>();
    int indices_max = indices.max().item<int>();
    if (num_classes == -1) {
        num_classes = indices_max + 1;
    }
    if (indices_min < 0 || indices_max >= num_classes) {
        TT_EXCEPTION(std::format("indices contains a value outside the expected range of [0, {:d})", num_classes));
    }
    Tensor base = eye(num_classes, num_classes, kDefaultInt, indices.device());
    std::vector<int> _shape = indices.shape().to_vec();
    _shape.push_back(num_classes);
    return index_select(base, indices.flatten(), 0).reshape(Shape(_shape));
}

// ------------------------------------------------
// Tensor Joining
// ------------------------------------------------

namespace {

auto cat_runner(const TensorList &tensors, int dim, const std::optional<Tensor> &out) -> Tensor {
    if (tensors.empty()) {
        TT_EXCEPTION("Given empty vector of tensors to concatenate.");
    }
    if (tensors.size() == 1) {
        return tensors[0];
    }
    // Check if all on same device
    for (const auto &tensor : std::views::drop(tensors, 1)) {
        if (tensor.device() != tensors[0].device()) {
            TT_EXCEPTION(
                std::format(
                    "Expected all tensors to be on same device. Given tensors on devices {:s}",
                    concat_device_str(tensors)
                )
            );
        }
    }

    // Ensure all shapes have same size
    Shape ref_shape = tensors[0].shape();
    ScalarType res_type = tensors[0].dtype();
    for (const auto &tensor : std::views::drop(tensors, 1)) {
        if (tensor.shape().size() != ref_shape.size()) {
            TT_EXCEPTION(
                std::format(
                    "Given tensors do not have the same number of dimensions. Given shapes: {:s}",
                    concat_shape_str(tensors)
                )
            );
        }
        res_type = promote_types(res_type, tensor.dtype());
    }
    // Ensure given dim is within range
    CHECK_VALID_INDEX_EXCLUSIVE(dim, ref_shape.ndim());
    dim = indexing_exclusive(dim, ref_shape);

    // Ensure all non-concatenated dims match
    for (const auto &tensor : std::views::drop(tensors, 1)) {
        const auto &_shape = tensor.shape();
        for (int i = 0; i < ref_shape.size(); ++i) {
            if (i == dim) {
                ref_shape[i] += _shape[i];
                continue;
            }
            if (_shape[i] != ref_shape[i]) {
                TT_EXCEPTION(
                    std::format(
                        "Shapes of tensors must match except in concatenated dimension {:d}. Given shapes: {:s}",
                        dim,
                        concat_shape_str(tensors)
                    )
                );
            }
        }
    }

    // Get input tensors by converting to common type after all preconditions met
    TensorList converted_tensors;
    for (auto &tensor : tensors) {
        converted_tensors.push_back(tensor.to(res_type));
    }

    // If given an output tensor for storage to concat into, ensure its shape is valid
    if (out) {
        if (ref_shape != out.value().shape()) {
            TT_EXCEPTION(
                std::format(
                    "Given output tensor shape {:s} does not match required output shape {:s}",
                    out.value().shape(),
                    ref_shape
                )
            );
        }
    }

    // If non-inplace, call autograd version
    if (!out) {
        return autograd::TensorCat::apply(converted_tensors, dim);
    }

    // If in-place, ensure we aren't modifying a tensor requiring gradient computations
    CHECK_INPLACE_AUTOGRAD(out.value());

    // Create resulting tensor, and copy each input into corresponding slice
    int start_idx = 0;
    int end_idx = 0;
    std::vector<indexing::Index> slices(static_cast<std::size_t>(ref_shape.size()), indexing::Slice());
    Tensor result = out.value();
    for (const auto &tensor : converted_tensors) {
        start_idx = end_idx;
        end_idx += tensor.size(dim);
        slices[static_cast<std::size_t>(dim)] = indexing::Slice(start_idx, end_idx);
        result[slices] = tensor.to(res_type);
    }
    return result;
}

auto stack_runner(const TensorList &tensors, int dim, const std::optional<Tensor> &out) -> Tensor {
    if (tensors.empty()) {
        TT_EXCEPTION("Given empty vector of tensors to concatenate.");
    }
    /// Check if all on same device
    for (const auto &tensor : std::views::drop(tensors, 1)) {
        if (tensor.device() != tensors[0].device()) {
            TT_EXCEPTION(
                std::format(
                    "Expected all tensors to be on same device. Given tensors on devices {:s}",
                    concat_device_str(tensors)
                )
            );
        }
    }

    // Ensure all tensors have same shape
    Shape ref_shape = tensors[0].shape();
    ScalarType res_type = tensors[0].dtype();
    for (const auto &tensor : std::views::drop(tensors, 1)) {
        if (tensor.shape() != ref_shape) {
            TT_EXCEPTION(
                std::format("Given tensors do not have the same shape. Given shapes: {:s}", concat_shape_str(tensors))
            );
        }
        res_type = promote_types(res_type, tensor.dtype());
    }

    // Ensure given dim is within range
    CHECK_VALID_INDEX_INCLUSIVE(dim, ref_shape.ndim());
    dim = indexing_inclusive(dim, ref_shape);

    // Stack = concatenate over unsqueezed tensors at given dim
    TensorList expanded_tensors =
        std::views::transform(tensors, [&](const Tensor &tensor) { return tensor.unsqueeze(dim); })
        | tinytensor::to<TensorList>();
    return cat_runner(expanded_tensors, dim, out);
}

}    // namespace

auto cat(const TensorList &tensors, int dim) -> Tensor {
    for (const auto &tensor : tensors) {
        CHECK_VALID_TENSOR(tensor);
    }
    return cat_runner(tensors, dim, {});
}
void cat(const TensorList &tensors, int dim, Tensor &out) {
    for (const auto &tensor : tensors) {
        CHECK_VALID_TENSOR(tensor);
    }
    out = cat_runner(tensors, dim, out);
}

auto stack(const TensorList &tensors, int dim) -> Tensor {
    for (const auto &tensor : tensors) {
        CHECK_VALID_TENSOR(tensor);
    }
    return stack_runner(tensors, dim, {});
}
void stack(const TensorList &tensors, int dim, Tensor &out) {
    for (const auto &tensor : tensors) {
        CHECK_VALID_TENSOR(tensor);
    }
    out = stack_runner(tensors, dim, out);
}

// ------------------------------------------------
// Uniform Distributions
// ------------------------------------------------

// NOLINTNEXTLINE*(-macro-usage)
#define CHECK_UNIFORM(low, high)                                                  \
    DISABLE_BOOL(low, uniform);                                                   \
    DISABLE_BOOL(high, uniform);                                                  \
    if ((high <= low).any()) {                                                    \
        TT_EXCEPTION("Expected low < high. At least one element violates this."); \
    }

auto uniform_real(const Tensor &low, const Tensor &high, bool requires_grad, Generator &gen) -> Tensor {
    CHECK_VALID_TENSOR(low);
    CHECK_VALID_TENSOR(high);
    CHECK_SAME_DEVICE(low, high);
    CHECK_SAME_SHAPE(low, high);
    CHECK_UNIFORM(low, high);
    auto result_type = promote_types(promote_types(low.dtype(), high.dtype()), kDefaultFloat);
    CHECK_REQUIRES_GRAD(result_type, requires_grad);
    auto result = get_backend(low.device())->uniform_real(low.to(result_type), high.to(result_type), gen);
    result.set_requires_grad(requires_grad);
    return result;
}
auto uniform_real(double low, double high, Shape shape, const TensorOptions &options, Generator &gen) -> Tensor {
    CHECK_EMPTY_SHAPE(shape);
    auto result_type = is_float_dtype(options.dtype()) ? options.dtype() : kDefaultFloat;
    return uniform_real(
        Tensor(Scalar(low, result_type), options.device()).expand(shape),
        Tensor(Scalar(high, result_type), options.device()).expand(shape),
        options.requires_grad(),
        gen
    );
}
auto Tensor::uniform_real_(const Tensor &low, const Tensor &high, Generator &gen) -> Tensor & {
    CHECK_VALID_TENSOR(low);
    CHECK_VALID_TENSOR(high);
    CHECK_SAME_DEVICE(*this, low, high);
    CHECK_SAME_SHAPE(*this, low, high);
    CHECK_SELF_FLOAT();
    CHECK_UNIFORM(low, high);
    CHECK_INPLACE_AUTOGRAD(*this);
    get_backend(device_)->uniform_real_(*this, low.to(scalar_type_), high.to(scalar_type_), gen);
    ++version_count_;
    return *this;
}
auto Tensor::uniform_real_(double low, double high, Generator &gen) -> Tensor & {
    return uniform_real_(
        Tensor(Scalar(low, scalar_type_), device_).expand(shape_),
        Tensor(Scalar(high, scalar_type_), device_).expand(shape_),
        gen
    );
}

auto uniform_int(const Tensor &low, const Tensor &high, Generator &gen) -> Tensor {
    CHECK_VALID_TENSOR(low);
    CHECK_VALID_TENSOR(high);
    CHECK_SAME_DEVICE(low, high);
    CHECK_SAME_SHAPE(low, high);
    CHECK_UNIFORM(low, high);
    auto result_type = promote_types(promote_types(low.dtype(), high.dtype()), kDefaultInt);
    return get_backend(low.device())->uniform_int(low.to(result_type), high.to(result_type), gen);
}
auto uniform_int(int64_t low, int64_t high, Shape shape, const TensorOptions &options, Generator &gen) -> Tensor {
    CHECK_EMPTY_SHAPE(shape);
    auto result_type = is_integral_dtype(options.dtype()) ? options.dtype() : kDefaultInt;
    return uniform_int(
        Tensor(Scalar(low, result_type), options.device()).expand(shape),
        Tensor(Scalar(high, result_type), options.device()).expand(shape),
        gen
    );
}
auto Tensor::uniform_int_(const Tensor &low, const Tensor &high, Generator &gen) -> Tensor & {
    CHECK_VALID_TENSOR(low);
    CHECK_VALID_TENSOR(high);
    CHECK_SAME_DEVICE(*this, low, high);
    CHECK_SAME_SHAPE(*this, low, high);
    DISABLE_BOOL(*this, uniform_int);
    CHECK_UNIFORM(low, high);
    CHECK_INPLACE_AUTOGRAD(*this);
    get_backend(device_)->uniform_int_(*this, low.to(scalar_type_), high.to(scalar_type_), gen);
    ++version_count_;
    return *this;
}
auto Tensor::uniform_int_(int64_t low, int64_t high, Generator &gen) -> Tensor & {
    return uniform_int_(
        Tensor(Scalar(low, scalar_type_), device_).expand(shape_),
        Tensor(Scalar(high, scalar_type_), device_).expand(shape_),
        gen
    );
}

#undef CHECK_UNIFORM

// ------------------------------------------------------------
// Bernoulli Distributions
// ------------------------------------------------------------

// NOLINTNEXTLINE*(-macro-usage)
#define CHECK_BERNOULLI(p)                                                                    \
    DISABLE_BOOL(p, bernoulli);                                                               \
    if ((p < 0 || p > 1).any()) {                                                             \
        TT_EXCEPTION("Expected p in range [0, 1]. At least one element in p violates this."); \
    }

// NOLINTNEXTLINE*(-macro-usage)
#define CHECK_BINOMIAL(p, num_draws)                                                               \
    DISABLE_BOOL(p, binomial);                                                                     \
    DISABLE_BOOL(num_draws, binomial);                                                             \
    if ((p < 0 || p > 1).any()) {                                                                  \
        TT_EXCEPTION("Expected p in range [0, 1]. At least one element in p violates this.");      \
    }                                                                                              \
    if ((num_draws < 0).any()) {                                                                   \
        TT_EXCEPTION("Expected num_draws >= 0. At least one element in num_draws violates this."); \
    }

// NOLINTNEXTLINE*(-macro-usage)
#define CHECK_GEOMETRIC(p)                                                                    \
    DISABLE_BOOL(p, geometric);                                                               \
    if ((p <= 0 || p > 1).any()) {                                                            \
        TT_EXCEPTION("Expected p in range (0, 1]. At least one element in p violates this."); \
    }

auto bernoulli(const Tensor &p, bool requires_grad, Generator &gen) -> Tensor {
    CHECK_VALID_TENSOR(p);
    CHECK_BERNOULLI(p);
    auto result_type = promote_types(p.dtype(), kDefaultFloat);
    CHECK_REQUIRES_GRAD(result_type, requires_grad);
    auto result = get_backend(p.device())->bernoulli(p.to(result_type), gen);
    result.set_requires_grad(true);
    return result;
}
auto bernoulli(double p, Shape shape, const TensorOptions &options, Generator &gen) -> Tensor {
    CHECK_EMPTY_SHAPE(shape);
    auto result_type = is_float_dtype(options.dtype()) ? options.dtype() : kDefaultFloat;
    return bernoulli(Tensor(Scalar(p, result_type), options.device()).expand(shape), options.requires_grad(), gen);
}
auto Tensor::bernoulli_(const Tensor &p, Generator &gen) -> Tensor & {
    CHECK_VALID_TENSOR(p);
    CHECK_SAME_DEVICE(*this, p);
    CHECK_SAME_SHAPE(*this, p);
    CHECK_BERNOULLI(p);
    CHECK_SELF_FLOAT();
    CHECK_INPLACE_AUTOGRAD(*this);
    get_backend(device_)->bernoulli_(*this, p.to(scalar_type_), gen);
    ++version_count_;
    return *this;
}
auto Tensor::bernoulli_(double p, Generator &gen) -> Tensor & {
    return bernoulli_(Tensor(Scalar(p, scalar_type_), device_).expand(shape_), gen);
}

auto binomial(const Tensor &p, const Tensor &num_draws, bool requires_grad, Generator &gen) -> Tensor {
    CHECK_VALID_TENSOR(p);
    CHECK_VALID_TENSOR(num_draws);
    CHECK_BINOMIAL(p, num_draws);
    auto result_type = promote_types(promote_types(p.dtype(), num_draws.dtype()), kDefaultFloat);
    CHECK_REQUIRES_GRAD(result_type, requires_grad);
    auto result = get_backend(p.device())->binomial(p.to(result_type), num_draws.to(result_type), gen);
    result.set_requires_grad(requires_grad);
    return result;
}
auto binomial(double p, int num_draws, Shape shape, const TensorOptions &options, Generator &gen) -> Tensor {
    CHECK_EMPTY_SHAPE(shape);
    auto result_type = is_float_dtype(options.dtype()) ? options.dtype() : kDefaultFloat;
    return binomial(
        Tensor(Scalar(p, result_type), options.device()).expand(shape),
        Tensor(Scalar(num_draws, result_type), options.device()).expand(shape),
        options.requires_grad(),
        gen
    );
}
auto Tensor::binomial_(const Tensor &p, const Tensor &num_draws, Generator &gen) -> Tensor & {
    CHECK_VALID_TENSOR(p);
    CHECK_VALID_TENSOR(num_draws);
    CHECK_SAME_DEVICE(*this, p, num_draws);
    CHECK_SAME_SHAPE(*this, p, num_draws);
    CHECK_BINOMIAL(p, num_draws);
    CHECK_SELF_FLOAT();
    CHECK_INPLACE_AUTOGRAD(*this);
    get_backend(device_)->binomial_(*this, p.to(scalar_type_), num_draws.to(scalar_type_), gen);
    ++version_count_;
    return *this;
}
auto Tensor::binomial_(double p, int num_draws, Generator &gen) -> Tensor & {
    return binomial_(
        Tensor(Scalar(p, scalar_type_), device_).expand(shape_),
        Tensor(Scalar(num_draws, scalar_type_), device_).expand(shape_),
        gen
    );
}

auto geometric(const Tensor &p, bool requires_grad, Generator &gen) -> Tensor {
    CHECK_VALID_TENSOR(p);
    CHECK_GEOMETRIC(p);
    auto result_type = promote_types(p.dtype(), kDefaultFloat);
    CHECK_REQUIRES_GRAD(result_type, requires_grad);
    auto result = get_backend(p.device())->geometric(p.to(result_type), gen);
    result.set_requires_grad(requires_grad);
    return result;
}
auto geometric(double p, Shape shape, const TensorOptions &options, Generator &gen) -> Tensor {
    CHECK_EMPTY_SHAPE(shape);
    auto result_type = is_float_dtype(options.dtype()) ? options.dtype() : kDefaultFloat;
    return geometric(Tensor(Scalar(p, result_type), options.device()).expand(shape), options.requires_grad(), gen);
}
auto Tensor::geometric_(const Tensor &p, Generator &gen) -> Tensor & {
    CHECK_VALID_TENSOR(p);
    CHECK_SAME_DEVICE(*this, p);
    CHECK_SAME_SHAPE(*this, p);
    CHECK_GEOMETRIC(p);
    CHECK_SELF_FLOAT();
    CHECK_INPLACE_AUTOGRAD(*this);
    get_backend(device_)->geometric_(*this, p.to(scalar_type_), gen);
    ++version_count_;
    return *this;
}
auto Tensor::geometric_(double p, Generator &gen) -> Tensor & {
    return geometric_(Tensor(Scalar(p, scalar_type_), device_).expand(shape_), gen);
}

#undef CHECK_GEOMETRIC
#undef CHECK_BINOMIAL
#undef CHECK_BERNOULLI

// ------------------------------------------------------------
// Exponential Distributions
// ------------------------------------------------------------

// NOLINTNEXTLINE*(-macro-usage)
#define CHECK_EXPONENTIAL(lambda)                                                           \
    if ((lambda <= 0).any()) {                                                              \
        TT_EXCEPTION("Expected lambda > 0. At least one element in lambda violates this."); \
    }

auto poisson(const Tensor &lambda, bool requires_grad, Generator &gen) -> Tensor {
    CHECK_VALID_TENSOR(lambda);
    DISABLE_BOOL(lambda, poisson);
    CHECK_EXPONENTIAL(lambda);
    auto result_type = promote_types(lambda.dtype(), kDefaultFloat);
    CHECK_REQUIRES_GRAD(result_type, requires_grad);
    auto result = get_backend(lambda.device())->poisson(lambda.to(result_type), gen);
    result.set_requires_grad(requires_grad);
    return result;
}
auto poisson(double lambda, Shape shape, const TensorOptions &options, Generator &gen) -> Tensor {
    auto result_type = is_float_dtype(options.dtype()) ? options.dtype() : kDefaultFloat;
    return poisson(Tensor(Scalar(lambda, result_type), options.device()).expand(shape), options.requires_grad(), gen);
}
auto Tensor::poisson_(const Tensor &lambda, Generator &gen) -> Tensor & {
    CHECK_VALID_TENSOR(lambda);
    CHECK_SAME_DEVICE(*this, lambda);
    CHECK_SAME_SHAPE(*this, lambda);
    DISABLE_BOOL(lambda, poisson);
    CHECK_EXPONENTIAL(lambda);
    CHECK_SELF_FLOAT();
    CHECK_INPLACE_AUTOGRAD(*this);
    get_backend(device_)->poisson_(*this, lambda.to(scalar_type_), gen);
    ++version_count_;
    return *this;
}
auto Tensor::poisson_(double lambda, Generator &gen) -> Tensor & {
    return poisson_(Tensor(Scalar(lambda, scalar_type_), device_).expand(shape_), gen);
}

auto exponential(const Tensor &lambda, bool requires_grad, Generator &gen) -> Tensor {
    CHECK_VALID_TENSOR(lambda);
    DISABLE_BOOL(lambda, exponential);
    CHECK_EXPONENTIAL(lambda);
    auto result_type = promote_types(lambda.dtype(), kDefaultFloat);
    CHECK_REQUIRES_GRAD(result_type, requires_grad);
    auto result = get_backend(lambda.device())->exponential(lambda.to(result_type), gen);
    result.set_requires_grad(requires_grad);
    return result;
}
auto exponential(double lambda, Shape shape, const TensorOptions &options, Generator &gen) -> Tensor {
    CHECK_EMPTY_SHAPE(shape);
    auto result_type = is_float_dtype(options.dtype()) ? options.dtype() : kDefaultFloat;
    return exponential(
        Tensor(Scalar(lambda, result_type), options.device()).expand(shape),
        options.requires_grad(),
        gen
    );
}
auto Tensor::exponential_(const Tensor &lambda, Generator &gen) -> Tensor & {
    CHECK_VALID_TENSOR(lambda);
    CHECK_SAME_DEVICE(*this, lambda);
    CHECK_SAME_SHAPE(*this, lambda);
    DISABLE_BOOL(lambda, exponential);
    CHECK_EXPONENTIAL(lambda);
    CHECK_SELF_FLOAT();
    CHECK_INPLACE_AUTOGRAD(*this);
    get_backend(device_)->exponential_(*this, lambda.to(scalar_type_), gen);
    ++version_count_;
    return *this;
}
auto Tensor::exponential_(double lambda, Generator &gen) -> Tensor & {
    return exponential_(Tensor(Scalar(lambda, scalar_type_), device_).expand(shape_), gen);
}

#undef CHECK_EXPONENTIAL

// ------------------------------------------------------------
// Normal Distributions
// ------------------------------------------------------------

// NOLINTNEXTLINE*(-macro-usage)
#define CHECK_NORMAL(std)                                                             \
    if ((std <= 0).any()) {                                                           \
        TT_EXCEPTION("Expected std > 0. At least one element in std violates this."); \
    }

// NOLINTNEXTLINE*(-macro-usage)
#define CHECK_CAUCHY(scale)                                                               \
    if ((scale <= 0).any()) {                                                             \
        TT_EXCEPTION("Expected scale > 0. At least one element in scale violates this."); \
    }

// NOLINTNEXTLINE*(-macro-usage)
#define CHECK_WEIBULL(lambda, k)                                                              \
    if ((lambda <= 0 || k <= 0).any()) {                                                      \
        TT_EXCEPTION("Expected both lambda > 0, k > 0. At least one element violates this."); \
    }

auto normal(const Tensor &mu, const Tensor &std, bool requires_grad, Generator &gen) -> Tensor {
    CHECK_VALID_TENSOR(mu);
    CHECK_VALID_TENSOR(std);
    CHECK_SAME_DEVICE(mu, std);
    CHECK_SAME_SHAPE(mu, std);
    DISABLE_BOOL(mu, normal);
    DISABLE_BOOL(std, normal);
    CHECK_NORMAL(std);
    auto result_type = promote_types(promote_types(mu.dtype(), std.dtype()), kDefaultFloat);
    CHECK_REQUIRES_GRAD(result_type, requires_grad);
    auto result = get_backend(mu.device())->normal(mu.to(result_type), std.to(result_type), gen);
    result.set_requires_grad(requires_grad);
    return result;
}
auto normal(double mu, double std, Shape shape, const TensorOptions &options, Generator &gen) -> Tensor {
    CHECK_EMPTY_SHAPE(shape);
    auto result_type = is_float_dtype(options.dtype()) ? options.dtype() : kDefaultFloat;
    return normal(
        Tensor(Scalar(mu, result_type), options.device()).expand(shape),
        Tensor(Scalar(std, result_type), options.device()).expand(shape),
        options.requires_grad(),
        gen
    );
}
auto Tensor::normal_(const Tensor &mu, const Tensor &std, Generator &gen) -> Tensor & {
    CHECK_VALID_TENSOR(mu);
    CHECK_VALID_TENSOR(std);
    CHECK_SAME_DEVICE(*this, mu, std);
    CHECK_SAME_SHAPE(*this, mu, std);
    DISABLE_BOOL(mu, normal);
    DISABLE_BOOL(std, normal);
    CHECK_NORMAL(std);
    CHECK_SELF_FLOAT();
    CHECK_INPLACE_AUTOGRAD(*this);
    get_backend(device_)->normal_(*this, mu.to(scalar_type_), std.to(scalar_type_), gen);
    ++version_count_;
    return *this;
}
auto Tensor::normal_(double mu, double std, Generator &gen) -> Tensor & {
    return normal_(
        Tensor(Scalar(mu, scalar_type_), device_).expand(shape_),
        Tensor(Scalar(std, scalar_type_), device_).expand(shape_),
        gen
    );
}

auto cauchy(const Tensor &loc, const Tensor &scale, bool requires_grad, Generator &gen) -> Tensor {
    CHECK_VALID_TENSOR(loc);
    CHECK_VALID_TENSOR(scale);
    CHECK_SAME_DEVICE(loc, scale);
    CHECK_SAME_SHAPE(loc, scale);
    DISABLE_BOOL(loc, cauchy);
    DISABLE_BOOL(scale, cauchy);
    CHECK_CAUCHY(scale);
    auto result_type = promote_types(promote_types(loc.dtype(), scale.dtype()), kDefaultFloat);
    CHECK_REQUIRES_GRAD(result_type, requires_grad);
    auto result = get_backend(loc.device())->cauchy(loc.to(result_type), scale.to(result_type), gen);
    result.set_requires_grad(requires_grad);
    return result;
}
auto cauchy(double loc, double scale, Shape shape, const TensorOptions &options, Generator &gen) -> Tensor {
    CHECK_EMPTY_SHAPE(shape);
    auto result_type = is_float_dtype(options.dtype()) ? options.dtype() : kDefaultFloat;
    return cauchy(
        Tensor(Scalar(loc, result_type), options.device()).expand(shape),
        Tensor(Scalar(scale, result_type), options.device()).expand(shape),
        options.requires_grad(),
        gen
    );
}
auto Tensor::cauchy_(const Tensor &loc, const Tensor &scale, Generator &gen) -> Tensor & {
    CHECK_VALID_TENSOR(loc);
    CHECK_VALID_TENSOR(scale);
    CHECK_SAME_DEVICE(*this, loc, scale);
    CHECK_SAME_SHAPE(*this, loc, scale);
    DISABLE_BOOL(loc, cauchy);
    DISABLE_BOOL(scale, cauchy);
    CHECK_CAUCHY(scale);
    CHECK_SELF_FLOAT();
    CHECK_INPLACE_AUTOGRAD(*this);
    get_backend(device_)->cauchy_(*this, loc.to(scalar_type_), scale.to(scalar_type_), gen);
    ++version_count_;
    return *this;
}
auto Tensor::cauchy_(double loc, double scale, Generator &gen) -> Tensor & {
    return cauchy_(
        Tensor(Scalar(loc, scalar_type_), device_).expand(shape_),
        Tensor(Scalar(scale, scalar_type_), device_).expand(shape_),
        gen
    );
}

auto lognormal(const Tensor &mu, const Tensor &std, bool requires_grad, Generator &gen) -> Tensor {
    CHECK_VALID_TENSOR(mu);
    CHECK_VALID_TENSOR(std);
    CHECK_SAME_DEVICE(mu, std);
    CHECK_SAME_SHAPE(mu, std);
    DISABLE_BOOL(mu, lognormal);
    DISABLE_BOOL(std, lognormal);
    CHECK_NORMAL(std);
    auto result_type = promote_types(promote_types(mu.dtype(), std.dtype()), kDefaultFloat);
    CHECK_REQUIRES_GRAD(result_type, requires_grad);
    auto result = get_backend(mu.device())->lognormal(mu.to(result_type), std.to(result_type), gen);
    result.set_requires_grad(requires_grad);
    return result;
}
auto lognormal(double mu, double std, Shape shape, const TensorOptions &options, Generator &gen) -> Tensor {
    CHECK_EMPTY_SHAPE(shape);
    auto result_type = is_float_dtype(options.dtype()) ? options.dtype() : kDefaultFloat;
    return lognormal(
        Tensor(Scalar(mu, result_type), options.device()).expand(shape),
        Tensor(Scalar(std, result_type), options.device()).expand(shape),
        options.requires_grad(),
        gen
    );
}
auto Tensor::lognormal_(const Tensor &mu, const Tensor &std, Generator &gen) -> Tensor & {
    CHECK_VALID_TENSOR(mu);
    CHECK_VALID_TENSOR(std);
    CHECK_SAME_DEVICE(*this, mu, std);
    CHECK_SAME_SHAPE(*this, mu, std);
    DISABLE_BOOL(mu, lognormal);
    DISABLE_BOOL(std, lognormal);
    CHECK_NORMAL(std);
    CHECK_SELF_FLOAT();
    CHECK_INPLACE_AUTOGRAD(*this);
    get_backend(device_)->lognormal_(*this, mu.to(scalar_type_), std.to(scalar_type_), gen);
    ++version_count_;
    return *this;
}
auto Tensor::lognormal_(double mu, double std, Generator &gen) -> Tensor & {
    return lognormal_(
        Tensor(Scalar(mu, scalar_type_), device_).expand(shape_),
        Tensor(Scalar(std, scalar_type_), device_).expand(shape_),
        gen
    );
}

auto weibull(const Tensor &lambda, const Tensor &k, bool requires_grad, Generator &gen) -> Tensor {
    CHECK_VALID_TENSOR(lambda);
    CHECK_VALID_TENSOR(k);
    CHECK_SAME_DEVICE(lambda, k);
    CHECK_SAME_SHAPE(lambda, k);
    DISABLE_BOOL(lambda, weibull);
    DISABLE_BOOL(k, weibull);
    CHECK_WEIBULL(lambda, k);
    auto result_type = promote_types(promote_types(lambda.dtype(), k.dtype()), kDefaultFloat);
    CHECK_REQUIRES_GRAD(result_type, requires_grad);
    auto result = get_backend(lambda.device())->weibull(lambda.to(result_type), k.to(result_type), gen);
    result.set_requires_grad(requires_grad);
    return result;
}
auto weibull(double lambda, double k, Shape shape, const TensorOptions &options, Generator &gen) -> Tensor {
    CHECK_EMPTY_SHAPE(shape);
    auto result_type = is_float_dtype(options.dtype()) ? options.dtype() : kDefaultFloat;
    return weibull(
        Tensor(Scalar(lambda, result_type), options.device()).expand(shape),
        Tensor(Scalar(k, result_type), options.device()).expand(shape),
        options.requires_grad(),
        gen
    );
}
auto Tensor::weibull_(const Tensor &lambda, const Tensor &k, Generator &gen) -> Tensor & {
    CHECK_VALID_TENSOR(lambda);
    CHECK_VALID_TENSOR(k);
    CHECK_SAME_DEVICE(*this, lambda, k);
    CHECK_SAME_SHAPE(*this, lambda, k);
    DISABLE_BOOL(lambda, weibull);
    DISABLE_BOOL(k, weibull);
    CHECK_WEIBULL(lambda, k);
    CHECK_SELF_FLOAT();
    CHECK_INPLACE_AUTOGRAD(*this);
    get_backend(device_)->weibull_(*this, lambda.to(scalar_type_), k.to(scalar_type_), gen);
    ++version_count_;
    return *this;
}
auto Tensor::weibull_(double lambda, double k, Generator &gen) -> Tensor & {
    return weibull_(
        Tensor(Scalar(lambda, scalar_type_), device_).expand(shape_),
        Tensor(Scalar(k, scalar_type_), device_).expand(shape_),
        gen
    );
}

#undef CHECK_WEIBULL
#undef CHECK_CAUCHY
#undef CHECK_NORMAL

// ------------------------------------------------
// Binary Operators
// ------------------------------------------------

// NOLINTNEXTLINE(*-macro-usage)
#define DECLARE_BINARY_AUTOGRAD_OPS(FUNC, AUTOGRAD_NAME, IS_BOOL_DISABLED)  \
    auto FUNC(const Tensor &lhs, const Tensor &rhs) -> Tensor {             \
        CHECK_VALID_TENSOR(lhs);                                            \
        CHECK_VALID_TENSOR(rhs);                                            \
        CHECK_SAME_DEVICE(lhs, rhs);                                        \
        CHECK_SAME_SHAPE(lhs, rhs);                                         \
        const auto promoted_type = promote_types(lhs.dtype(), rhs.dtype()); \
        const auto lhs_promoted = check_promote(lhs, promoted_type);        \
        const auto rhs_promoted = check_promote(rhs, promoted_type);        \
        if (IS_BOOL_DISABLED) {                                             \
            DISABLE_BOOL(lhs_promoted, FUNC);                               \
            DISABLE_BOOL(rhs_promoted, FUNC);                               \
        }                                                                   \
        return autograd::AUTOGRAD_NAME::apply(lhs_promoted, rhs_promoted);  \
    };                                                                      \
    auto FUNC(const Tensor &lhs, Scalar rhs) -> Tensor {                    \
        CHECK_VALID_TENSOR(lhs);                                            \
        return FUNC(lhs, Tensor(rhs, lhs.device()).expand(lhs.shape()));    \
    }                                                                       \
    auto FUNC(Scalar lhs, const Tensor &rhs) -> Tensor {                    \
        CHECK_VALID_TENSOR(rhs);                                            \
        return FUNC(Tensor(lhs, rhs.device()).expand(rhs.shape()), rhs);    \
    }

// NOLINTNEXTLINE(*-macro-usage)
#define DECLARE_BINARY_OPS(FUNC, IS_BOOL_DISABLED)                          \
    auto FUNC(const Tensor &lhs, const Tensor &rhs) -> Tensor {             \
        CHECK_VALID_TENSOR(lhs);                                            \
        CHECK_VALID_TENSOR(rhs);                                            \
        CHECK_SAME_DEVICE(lhs, rhs);                                        \
        CHECK_SAME_SHAPE(lhs, rhs);                                         \
        const auto promoted_type = promote_types(lhs.dtype(), rhs.dtype()); \
        const auto lhs_promoted = check_promote(lhs, promoted_type);        \
        const auto rhs_promoted = check_promote(rhs, promoted_type);        \
        if (IS_BOOL_DISABLED) {                                             \
            DISABLE_BOOL(lhs_promoted, FUNC);                               \
            DISABLE_BOOL(rhs_promoted, FUNC);                               \
        }                                                                   \
        return get_backend(lhs.device())->FUNC(lhs_promoted, rhs_promoted); \
    };                                                                      \
    auto FUNC(const Tensor &lhs, Scalar rhs) -> Tensor {                    \
        CHECK_VALID_TENSOR(lhs);                                            \
        return FUNC(lhs, Tensor(rhs, lhs.device()).expand(lhs.shape()));    \
    }                                                                       \
    auto FUNC(Scalar lhs, const Tensor &rhs) -> Tensor {                    \
        CHECK_VALID_TENSOR(rhs);                                            \
        return FUNC(Tensor(lhs, rhs.device()).expand(rhs.shape()), rhs);    \
    }

DECLARE_BINARY_OPS(eq, false);
DECLARE_BINARY_OPS(ne, false);
DECLARE_BINARY_OPS(lt, false);
DECLARE_BINARY_OPS(le, false);
DECLARE_BINARY_OPS(gt, false);
DECLARE_BINARY_OPS(ge, false);
DECLARE_BINARY_OPS(logical_or, false);
DECLARE_BINARY_OPS(logical_and, false);
DECLARE_BINARY_OPS(modulo, true);
#undef DECLARE_BINARY_OPS

DECLARE_BINARY_AUTOGRAD_OPS(add, TensorAdd, true);
DECLARE_BINARY_AUTOGRAD_OPS(sub, TensorSub, true);
DECLARE_BINARY_AUTOGRAD_OPS(mul, TensorMul, true);
DECLARE_BINARY_AUTOGRAD_OPS(div, TensorDiv, true);
DECLARE_BINARY_AUTOGRAD_OPS(maximum, TensorMaximum, false);
DECLARE_BINARY_AUTOGRAD_OPS(minimum, TensorMinimum, false);
DECLARE_BINARY_AUTOGRAD_OPS(pow, TensorPow, true);
#undef DECLARE_BINARY_AUTOGRAD_OPS

// Bitwise operators and others that need special treatment with respect to floating point values

// NOLINTNEXTLINE(*-macro-usage)
#define DECLARE_BINARY_INT_OPS(FUNC)                                                                                  \
    auto FUNC(const Tensor &lhs, const Tensor &rhs) -> Tensor {                                                       \
        CHECK_VALID_TENSOR(lhs);                                                                                      \
        CHECK_VALID_TENSOR(rhs);                                                                                      \
        CHECK_SAME_DEVICE(lhs, rhs);                                                                                  \
        CHECK_SAME_SHAPE(lhs, rhs);                                                                                   \
        if (is_float_dtype(lhs.dtype()) || is_float_dtype(rhs.dtype())) {                                             \
            TT_EXCEPTION("Expects arguments to be of integral dtype");                                                \
        }                                                                                                             \
        const auto promoted_type = promote_types(lhs.dtype(), rhs.dtype());                                           \
        return get_backend(lhs.device())->FUNC(check_promote(lhs, promoted_type), check_promote(rhs, promoted_type)); \
    };                                                                                                                \
    auto FUNC(const Tensor &lhs, Scalar rhs) -> Tensor {                                                              \
        CHECK_VALID_TENSOR(lhs);                                                                                      \
        return FUNC(lhs, Tensor(rhs, lhs.device()).expand(lhs.shape()));                                              \
    }                                                                                                                 \
    auto FUNC(Scalar lhs, const Tensor &rhs) -> Tensor {                                                              \
        CHECK_VALID_TENSOR(rhs);                                                                                      \
        return FUNC(Tensor(lhs, rhs.device()).expand(rhs.shape()), rhs);                                              \
    }
DECLARE_BINARY_INT_OPS(bitwise_or);
DECLARE_BINARY_INT_OPS(bitwise_and);
DECLARE_BINARY_INT_OPS(bitwise_xor);
#undef DECLARE_BINARY_INT_OPS

// NOLINTNEXTLINE(*-macro-usage)
#define DECLARE_BINARY_SHIFT_OPS(FUNC)                                                \
    auto FUNC(const Tensor &lhs, const Tensor &rhs) -> Tensor {                       \
        CHECK_VALID_TENSOR(lhs);                                                      \
        CHECK_VALID_TENSOR(rhs);                                                      \
        CHECK_SAME_DEVICE(lhs, rhs);                                                  \
        CHECK_SAME_SHAPE(lhs, rhs);                                                   \
        DISABLE_BOOL(lhs, FUNC);                                                      \
        DISABLE_BOOL(rhs, FUNC);                                                      \
        if (is_float_dtype(lhs.dtype()) || is_float_dtype(rhs.dtype())) {             \
            TT_EXCEPTION("Bitwise shift expects arguments to be of integral dtype");  \
        }                                                                             \
        return get_backend(lhs.device())->FUNC(lhs, check_promote(rhs, lhs.dtype())); \
    };                                                                                \
    auto FUNC(const Tensor &lhs, Scalar rhs) -> Tensor {                              \
        CHECK_VALID_TENSOR(lhs);                                                      \
        return FUNC(lhs, Tensor(rhs, lhs.device()).expand(lhs.shape()));              \
    }                                                                                 \
    auto FUNC(Scalar lhs, const Tensor &rhs) -> Tensor {                              \
        CHECK_VALID_TENSOR(rhs);                                                      \
        return FUNC(Tensor(lhs, rhs.device()).expand(rhs.shape()), rhs);              \
    }
DECLARE_BINARY_SHIFT_OPS(bitwise_left_shift);
DECLARE_BINARY_SHIFT_OPS(bitwise_right_shift);
#undef DECLARE_BINARY_SHIFT_OPS

// Misc binary operators
auto matmul(const Tensor &_lhs, const Tensor &_rhs) -> Tensor {
    CHECK_VALID_TENSOR(_lhs);
    CHECK_VALID_TENSOR(_rhs);
    CHECK_SAME_DEVICE(_lhs, _rhs);
    const auto promoted_type = promote_types(_lhs.dtype(), _rhs.dtype());
    if (promoted_type == kBool) {
        TT_EXCEPTION(std::format("matmul does not support boolean tensors. Given {:s}", concat_dtype_str(_lhs, _rhs)));
    }
    auto lhs = check_promote(_lhs, promoted_type);
    auto rhs = check_promote(_rhs, promoted_type);
    const auto lhs_shape = lhs.shape();
    const auto rhs_shape = rhs.shape();

    if (lhs_shape.ndim() == 1 && rhs_shape.ndim() == 1) {    // Dot product
        if (lhs_shape[0] != rhs_shape[0]) {
            TT_EXCEPTION(
                std::format(
                    "Invalid shapes for dot product. Both LHS and RHS must have same dimensions: "
                    "LHS={} and RHS={}",
                    lhs_shape.to_string(),
                    rhs_shape.to_string()
                )
            );
        }
        return autograd::TensorBatchedMatmul::apply(
                   lhs.reshape({1, 1, lhs_shape[0]}),
                   rhs.reshape({1, rhs_shape[0], 1})
        )
            .flatten();
    } else if (lhs_shape.ndim() == 1 && rhs_shape.ndim() == 2) {    // Vector-Matrix product
        if (lhs_shape[0] != rhs_shape[0]) {
            TT_EXCEPTION(
                std::format(
                    "Invalid shape for matrix multiplication. Length of LHS should equal size of RHS "
                    "dimension 0: LHS={} "
                    "and RHS={}",
                    lhs_shape.to_string(),
                    rhs_shape.to_string()
                )
            );
        }
        return autograd::TensorBatchedMatmul::apply(lhs.reshape({1, 1, lhs_shape[0]}), rhs.unsqueeze(0)).flatten();
    } else if (lhs_shape.ndim() == 2 && rhs_shape.ndim() == 1) {    // Matrix-Vector product
        if (lhs_shape[1] != rhs_shape[0]) {
            TT_EXCEPTION(
                std::format(
                    "Invalid shape for matrix multiplication. Length of RHS should equal size of LHS "
                    "dimension 1: LHS={} "
                    "and RHS={}",
                    lhs_shape.to_string(),
                    rhs_shape.to_string()
                )
            );
        }
        return autograd::TensorBatchedMatmul::apply(lhs.unsqueeze(0), rhs.reshape({1, rhs_shape[0], 1})).flatten();
    } else if (lhs_shape.ndim() == 2 && rhs_shape.ndim() == 2) {    // Matmul product
        if (lhs_shape[1] != rhs_shape[0]) {
            TT_EXCEPTION(
                std::format(
                    "Invalid shape for matrix multiplication. Size of LHS dimension 1 should equal "
                    "size of RHS dimension "
                    "0: "
                    "LHS={} "
                    "and RHS={}",
                    lhs_shape.to_string(),
                    rhs_shape.to_string()
                )
            );
        }
        return autograd::TensorBatchedMatmul::apply(lhs.unsqueeze(0), rhs.unsqueeze(0)).squeeze(0);
    } else if (lhs_shape.ndim() == 3 && rhs_shape.ndim() == 3) {    // Batched Matmul product
        if (lhs_shape[2] != rhs_shape[1]) {
            TT_EXCEPTION(
                std::format(
                    "Invalid shape for batched matrix multiplication. Size of LHS dimension 2 should "
                    "equal size of RHS "
                    "dimension 1: LHS={} and RHS={}",
                    lhs_shape.to_string(),
                    rhs_shape.to_string()
                )
            );
        }
        return autograd::TensorBatchedMatmul::apply(lhs, rhs);
    }

    // Unsupported shape operation
    TT_EXCEPTION(
        "Unsupported sizes. Try reshaping or broadcasting. Expected dimensions of: "
        "\n\t(LHS=1, RHS=1) for dot product,"
        "\n\t(LHS=1, RHS=2) for vector-matrix product,"
        "\n\t(LHS=2, RHS=1) for matrix-vector product,"
        "\n\t(LHS=2, RHS=2) for matrix-matrix multiplication, or"
        "\n\t(LHS=3, RHS=3) for batched matrix multiplication."
    );
}

// ------------------------------------------------
// Reduction operations
// ------------------------------------------------

// NOLINTNEXTLINE(*-macro-usage)
#define DECLARE_REDUCTION_OP(FUNC, ALL_REDUCE_TYPE)             \
    auto Tensor::FUNC() const -> ALL_REDUCE_TYPE {              \
        CHECK_VALID_TENSOR(*this);                              \
        return get_backend(device_)->FUNC(*this);               \
    }                                                           \
    auto Tensor::FUNC(int dim, bool keep_dim) const -> Tensor { \
        CHECK_VALID_TENSOR(*this);                              \
        CHECK_VALID_INDEX_EXCLUSIVE(dim, shape_.ndim());        \
        dim = indexing_exclusive(dim, shape_);                  \
        Tensor result = get_backend(device_)->FUNC(*this, dim); \
        return !keep_dim ? result.squeeze(dim) : result;        \
    }

// NOLINTNEXTLINE(*-macro-usage)
#define DECLARE_REDUCTION_AUTOGRAD_OP(FUNC, AUTOGRAD_ALL_NAME, AUTOGRAD_DIM_NAME, ALL_REDUCE_TYPE) \
    auto Tensor::FUNC() const -> ALL_REDUCE_TYPE {                                                 \
        CHECK_VALID_TENSOR(*this);                                                                 \
        return autograd::AUTOGRAD_ALL_NAME::apply(*this);                                          \
    }                                                                                              \
    auto Tensor::FUNC(int dim, bool keep_dim) const -> Tensor {                                    \
        CHECK_VALID_TENSOR(*this);                                                                 \
        CHECK_VALID_INDEX_EXCLUSIVE(dim, shape_.ndim());                                           \
        dim = indexing_exclusive(dim, shape_);                                                     \
        return autograd::AUTOGRAD_DIM_NAME::apply(*this, dim, keep_dim);                           \
    }

// NOLINTNEXTLINE(*-macro-usage)
#define DECLARE_REDUCTION_GLOBAL_OP(FUNC, ALL_REDUCE_TYPE)              \
    auto FUNC(const Tensor &tensor) -> ALL_REDUCE_TYPE {                \
        CHECK_VALID_TENSOR(tensor);                                     \
        return tensor.FUNC();                                           \
    }                                                                   \
    auto FUNC(const Tensor &tensor, int dim, bool keep_dim) -> Tensor { \
        CHECK_VALID_TENSOR(tensor);                                     \
        return tensor.FUNC(dim, keep_dim);                              \
    }

// NOLINTNEXTLINE(*-macro-usage)
#define DECLARE_REDUCTION_OPS(FUNC, ALL_REDUCE_TYPE) \
    DECLARE_REDUCTION_OP(FUNC, ALL_REDUCE_TYPE)      \
    DECLARE_REDUCTION_GLOBAL_OP(FUNC, ALL_REDUCE_TYPE)

// NOLINTNEXTLINE(*-macro-usage)
#define DECLARE_REDUCTION_AUTOGRAD_OPS(FUNC, AUTOGRAD_ALL_NAME, AUTOGRAD_DIM_NAME, ALL_REDUCE_TYPE) \
    DECLARE_REDUCTION_AUTOGRAD_OP(FUNC, AUTOGRAD_ALL_NAME, AUTOGRAD_DIM_NAME, ALL_REDUCE_TYPE)      \
    DECLARE_REDUCTION_GLOBAL_OP(FUNC, ALL_REDUCE_TYPE)

DECLARE_REDUCTION_AUTOGRAD_OPS(min, TensorMinAll, TensorMinDim, Tensor);
DECLARE_REDUCTION_AUTOGRAD_OPS(max, TensorMaxAll, TensorMaxDim, Tensor);
DECLARE_REDUCTION_AUTOGRAD_OPS(sum, TensorSumAll, TensorSumDim, Tensor);
DECLARE_REDUCTION_AUTOGRAD_OPS(mean, TensorMeanAll, TensorMeanDim, Tensor);
DECLARE_REDUCTION_OPS(argmin, Tensor);
DECLARE_REDUCTION_OPS(argmax, Tensor);
DECLARE_REDUCTION_OPS(all, bool);
DECLARE_REDUCTION_OPS(any, bool);

#undef DECLARE_REDUCTION_AUTOGRAD_OPS
#undef DECLARE_REDUCTION_OPS
#undef DECLARE_REDUCTION_GLOBAL_OP
#undef DECLARE_REDUCTION_AUTOGRAD_OP
#undef DECLARE_REDUCTION_OP

namespace {
auto prepare_multi_dim_reduce(const Tensor &input, std::vector<int> dims) -> std::pair<Tensor, std::vector<int>> {
    CHECK_VALID_TENSOR(input);
    for (auto &dim : dims) {
        CHECK_VALID_INDEX_EXCLUSIVE(dim, input.dim());
        dim = indexing_exclusive(dim, input.shape());
    }
    // Get unique values
    int N_old = static_cast<int>(dims.size());
    std::ranges::sort(dims);
    const auto [first, last] = std::ranges::unique(dims);
    dims.erase(first, last);
    if (static_cast<int>(dims.size()) != N_old) {
        TT_EXCEPTION("Given dim list contains no-unique values");
    }

    // Reshape so reduction dims are innermost
    auto permuted_dims = dims;
    std::vector<int> stay_dims;
    for (int i : std::views::iota(0, input.dim()) | std::views::reverse) {
        if (std::ranges::find(dims, i) == dims.end()) {
            permuted_dims.insert(permuted_dims.begin(), i);
            stay_dims.insert(stay_dims.begin(), input.size(i));
        }
    }
    int N = 1;
    for (const auto &i : dims) {
        N *= input.size(i);
    }

    stay_dims.push_back(N);
    Tensor x = input.permute(permuted_dims).reshape(Shape{stay_dims});
    return {x, permuted_dims};
}

auto undo_multi_dim_reduce(Tensor result, const std::vector<int> &permuted_dims, int N) -> Tensor {
    for ([[maybe_unused]] int i : std::views::iota(0, N - result.dim())) {
        result = result.unsqueeze(-1);
    }
    // Compute inverse permutation
    auto inverse_dims = permuted_dims;
    for (int i = 0; i < static_cast<int>(permuted_dims.size()); ++i) {
        inverse_dims[static_cast<std::size_t>(permuted_dims[static_cast<std::size_t>(i)])] = i;
    }
    return result.permute(inverse_dims);
}
}    // namespace

// NOLINTNEXTLINE(*-macro-usage)
#define DECLARE_REDUCTION_MULTI_DIM(FUNC)                                                   \
    auto FUNC(const Tensor &input, const std::vector<int> &dims, bool keep_dim) -> Tensor { \
        CHECK_VALID_TENSOR(input);                                                          \
        auto [x, permuted_dims] = prepare_multi_dim_reduce(input, dims);                    \
        Tensor result = FUNC(x, -1, keep_dim);                                              \
        if (keep_dim) {                                                                     \
            result = undo_multi_dim_reduce(result, permuted_dims, input.dim());             \
        }                                                                                   \
        return result;                                                                      \
    }                                                                                       \
    auto Tensor::FUNC(const std::vector<int> &dims, bool keep_dim) const -> Tensor {        \
        CHECK_VALID_TENSOR(*this);                                                          \
        return tinytensor::FUNC(*this, dims, keep_dim);                                     \
    }

DECLARE_REDUCTION_MULTI_DIM(min);
DECLARE_REDUCTION_MULTI_DIM(argmin);
DECLARE_REDUCTION_MULTI_DIM(max);
DECLARE_REDUCTION_MULTI_DIM(argmax);
DECLARE_REDUCTION_MULTI_DIM(sum);
DECLARE_REDUCTION_MULTI_DIM(mean);
DECLARE_REDUCTION_MULTI_DIM(all);
DECLARE_REDUCTION_MULTI_DIM(any);

#undef DECLARE_REDUCTION_MULTI_DIM

auto var(const Tensor &input, int dim, bool keep_dim, int correction) -> Tensor {
    CHECK_VALID_TENSOR(input);
    CHECK_VALID_INDEX_EXCLUSIVE(dim, input.dim());
    dim = indexing_exclusive(dim, input.shape());
    int N = input.size(dim);
    double denom = 1.0 / std::max(0.0, static_cast<double>(N - correction));
    Tensor m = mean(input, dim, true).expand(input.shape());
    Tensor result = denom * sum(pow(input - m, 2), dim, true);
    return keep_dim ? result : result.squeeze(dim);
}
auto var(const Tensor &input, const std::vector<int> &dims, bool keep_dim, int correction) -> Tensor {
    auto [x, permuted_dims] = prepare_multi_dim_reduce(input, dims);
    Tensor result = var(x, -1, keep_dim, correction);
    if (keep_dim) {
        result = undo_multi_dim_reduce(result, permuted_dims, input.dim());
    }

    return result;
}

auto var(const Tensor &input, bool keep_dim, int correction) -> Tensor {
    CHECK_VALID_TENSOR(input);
    int N = input.numel();
    double denom = 1.0 / std::max(0.0, static_cast<double>(N - correction));
    Tensor m = mean(input).expand(input.shape());
    Tensor result = denom * sum(pow(input - m, 2));
    return keep_dim ? result : result.reshape({1});
}
auto Tensor::var(int dim, bool keep_dim, int correction) const -> Tensor {
    CHECK_VALID_TENSOR(*this);
    CHECK_VALID_INDEX_EXCLUSIVE(dim, shape_.ndim());
    return tinytensor::var(*this, dim, keep_dim, correction);
}
auto Tensor::var(const std::vector<int> &dims, bool keep_dim, int correction) const -> Tensor {
    CHECK_VALID_TENSOR(*this);
    return tinytensor::var(*this, dims, keep_dim, correction);
}
auto Tensor::var(bool keep_dim, int correction) const -> Tensor {
    CHECK_VALID_TENSOR(*this);
    return tinytensor::var(*this, keep_dim, correction);
}

// ------------------------------------------------
// Unary operations
// ------------------------------------------------

namespace {
template <common::unary::UnaryOpT Op>
auto unary_conversion_required(const Tensor &tensor) -> bool {
    return DISPATCH_ALL_TYPES(tensor.dtype(), "tensor::unary_conversion_required", [&]() {
        return !std::is_same_v<scalar_t, typename common::unary::Result<scalar_t, Op>::type>;
    });
}
}    // namespace

// NOLINTNEXTLINE(*-macro-usage)
#define DECLARE_BOOL_UNARY_OP(FUNC)                             \
    auto FUNC(const Tensor &tensor) -> Tensor {                 \
        CHECK_VALID_TENSOR(tensor);                             \
        if (tensor.dtype() != kBool) {                          \
            TT_EXCEPTION("Only boolean tensors are supported"); \
        }                                                       \
        return get_backend(tensor.device())->FUNC(tensor);      \
    }
// NOLINTNEXTLINE(*-macro-usage)
#define DECLARE_UNARY_OP(FUNC)                             \
    auto FUNC(const Tensor &tensor) -> Tensor {            \
        CHECK_VALID_TENSOR(tensor);                        \
        DISABLE_BOOL(tensor, FUNC);                        \
        return get_backend(tensor.device())->FUNC(tensor); \
    }
// NOLINTNEXTLINE(*-macro-usage)
#define DECLARE_UNARY_AUTOGRAD_OP(FUNC, AUTOGRAD_NAME) \
    auto FUNC(const Tensor &tensor) -> Tensor {        \
        CHECK_VALID_TENSOR(tensor);                    \
        DISABLE_BOOL(tensor, FUNC);                    \
        return autograd::AUTOGRAD_NAME::apply(tensor); \
    }
// NOLINTNEXTLINE(*-macro-usage)
#define DECLARE_BOOL_UNARY_INPLACE_OP(FUNC)                     \
    auto Tensor::FUNC##_()->Tensor & {                          \
        CHECK_VALID_TENSOR(*this);                              \
        CHECK_INPLACE_AUTOGRAD(*this);                          \
        if (dtype() != kBool) {                                 \
            TT_EXCEPTION("Only boolean tensors are supported"); \
        }                                                       \
        get_backend(this->device())->FUNC##_(*this);            \
        ++version_count_;                                       \
        return *this;                                           \
    }
// NOLINTNEXTLINE(*-macro-usage)
#define DECLARE_UNARY_INPLACE_OP(FUNC)                                         \
    auto Tensor::FUNC##_()->Tensor & {                                         \
        CHECK_VALID_TENSOR(*this);                                             \
        CHECK_INPLACE_AUTOGRAD(*this);                                         \
        DISABLE_BOOL(*this, FUNC);                                             \
        if (unary_conversion_required<common::unary::UnaryOpT::FUNC>(*this)) { \
            TT_EXCEPTION("Inplace abs requires dtype conversion.");            \
        }                                                                      \
        get_backend(this->device())->FUNC##_(*this);                           \
        ++version_count_;                                                      \
        return *this;                                                          \
    }
// NOLINTNEXTLINE(*-macro-usage)
#define DECLARE_BOOL_UNARY_OPS(FUNC) \
    DECLARE_BOOL_UNARY_OP(FUNC);     \
    DECLARE_BOOL_UNARY_INPLACE_OP(FUNC);
// NOLINTNEXTLINE(*-macro-usage)
#define DECLARE_UNARY_OPS(FUNC) \
    DECLARE_UNARY_OP(FUNC);     \
    DECLARE_UNARY_INPLACE_OP(FUNC);
// NOLINTNEXTLINE(*-macro-usage)
#define DECLARE_UNARY_AUTOGRAD_OPS(FUNC, AUTOGRAD_NAME) \
    DECLARE_UNARY_AUTOGRAD_OP(FUNC, AUTOGRAD_NAME);     \
    DECLARE_UNARY_INPLACE_OP(FUNC);

// Regular and inplace versions
DECLARE_UNARY_AUTOGRAD_OPS(abs, TensorAbs);
DECLARE_UNARY_AUTOGRAD_OPS(negate, TensorNegate);
DECLARE_UNARY_AUTOGRAD_OPS(sign, TensorSign);
DECLARE_UNARY_AUTOGRAD_OPS(log, TensorLog);
DECLARE_UNARY_AUTOGRAD_OPS(log10, TensorLog10);
DECLARE_UNARY_AUTOGRAD_OPS(log2, TensorLog2);
DECLARE_UNARY_AUTOGRAD_OPS(log1p, TensorLog1p);
DECLARE_UNARY_AUTOGRAD_OPS(exp, TensorExp);
DECLARE_UNARY_AUTOGRAD_OPS(exp2, TensorExp2);
DECLARE_UNARY_AUTOGRAD_OPS(expm1, TensorExpm1);
DECLARE_UNARY_AUTOGRAD_OPS(sqrt, TensorSqrt);
DECLARE_UNARY_AUTOGRAD_OPS(sin, TensorSin);
DECLARE_UNARY_AUTOGRAD_OPS(cos, TensorCos);
DECLARE_UNARY_AUTOGRAD_OPS(tan, TensorTan);
DECLARE_UNARY_AUTOGRAD_OPS(asin, TensorASin);
DECLARE_UNARY_AUTOGRAD_OPS(acos, TensorACos);
DECLARE_UNARY_AUTOGRAD_OPS(atan, TensorATan);
DECLARE_UNARY_AUTOGRAD_OPS(sinh, TensorSinh);
DECLARE_UNARY_AUTOGRAD_OPS(cosh, TensorCosh);
DECLARE_UNARY_AUTOGRAD_OPS(tanh, TensorTanh);
DECLARE_UNARY_AUTOGRAD_OPS(asinh, TensorASinh);
DECLARE_UNARY_AUTOGRAD_OPS(acosh, TensorACosh);
DECLARE_UNARY_AUTOGRAD_OPS(atanh, TensorATanh);
DECLARE_UNARY_AUTOGRAD_OPS(erf, TensorErf);
DECLARE_UNARY_AUTOGRAD_OPS(erfc, TensorErfc);
DECLARE_UNARY_AUTOGRAD_OPS(tgamma, TensorTGamma);
DECLARE_UNARY_AUTOGRAD_OPS(lgamma, TensorLGamma);
DECLARE_BOOL_UNARY_OPS(logical_not);
DECLARE_UNARY_OPS(digamma);
DECLARE_UNARY_OPS(ceil);
DECLARE_UNARY_OPS(floor);
DECLARE_UNARY_OPS(round);

// These dont have inplace versions
DECLARE_UNARY_OP(isinf);
DECLARE_UNARY_OP(isnan);
DECLARE_UNARY_OP(isfinite);

// Non-parameter activation functions
DECLARE_UNARY_AUTOGRAD_OPS(sigmoid, TensorSigmoid);
DECLARE_UNARY_AUTOGRAD_OPS(log_sigmoid, TensorLogSigmoid);
DECLARE_UNARY_AUTOGRAD_OPS(hardsigmoid, TensorHardSigmoid);
DECLARE_UNARY_AUTOGRAD_OPS(relu, TensorRelu);
DECLARE_UNARY_AUTOGRAD_OPS(relu6, TensorRelu6);
DECLARE_UNARY_AUTOGRAD_OPS(selu, TensorSelu);
DECLARE_UNARY_AUTOGRAD_OPS(silu, TensorSilu);
DECLARE_UNARY_AUTOGRAD_OPS(softsign, TensorSoftsign);

#undef DECLARE_BOOL_UNARY_OPS
#undef DECLARE_UNARY_OPS
#undef DECLARE_UNARY_AUTOGRAD_OPS
#undef DECLARE_BOOL_UNARY_INPLACE_OP
#undef DECLARE_UNARY_INPLACE_OP
#undef DECLARE_BOOL_UNARY_OP
#undef DECLARE_UNARY_OP
#undef DECLARE_UNARY_AUTOGRAD_OP

// Special unary ops
auto operator-(const Tensor &tensor) -> Tensor {
    return negate(tensor);
}
auto operator!(const Tensor &tensor) -> Tensor {
    return logical_not(tensor);
}

// ------------------------------------------------
// Activation Functions
// ------------------------------------------------

auto softplus(const Tensor &tensor, double beta, double threshold) -> Tensor {
    CHECK_VALID_TENSOR(tensor);
    DISABLE_BOOL(tensor, softplus);
    return autograd::TensorSoftplus::apply(tensor, beta, threshold);
}
auto Tensor::softplus_(double beta, double threshold) -> Tensor & {
    CHECK_VALID_TENSOR(*this);
    CHECK_INPLACE_AUTOGRAD(*this);
    DISABLE_BOOL(*this, softplus_);
    if (unary_conversion_required<common::unary::UnaryOpT::softplus>(*this)) {
        TT_EXCEPTION("Inplace softplus requires dtype conversion.");
    }
    get_backend(this->device())->softplus_(*this, beta, threshold);
    ++version_count_;
    return *this;
}

auto leaky_relu(const Tensor &tensor, double negative_slope) -> Tensor {
    CHECK_VALID_TENSOR(tensor);
    DISABLE_BOOL(tensor, leaky_relu);
    return autograd::TensorLeakyRelu::apply(tensor, negative_slope);
}
auto Tensor::leaky_relu_(double negative_slope) -> Tensor & {
    CHECK_VALID_TENSOR(*this);
    CHECK_INPLACE_AUTOGRAD(*this);
    DISABLE_BOOL(*this, leaky_relu_);
    if (unary_conversion_required<common::unary::UnaryOpT::leaky_relu>(*this)) {
        TT_EXCEPTION("Inplace leaky_relu requires dtype conversion.");
    }
    get_backend(this->device())->leaky_relu_(*this, negative_slope);
    ++version_count_;
    return *this;
}

auto elu(const Tensor &tensor, double alpha) -> Tensor {
    CHECK_VALID_TENSOR(tensor);
    DISABLE_BOOL(tensor, elu);
    return autograd::TensorElu::apply(tensor, alpha);
}
auto Tensor::elu_(double alpha) -> Tensor & {
    CHECK_VALID_TENSOR(*this);
    CHECK_INPLACE_AUTOGRAD(*this);
    DISABLE_BOOL(*this, elu_);
    if (unary_conversion_required<common::unary::UnaryOpT::elu>(*this)) {
        TT_EXCEPTION("Inplace elu requires dtype conversion.");
    }
    get_backend(this->device())->elu_(*this, alpha);
    ++version_count_;
    return *this;
}

auto hardtanh(const Tensor &tensor, double min, double max) -> Tensor {
    CHECK_VALID_TENSOR(tensor);
    DISABLE_BOOL(tensor, hardtanh);
    return autograd::TensorHardtanh::apply(tensor, min, max);
}
auto Tensor::hardtanh_(double min, double max) -> Tensor & {
    CHECK_VALID_TENSOR(*this);
    CHECK_INPLACE_AUTOGRAD(*this);
    DISABLE_BOOL(*this, hardtanh_);
    if (unary_conversion_required<common::unary::UnaryOpT::hardtanh>(*this)) {
        TT_EXCEPTION("Inplace hardtanh requires dtype conversion.");
    }
    get_backend(this->device())->hardtanh_(*this, min, max);
    ++version_count_;
    return *this;
}

auto softmax(const Tensor &tensor, int dim) -> Tensor {
    CHECK_VALID_TENSOR(tensor);
    DISABLE_BOOL(tensor, softmax);
    dim = indexing_exclusive(dim, tensor.shape());
    return autograd::TensorSoftmax::apply(tensor, dim);
}
auto Tensor::softmax_(int dim) -> Tensor & {
    CHECK_VALID_TENSOR(*this);
    CHECK_INPLACE_AUTOGRAD(*this);
    DISABLE_BOOL(*this, softmax_);
    if (!is_float_dtype(scalar_type_)) {
        TT_EXCEPTION("Inplace softmax requires dtype conversion.");
    }
    CHECK_VALID_INDEX_EXCLUSIVE(dim, shape_.ndim());
    dim = indexing_exclusive(dim, shape_);
    get_backend(this->device())->softmax_(*this, dim);
    ++version_count_;
    return *this;
}

auto log_softmax(const Tensor &tensor, int dim) -> Tensor {
    CHECK_VALID_TENSOR(tensor);
    DISABLE_BOOL(tensor, softmax);
    dim = indexing_exclusive(dim, tensor.shape());
    return autograd::TensorLogSoftmax::apply(tensor, dim);
}
auto Tensor::log_softmax_(int dim) -> Tensor & {
    CHECK_VALID_TENSOR(*this);
    CHECK_INPLACE_AUTOGRAD(*this);
    DISABLE_BOOL(*this, softmax_);
    if (!is_float_dtype(scalar_type_)) {
        TT_EXCEPTION("Inplace log_softmax requires dtype conversion.");
    }
    CHECK_VALID_INDEX_EXCLUSIVE(dim, shape_.ndim());
    dim = indexing_exclusive(dim, shape_);
    get_backend(this->device())->log_softmax_(*this, dim);
    ++version_count_;
    return *this;
}

// ------------------------------------------------
// Util/misc
// ------------------------------------------------

auto where(const Tensor &cond, Scalar lhs, Scalar rhs) -> Tensor {
    return where(
        cond,
        Tensor(lhs, cond.device()).expand(cond.shape()),
        Tensor(rhs, cond.device()).expand(cond.shape())
    );
}

auto where(const Tensor &cond, const Tensor &lhs, Scalar rhs) -> Tensor {
    return where(cond, lhs, Tensor(rhs.to(lhs.dtype()), cond.device()).expand(cond.shape()));
}

auto where(const Tensor &cond, Scalar lhs, const Tensor &rhs) -> Tensor {
    return where(cond, Tensor(lhs.to(rhs.dtype()), cond.device()).expand(cond.shape()), rhs);
}

auto where(const Tensor &cond, const Tensor &lhs, const Tensor &rhs) -> Tensor {
    CHECK_VALID_TENSOR(cond);
    CHECK_VALID_TENSOR(lhs);
    CHECK_VALID_TENSOR(rhs);
    CHECK_SAME_DEVICE(cond, lhs, rhs);
    CHECK_SAME_SHAPE(cond, lhs, rhs);
    CHECK_SAME_TYPE(lhs, rhs);
    if (cond.dtype() != kBool) {
        TT_EXCEPTION(std::format("Condition type needs to be kBool, given {:s}", cond.dtype()));
    }
    return autograd::TensorWhere::apply(cond, lhs, rhs);
}

namespace {
auto compute_close(const Tensor &lhs, const Tensor &rhs, double rtol, double atol) -> Tensor {
    const auto allowed_error = abs(rhs * rtol) + atol;
    const auto actual_error = abs(lhs - rhs);
    return actual_error < allowed_error;
}
}    // namespace

auto isclose(const Tensor &lhs, const Tensor &rhs, const CloseOptions &options) -> Tensor {
    CHECK_VALID_TENSOR(lhs);
    CHECK_VALID_TENSOR(rhs);
    CHECK_SAME_DEVICE(lhs, rhs);
    CHECK_SAME_SHAPE(lhs, rhs);
    if (options.atol() < 0) {
        TT_EXCEPTION("atol should be greater than or equal to 0.");
    }
    // tolerance of 0 is just equality
    // Absolute error: |lhs - rhs| < atol
    // Relative error: |lhs - rhs| < |rtol * rhs|
    auto result = (options.atol() == 0) ? (lhs == rhs)
                                        : compute_close(
                                              lhs.to(to_scalar<double>::type),
                                              rhs.to(to_scalar<double>::type),
                                              options.rtol(),
                                              options.atol()
                                          );
    // Treat two NaNs as equal
    if (options.equal_nan()) {
        const auto nan_mask = isnan(lhs) && isnan(rhs);
        const auto inf_mask = isinf(lhs) && isinf(rhs);
        result = result || nan_mask || inf_mask;
    }
    return result;
}
auto allclose(const Tensor &input, const Tensor &other, const CloseOptions &options) -> bool {
    CHECK_VALID_TENSOR(input);
    CHECK_VALID_TENSOR(other);
    // Non-matching tensor properties
    if (input.device() != other.device() || input.dtype() != other.dtype() || input.shape() != other.shape()) {
        return false;
    }
    // Handle empty tensors
    if (input.numel() == 0 && other.numel() == 0) {
        return true;
    }
    return all(isclose(input, other, options));
}

auto ClampOptions::min_to(ScalarType dtype) const -> std::optional<Scalar> {
    if (_min) {
        return DISPATCH_ALL_TYPES(dtype, "ClampOptions::get_min", [&]() {
            return Scalar(_min.value().to<scalar_t>());
        });
    }
    return _min;
}
auto ClampOptions::max_to(ScalarType dtype) const -> std::optional<Scalar> {
    if (_max) {
        return DISPATCH_ALL_TYPES(dtype, "ClampOptions::get_max", [&]() {
            return Scalar(_max.value().to<scalar_t>());
        });
    }
    return _max;
}

auto Tensor::clamp_(const ClampOptions &options) -> Tensor & {
    CHECK_VALID_TENSOR(*this);
    CHECK_INPLACE_AUTOGRAD(*this);
    DISABLE_BOOL(*this, clamp);
    DISPATCH_ALL_TYPES(scalar_type_, "clamp", [&]() {
        Scalar min_scalar = options.min_to(scalar_type_).value_or(Scalar(std::numeric_limits<scalar_t>::lowest()));
        Scalar max_scalar = options.max_to(scalar_type_).value_or(Scalar(std::numeric_limits<scalar_t>::max()));
        const Tensor min_tensor = Tensor(min_scalar, device_).expand(shape_);
        const Tensor max_tensor = Tensor(max_scalar, device_).expand(shape_);
        get_backend(device_)->clamp_(*this, min_tensor, max_tensor);
    });
    return *this;
}
auto Tensor::clamp_(const Tensor &min, const Tensor &max) -> Tensor & {
    CHECK_VALID_TENSOR(min);
    CHECK_VALID_TENSOR(max);
    CHECK_SAME_DEVICE(*this, min, max);
    CHECK_SAME_SHAPE(*this, min, max);
    CHECK_SAME_TYPE(*this, min, max);
    CHECK_INPLACE_AUTOGRAD(*this);
    DISABLE_BOOL(*this, clamp);
    get_backend(device_)->clamp_(*this, min, max);
    return *this;
}

auto Tensor::clamp(const ClampOptions &options) const -> Tensor {
    CHECK_VALID_TENSOR(*this);
    DISABLE_BOOL(*this, clamp);
    const auto dtype = scalar_type_;
    return DISPATCH_ALL_TYPES(dtype, "clamp", [&]() {
        Scalar min_scalar = options.min_to(dtype).value_or(Scalar(std::numeric_limits<scalar_t>::lowest()));
        Scalar max_scalar = options.max_to(dtype).value_or(Scalar(std::numeric_limits<scalar_t>::max()));
        const Tensor min_tensor = Tensor(min_scalar, device_).expand(shape_);
        const Tensor max_tensor = Tensor(max_scalar, device_).expand(shape_);
        return autograd::TensorClamp::apply(*this, min_tensor, max_tensor);
    });
}
auto clamp(const Tensor &input, const ClampOptions &options) -> Tensor {
    CHECK_VALID_TENSOR(input);
    return input.clamp(options);
}

auto Tensor::clamp(const Tensor &min, const Tensor &max) const -> Tensor {
    CHECK_VALID_TENSOR(*this);
    CHECK_VALID_TENSOR(min);
    CHECK_VALID_TENSOR(max);
    CHECK_SAME_DEVICE(*this, min, max);
    CHECK_SAME_SHAPE(*this, min, max);
    CHECK_SAME_TYPE(*this, min, max);
    DISABLE_BOOL(*this, clamp);
    return autograd::TensorClamp::apply(*this, min, max);
}
auto clamp(const Tensor &input, const Tensor &min, const Tensor &max) -> Tensor {
    CHECK_VALID_TENSOR(input);
    return input.clamp(min, max);
}

auto CloseOptions::rtol(double tol) -> CloseOptions & {
    _rtol = tol;
    return *this;
}
auto CloseOptions::atol(double tol) -> CloseOptions & {
    _atol = tol;
    return *this;
}
auto CloseOptions::equal_nan() -> CloseOptions & {
    _equal_nan = true;
    return *this;
}
[[nodiscard]] auto
    conv2d(const Tensor &input, const Tensor &weight, const std::optional<Tensor> &bias, int stride, int padding)
        -> Tensor {
    CHECK_VALID_TENSOR(input);
    CHECK_VALID_TENSOR(weight);
    DISABLE_BOOL(input, conv2d);
    if (input.device() != weight.device()) {
        TT_EXCEPTION(
            std::format("Device of input({:s}) does not match device of weight({:s}).", input.device(), weight.device())
        );
    }
    if (input.dim() != 4) {
        TT_EXCEPTION(std::format("Input shape {:s} expected to have 4 dimensions.", input.shape().to_string()));
    }
    if (weight.dim() != 4) {
        TT_EXCEPTION(std::format("Weight shape {:s} expected to have 4 dimensions.", weight.shape().to_string()));
    }
    if (input.size(1) != weight.size(1)) {
        TT_EXCEPTION(
            std::format(
                "Size at dim 1 (input channels) of input ({:d}) and weight ({:d}) expected to be the same.",
                input.size(1),
                weight.size(1)
            )
        );
    }
    if (weight.size(2) != weight.size(3)) {
        TT_EXCEPTION(
            std::format(
                "No support for non-square kernel size. Weight sizes at dim 2 ({:d}) and 3 ({:d}) should match.",
                weight.size(2),
                weight.size(3)
            )
        );
    }
    std::optional<Tensor> expanded_bias;
    if (bias) {
        CHECK_VALID_TENSOR(*bias);
        CHECK_SAME_TYPE(input, weight, *bias);
        if (bias->device() != input.device()) {
            TT_EXCEPTION(
                std::format(
                    "Device of bias({:s}) does not match device of input({:s}).",
                    bias->device(),
                    input.device()
                )
            );
        }
        if (bias->dim() != 1 || bias->size(0) != weight.size(0)) {
            TT_EXCEPTION(
                std::format(
                    "Bias of shape {:s} must equal output channels (:d)",
                    bias->shape().to_string(),
                    weight.size(0)
                )
            );
        }
    } else {
        CHECK_SAME_TYPE(input, weight);
    }
    if (stride <= 0) {
        TT_EXCEPTION(std::format("Stide of {:d} is expected to be positive.", stride));
    }
    if (padding < 0) {
        TT_EXCEPTION(std::format("Padding of {:d} is expected to be non-negative.", stride));
    }
    if (input.size(2) + padding < weight.size(2) || input.size(3) + padding < weight.size(2)) {
        TT_EXCEPTION(
            std::format(
                "Kernel size {:d} cannot be greater than input size ({:d}, {:d}) + padding {:d}",
                weight.size(2),
                input.size(2),
                input.size(3),
                padding
            )
        );
    }
    return autograd::TensorConv2d::apply(input, weight, bias, stride, padding);
}

// NOLINTNEXTLINE(*-macro-usage)
#define DECLARE_POOL2D_UTIL(FUNC, AUTOGRAD_NAME)                                                                     \
    auto FUNC(const Tensor &input, int kernel_size, int stride, int padding) -> Tensor {                             \
        CHECK_VALID_TENSOR(input);                                                                                   \
        DISABLE_BOOL(input, FUNC);                                                                                   \
        if (input.dim() != 4) {                                                                                      \
            TT_EXCEPTION(std::format("Input shape {:s} expected to have 4 dimensions.", input.shape().to_string())); \
        }                                                                                                            \
        if (stride <= 0) {                                                                                           \
            TT_EXCEPTION(std::format("Stide of {:d} is expected to be positive.", stride));                          \
        }                                                                                                            \
        if (padding < 0) {                                                                                           \
            TT_EXCEPTION(std::format("Padding of {:d} is expected to be non-negative.", stride));                    \
        }                                                                                                            \
        if (input.size(2) + padding < kernel_size || input.size(3) + padding < kernel_size) {                        \
            TT_EXCEPTION(                                                                                            \
                std::format(                                                                                         \
                    "Kernel size {:d} cannot be greater than input size ({:d}, {:d}) + padding {:d}",                \
                    kernel_size,                                                                                     \
                    input.size(2),                                                                                   \
                    input.size(3),                                                                                   \
                    padding                                                                                          \
                )                                                                                                    \
            );                                                                                                       \
        }                                                                                                            \
        return autograd::AUTOGRAD_NAME::apply(input, kernel_size, stride, padding);                                  \
    }
DECLARE_POOL2D_UTIL(max_pool2d, TensorMaxPool2d);
DECLARE_POOL2D_UTIL(min_pool2d, TensorMinPool2d);
DECLARE_POOL2D_UTIL(avg_pool2d, TensorAvgPool2d);
#undef DECLARE_POOL2D_UTIL

auto embedding(const Tensor &input, const Tensor &weight) -> Tensor {
    if (!is_integral_dtype(input.dtype())) {
        TT_EXCEPTION(std::format("Expected input to be integral dtype, given {:s}", input.dtype()));
    }
    if (weight.dim() != 2) {
        TT_EXCEPTION(std::format("Expected weight to have 2 dimensions, given shape {:s}", weight.shape()));
    }

    auto result_shape = input.shape();
    result_shape.insert(weight.size(1));
    return index_select(weight, input, 0).reshape(result_shape);
}

auto current_memory_allocated(const Device &device) -> uint64_t {
    return get_backend(device)->current_memory_allocated(device.id);
}
auto total_memory_allocated(const Device &device) -> uint64_t {
    return get_backend(device)->total_memory_allocated(device.id);
}

auto make_dot(const Tensor &tensor) -> std::string {
    std::stringstream ss;
    ss << "digraph ComputationGraph { ";
    TensorList dag = autograd::build_dag(tensor);

    std::unordered_map<uintptr_t, int> node_ids;
    for (const auto &t : std::ranges::reverse_view(dag)) {
        int id = static_cast<int>(node_ids.size());
        node_ids[reinterpret_cast<uintptr_t>(t.ctx_.get())] = id;    // NOLINT(*-reinterpret-cast)
        if (t.is_leaf()) {
            ss << std::format("{:d}[label=\"{:s}\",shape=ellipse]; ", id, t.shape());
        } else {
            ss << std::format("{:d}[label=\"{:s}\\n{:s}\",shape=box]; ", id, t.ctx_->grad_func_name, t.shape());
        }
    }

    for (const auto &t : std::ranges::reverse_view(dag)) {
        for (const auto &par : t.ctx_->parents) {
            const auto ptr_par = reinterpret_cast<uintptr_t>(par.ctx_.get());    // NOLINT(*-reinterpret-cast)
            const auto ptr_t = reinterpret_cast<uintptr_t>(t.ctx_.get());        // NOLINT(*-reinterpret-cast)
            ss << std::format("{:d} -> {:d}; ", node_ids.at(ptr_par), node_ids.at(ptr_t));
        }
    }
    const auto ptr_back = reinterpret_cast<uintptr_t>(dag[-1].ctx_.get());    // NOLINT(*-reinterpret-cast)
    ss << std::format("{:d} -> \"{:s}\"; ", node_ids.at(ptr_back), tensor.shape());
    ss << "}";
    return ss.str();
}

}    // namespace tinytensor
