// tensor.h
// Multi-dimensional tensor class

#ifndef TINYTENSOR_TENSOR_H_
#define TINYTENSOR_TENSOR_H_

#include <tt/device.h>
#include <tt/exception.h>
#include <tt/export.h>
#include <tt/index.h>
#include <tt/random.h>
#include <tt/scalar.h>
#include <tt/shape.h>

#include <cstddef>
#include <cstdint>
#include <format>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace tinytensor {

// Forward declarations
class Tensor;
class StorageBase;
template <typename T>
class CheckedVec;

using TensorList = CheckedVec<Tensor>;

namespace autograd {
struct SharedGrad;

template <typename T>
struct TensorFunction;

// Hook to apply on the computed gradient
using GradHook = std::function<void(Tensor &grad)>;

auto build_dag(const Tensor &tensor) -> TensorList;
void backward(Tensor &tensor, const Tensor &grad, bool retain_graph);
void calc_grad_input(Tensor &tensor, bool retain_graph);
void add_grad(Tensor &tensor, const Tensor &grad);
}    // namespace autograd

// Auxiliary options for Tensors, enables chaining of setting options and
// storing as a variable for use with multiple Tensor creations
class TINYTENSOR_EXPORT TensorOptions {
public:
    TensorOptions() = default;

    // Converting constructors
    explicit TensorOptions(Device device)
        : device_(device) {}
    explicit TensorOptions(ScalarType dtype)
        : dtype_(dtype) {}
    explicit TensorOptions(bool requires_grad)
        : requires_grad_(requires_grad) {}

    auto dtype(ScalarType dtype) -> TensorOptions & {
        dtype_ = dtype;
        return *this;
    }
    [[nodiscard]] auto dtype() const -> ScalarType {
        return dtype_;
    }
    auto device(Device device) -> TensorOptions & {
        device_ = device;
        return *this;
    }
    [[nodiscard]] auto device() const -> Device {
        return device_;
    }
    auto requires_grad(bool requires_grad) -> TensorOptions & {
        requires_grad_ = requires_grad;
        return *this;
    }
    [[nodiscard]] auto requires_grad() const -> bool {
        return requires_grad_;
    }

private:
    ScalarType dtype_ = kF32;
    Device device_ = kCPU;
    bool requires_grad_ = false;
};

// Auxiliary options for clamp
class TINYTENSOR_EXPORT ClampOptions {
public:
    template <IsScalarType T>
    auto min(T min_value) -> ClampOptions & {
        _min = Scalar(min_value);
        return *this;
    }
    template <IsScalarType T>
    auto max(T max_value) -> ClampOptions & {
        _max = Scalar(max_value);
        return *this;
    }
    [[nodiscard]] inline auto min() const -> std::optional<Scalar> {
        return _min;
    }
    [[nodiscard]] inline auto max() const -> std::optional<Scalar> {
        return _max;
    }
    [[nodiscard]] auto min_to(ScalarType dtype) const -> std::optional<Scalar>;
    [[nodiscard]] auto max_to(ScalarType dtype) const -> std::optional<Scalar>;

private:
    std::optional<Scalar> _min;
    std::optional<Scalar> _max;
};

// Multi-Dimensional Tensor class
class TINYTENSOR_EXPORT Tensor {
public:
    /**
     * Construct from a vector
     * @param data The vector data
     * @param shape The shape to represent the data
     * @param device The device for the Tensor
     * @param requires_grad Flag if autograd operations should be recorded
     */
    template <typename T>
        requires(IsScalarType<T> || std::is_same_v<T, bool>)
    Tensor(const std::vector<T> &data, Shape shape, Device device, bool requires_grad = false);

    /**
     * Construct from a vector
     * @param data The vector data
     * @param shape The shape to represent the data
     * @param device The device for the Tensor
     * @param requires_grad Flag if autograd operations should be recorded
     */
    template <typename T>
        requires(IsScalarType<T> || std::is_same_v<T, bool>)
    Tensor(std::vector<T> &&data, Shape shape, Device device, bool requires_grad = false);

    /**
     * Construct from an initializer_list
     * @param data The initializer list of data
     * @param shape The shape to represent the data
     * @param device The device for the Tensor
     * @param requires_grad Flag if autograd operations should be recorded
     */
    template <typename T>
        requires(IsScalarType<T> || std::is_same_v<T, bool>)
    Tensor(const std::initializer_list<T> &data, Shape shape, Device device, bool requires_grad = false)
        : Tensor(std::vector<T>(data), shape, device, requires_grad) {}

    /**
     * Construct from a vector, with inferred flattened shape
     * @param data The vector data
     * @param device The device for the Tensor
     * @param requires_grad Flag if autograd operations should be recorded
     */
    template <typename T>
        requires(IsScalarType<T> || std::is_same_v<T, bool>)
    Tensor(const std::vector<T> &data, Device device, bool requires_grad = false)
        : Tensor(data, {static_cast<int>(data.size())}, device, requires_grad) {}

    /**
     * Construct from a vector, with inferred flattened shape
     * @param data The vector data
     * @param device The device for the Tensor
     * @param requires_grad Flag if autograd operations should be recorded
     */
    template <typename T>
        requires(IsScalarType<T> || std::is_same_v<T, bool>)
    Tensor(std::vector<T> &&data, Device device, bool requires_grad = false);

    /**
     * Construct from an initializer list, with inferred flattened shape
     * @param data The initializer list of data
     * @param device The device for the Tensor
     * @param requires_grad Flag if autograd operations should be recorded
     */
    template <typename T>
        requires(IsScalarType<T> || std::is_same_v<T, bool>)
    Tensor(const std::initializer_list<T> &data, Device device, bool requires_grad = false)
        : Tensor(std::vector<T>(data), device, requires_grad) {}

    /**
     * Construct from a Scalar value
     * @param scalar The scalar value
     * @param device The device for the Tensor
     */
    Tensor(Scalar scalar, Device device, bool requires_grad = false);

    // Internal constructor for backends
    Tensor(std::shared_ptr<StorageBase> storage, ScalarType dtype, Shape shape, Device device);

    /**
     * Set the internal data from another tensor
     * @note This is useful to change the underlying state while keeping external
     * references valid
     * @param other The other Tensor to set from
     */
    void set_from(const Tensor &other);

    ~Tensor() = default;

    /**
     * Searialize the current tensor into the given stream
     */
    auto serialize() const -> std::vector<char>;

    /**
     * Deserialize the given data to replace the contents of the current tensor
     * @note This change will NOT reflect to other references/views of the current
     * tensor
     * @note The underlying dtype of the tensor may also change, depending on the
     * contents of the serialized data, but the underlying device will NOT change
     * @param serialized_data The serialized tensor
     */
    void deserialize(const std::vector<char> &serialized_data);

    /**
     * Check if Tensor has underlying storage
     * @note this is used for debugging
     */
    [[nodiscard]] auto has_storage() const -> bool;

    /**
     * Get the dtype of the Tensor
     */
    [[nodiscard]] auto dtype() const -> ScalarType;

    /**
     * Get the device of the Tensor
     */
    [[nodiscard]] auto device() const -> Device;

    /**
     * Get the storage offset of the Tensor
     */
    [[nodiscard]] auto offset() const -> int;

    /**
     * Get the shape of the Tensor
     */
    [[nodiscard]] auto shape() const -> const Shape &;

    /**
     * Get the stride of the Tensor
     */
    [[nodiscard]] auto stride() const -> const Shape &;

    /**
     * Get the total number of elements the Tensors contains
     */
    [[nodiscard]] auto numel() const -> int;

    /**
     * Get the number of dimensions the Tensor represents
     */
    [[nodiscard]] auto dim() const -> int;

    /**
     * Get the size of a dimension
     * @param dim The dimension
     */
    [[nodiscard]] auto size(int dim) const -> int;

    /**
     * Check if the underlying storage of the Tensor represents is contiguous
     */
    [[nodiscard]] auto is_contiguous() const -> bool;

    /**
     * Return a copy with the same values but contiguous storage
     * @note If the Tensor is already contiguous, a shallow copy of self is
     * performed with the resulting Tensor containing the same underlying storage
     */
    [[nodiscard]] auto contiguous() const -> Tensor;

    /**
     * Get the underlying storage derived class as a result of a cast
     * @note Used in backend methods to recover type from the pointer
     */
    template <typename T>
    [[nodiscard]] auto get_storage() const -> T & {
        return *static_cast<T *>(storage_.get());
    }

    /**
     * Gets the first value in the Tensor type erased into a Scalar
     */
    [[nodiscard]] auto item() const -> Scalar;

    /**
     * Gets the first value in the Tensor casted to the templated type
     */
    template <typename T>
        requires(IsScalarType<T> || std::is_same_v<T, bool>)
    [[nodiscard]] auto item() const -> T {
        return item().to<T>();
    }

    /**
     * Gets the first value in the Tensor casted to the templated scalar type
     */
    template <ScalarType T>
    [[nodiscard]] auto item() const -> to_ctype_t<T> {
        return item().to<std::conditional_t<T == kBool, bool, to_ctype_t<T>>>();
    }

    /**
     * Cast the underlying storage to the given scalar type
     * @note If the Tensor is already the given scalar type, a shallow copy of
     * self is performed with the resulting Tensor containing the same underlying
     * storage
     * @param dtype The scalar type to cast the storage to
     * @return The casted Tensor
     */
    [[nodiscard]] auto to(ScalarType dtype) const -> Tensor;

    /**
     * Create a copy of the underlying Tensor on the given device
     * @note If the device matches the current device, a shallow copy is made
     * @param device The device to copy the tensor to
     * @return The Tensor
     */
    [[nodiscard]] auto to(Device device) const -> Tensor;

    /**
     * Return a deep clone of the Tensor with same values but separate underlying
     * storage
     * @note This is an autograd-aware operation
     */
    [[nodiscard]] auto clone() const -> Tensor;

    /**
     * Return a new tensor that is removed from the computation graph
     * @note This does not share the underlying data, so inplace operations are
     * safe on the returned tensor
     */
    [[nodiscard]] auto detach() const -> Tensor;

    /**
     * Get address of the underlying pointer, helpful to check if tensor points to
     * the same storage after various operations
     */
    [[nodiscard]] auto data_ptr() const -> uintptr_t;

    /**
     * Get the version count of the Tensor
     * Inplace operations will increment the version count, with Tensors created
     * with version_count = 0
     * @TODO: This may not actually be needed as we do disable inplace ops on
     * tensors requiring grads
     */
    [[nodiscard]] auto version_count() const -> int;

    /**
     * Get the underlying values as a flattened vector
     * @note This will cast the underlying data element by element to the casted
     * type, irrespective of the underlying storage type
     */
    template <typename T>
        requires(IsScalarType<T> || std::is_same_v<T, bool>)
    [[nodiscard]] auto to_vec() const -> std::vector<T>;

    // Copy, shallow by default
    Tensor(const Tensor &rhs) = default;
    Tensor(Tensor &&rhs) = default;

    // Assignment

    /**
     * Assign all values of the Tensor to the given value
     */
    template <typename T>
        requires(IsScalarType<T> || std::is_same_v<T, bool>)
    auto operator=(T rhs) -> Tensor & {
        this->operator=(Scalar(rhs, scalar_type_));
        return *this;
    }

    /**
     * Assign all values of the Tensor to the given value
     */
    template <typename T>
        requires(IsScalarType<T> || std::is_same_v<T, bool>)
    auto fill_(T rhs) -> Tensor & {
        this->operator=(Scalar(rhs, scalar_type_));
        return *this;
    }

    /**
     * Assign all values of the Tensor to the given Scalar
     * @note Since this is an in-place operation, this will throw an exception on
     * tensors requiring grads
     */
    auto operator=(const Scalar &rhs) -> Tensor &;

    /**
     * Assign all values of the Tensor to the given Scalar
     * @note Since this is an in-place operation, this will throw an exception on
     * tensors requiring grads
     */
    auto fill_(const Scalar &rhs) -> Tensor &;

    /**
     * Assignment on lvalues behaves like a copy constructor
     */
    auto operator=(const Tensor &rhs) & -> Tensor &;
    auto operator=(Tensor &&rhs) & -> Tensor &;

    /**
     * Assignment on rvalues will assign the rhs data into the self view, to allow
     * indexing with assignment If RHS is a different type than self, it will be
     * casted to match the self scalar type
     * @note Since this is an in-place operation, this will throw an exception on
     * tensors requiring grads
     */
    auto operator=(const Tensor &rhs) && -> Tensor &;
    auto operator=(Tensor &&rhs) && -> Tensor &;

    /**
     * Element-wise inplace arithmetic
     * This will cast RHS to the type of LHS
     * @note Since this is an in-place operation, this will throw an exception on
     * tensors requiring grads
     */
    template <IsScalarType T>
    auto operator+=(T rhs) -> Tensor & {
        return this->operator+=(Scalar(rhs, scalar_type_));
    }
    auto operator+=(const Scalar &rhs) -> Tensor &;
    auto operator+=(const Tensor &rhs) -> Tensor &;

    template <IsScalarType T>
    auto operator-=(T rhs) -> Tensor & {
        return this->operator-=(Scalar(rhs, scalar_type_));
    }
    auto operator-=(const Scalar &rhs) -> Tensor &;
    auto operator-=(const Tensor &rhs) -> Tensor &;

    template <IsScalarType T>
    auto operator*=(T rhs) -> Tensor & {
        return this->operator*=(Scalar(rhs, scalar_type_));
    }
    auto operator*=(const Scalar &rhs) -> Tensor &;
    auto operator*=(const Tensor &rhs) -> Tensor &;

    template <IsScalarType T>
    auto operator/=(T rhs) -> Tensor & {
        return this->operator/=(Scalar(rhs, scalar_type_));
    }
    auto operator/=(const Scalar &rhs) -> Tensor &;
    auto operator/=(const Tensor &rhs) -> Tensor &;

    /**
     * Index tensor with a boolean mask
     * @note This returns a new Tensor with separate underlying storage
     * @note The result is a flat Tensor
     * @param mask The boolean mask
     * @return The indexed Tensor
     */
    [[nodiscard]] auto operator[](const Tensor &mask) const -> Tensor;

    /**
     * Index tensor with either an integer index or Slice
     * @note This will always return a view of the Tensor, sharing the underlying
     * storage
     * @note Indexing a 1D Tensor will return another 1D Tensor, unlike Torch
     * which returns scalar values
     * @index The Index (integer or Slice)
     * @return The indexed Tensor
     */
    [[nodiscard]] auto operator[](const indexing::Index &index) const -> Tensor;

    /**
     * Index tensor with a mixed sequence of integer indices or Slices
     * @note This will always return a view of the Tensor, sharing the underlying
     * storage
     * @note Indexing a 1D Tensor will return another 1D Tensor, unlike Torch
     * which returns scalar values
     * @indices The indices (integers and/or Slices)
     * @return The indexed Tensor
     */
    [[nodiscard]] auto operator[](const std::vector<indexing::Index> &indices) const -> Tensor;

    // ------------------------------------------------
    // Range based loop iterator support
    // ------------------------------------------------

    class Iterator {
    public:
        Iterator(Tensor &tensor, int idx)
            : tensor_(tensor), idx_(idx) {}

        auto operator!=(const Iterator &other) const -> bool;
        auto operator++() -> Iterator;
        auto operator*() const -> Tensor;

    private:
        Tensor &tensor_;    // NOLINT(*-ref-data-members)
        int idx_;
    };

    auto begin() -> Iterator;
    auto end() -> Iterator;

    // ------------------------------------------------
    // Autograd related operations
    // ------------------------------------------------

    /**
     * Check if gradient tracking is set for this Tensor
     * @return True if gradients are being tracked, false otherwise
     */
    [[nodiscard]] auto requires_grad() const -> bool;

    /**
     * Set the flag for requiring the Tensor's gradient to be computed
     * @note: Only floating point type Tensors support gradient tracking
     * @param set_grad True to track gradients for this Tensor, false otherwise
     */
    void set_requires_grad(bool set_grad);

    /**
     * Get the optionally stored gradient for the underlying Tensor
     * @return The gradient Tensor if exists, std::nullopt otherwise
     */
    [[nodiscard]] auto grad() const -> const std::optional<Tensor> &;

    /**
     * Clears the Tensor's stored gradient
     */
    void clear_grad();

    /**
     * Accumulate the given gradient to the Tensor's gradient
     * @param grad The grad to accumulate with
     */
    void add_grad(const Tensor &grad);

    /**
     * Register a hook to apply to gradient after being computed
     * @hook Function-like hook
     */
    void register_hook(const autograd::GradHook &hook);

    /**
     * Perform backward pass computing gradients starting from this Tensor
     * @param grad The gradient of the function with respect to the current
     * Tensor. If ommitted, it will default to ones_like(self)
     * @param retain_graph If false, will free the underlying computation graph
     * after performing backward pass
     */
    void backward(const std::optional<Tensor> &grad = {}, bool retain_graph = false);

    /**
     * Check if the current Tensor is a leaf
     * Tensors with requires_grad=False are leaf tensors
     * Tensors with requires_grad=True and are not results of autograd ops are
     * also leaf tensors
     */
    [[nodiscard]] auto is_leaf() const -> bool;

    // ------------------------------------------------
    // Shape Modification operators
    // ------------------------------------------------

    /**
     * Expands the Tensor to the given shape, which can include a larger number of
     * dimensions New dimensions are appended to the front, and any non-singleton
     * existing dimension must match the expanded-to dimension
     * @note: This will always return a view of the Tensor, sharing the underlying
     * storage
     * @note: This is equivalent to calling expand(tensor, shape)
     * @note: See
     * https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html
     * @param shape The shape to expand to
     * @return View of the expanded Tensor
     */
    [[nodiscard]] auto broadcast_to(const Shape &shape) const -> Tensor;

    /**
     * Expands the Tensor to the given shape, which can include a larger number of
     * dimensions New dimensions are appended to the front, and any non-singleton
     * existing dimension must match the expanded-to dimension
     * @note: This will always return a view of the Tensor, sharing the underlying
     * storage
     * @note: See
     * https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html
     * @param shape The shape to expand to
     * @return View of the expanded Tensor
     */
    [[nodiscard]] auto expand(const Shape &shape) const -> Tensor;

    /**
     * Removes a dimension of size one at a given dimension location to the Tensor
     * @note: This has the same view or copy properties as reshape
     * @note: If the given dimension to squeeze is not of size 1, the Tensor is
     * unchanged
     * @param dim The index to remove the singleton dimension
     * @return View of the Tensor with removed dimension
     */
    [[nodiscard]] auto squeeze(int dim) const -> Tensor;

    /**
     * Inserts a dimension of size one at a given dimension location to the Tensor
     * @note: This has the same view or copy properties as reshape
     * @param dim The index to insert the singleton dimension
     * @return View of the Tensor with inserted dimension
     */
    [[nodiscard]] auto unsqueeze(int dim) const -> Tensor;

    /**
     * Reshapes the Tensor to the specified shape, keeping the same underlying
     * data and number of elements
     * @note: This will return a view of the Tensor if possible (i.e. contiguous),
     * or a copy of the underlying storage
     * @param shape The shape of the resulting Tensor to reshape to
     * @return The reshaped Tensor
     */
    [[nodiscard]] auto reshape(const Shape &shape) const -> Tensor;

    /**
     * Flattens the Tensor into a one-dimensional Tensor, equivalent to
     * reshape({tensor.numel()}),
     * @note: If start_dim and end_dim are given, only dimensions between
     * start/end are flattened
     * @note: This has the same view or copy properties as reshape
     * @param start_dim Starting dim to flatten
     * @param end_dim Ending dim to flatten
     * @return View of flattened Tensor
     */
    [[nodiscard]] auto flatten(int start_dim = 0, int end_dim = -1) const -> Tensor;

    /**
     * Permutes the Tensor along its dimensions
     * @note: This will always return a view of the Tensor, sharing the underlying
     * storage
     * @param dims The permutation of the input tensor dimensions the new Tensor
     * should take
     * @return View of the permuted Tensor
     */
    [[nodiscard]] auto permute(const std::vector<int> &dims) const -> Tensor;

    // ------------------------------------------------
    // Inplace Unary operations
    // ------------------------------------------------

    /**
     * Performs element-wise absolute of the self Tensor
     * @return reference to self
     */
    auto abs_() -> Tensor &;

    /**
     * Performs element-wise negation of the self Tensor
     * @return reference to self
     */
    auto negate_() -> Tensor &;

    /**
     * Performs element-wise logical not on the self Tensor
     * @note The dtype of the self Tensor must be boolean
     * @note This function does not have autograd support
     * @return reference to self
     */
    auto logical_not_() -> Tensor &;

    /**
     * Performs element-wise sign of the self Tensor
     * @note see https://en.wikipedia.org/wiki/Sign_function
     * @return reference to self
     */
    auto sign_() -> Tensor &;

    /**
     * Performs element-wise natural logarithm of the self Tensor
     */
    auto log_() -> Tensor &;

    /**
     * Performs element-wise base 10 logarithm of the self Tensor
     * @return reference to self
     */
    auto log10_() -> Tensor &;

    /**
     * Performs element-wise base 2 logarithm of the self Tensor
     * @return reference to self
     */
    auto log2_() -> Tensor &;

    /**
     * Performs element-wise natural logarithm of one plus the self Tensor
     * @return reference to self
     */
    auto log1p_() -> Tensor &;

    /**
     * Performs element-wise exponential of the self Tensor
     * @return reference to self
     */
    auto exp_() -> Tensor &;

    /**
     * Performs element-wise base 2 exponential of the input Tensor
     * @return reference to self
     */
    auto exp2_() -> Tensor &;

    /**
     * Performs element-wise exp(x)-1 of the input Tensor
     * @return reference to self
     */
    auto expm1_() -> Tensor &;

    /**
     * Performs element-wise square root of the input Tensor
     * @return reference to self
     */
    auto sqrt_() -> Tensor &;

    /**
     * Performs element-wise sine of the input Tensor
     * @return reference to self
     */
    auto sin_() -> Tensor &;

    /**
     * Performs element-wise cosine of the input Tensor
     * @return reference to self
     */
    auto cos_() -> Tensor &;

    /**
     * Performs element-wise tangent of the input Tensor
     * @return reference to self
     */
    auto tan_() -> Tensor &;

    /**
     * Performs element-wise trigonometric inverse sine of the input Tensor
     * @return reference to self
     */
    auto asin_() -> Tensor &;

    /**
     * Performs element-wise trigonometric inverse cosine of the input Tensor
     * @return reference to self
     */
    auto acos_() -> Tensor &;

    /**
     * Performs element-wise trigonometric inverse tangent of the input Tensor
     * @return reference to self
     */
    auto atan_() -> Tensor &;

    /**
     * Performs element-wise hyperbolic sine of the input Tensor
     * @return reference to self
     */
    auto sinh_() -> Tensor &;

    /**
     * Performs element-wise hyperbolic cosine of the input Tensor
     * @return reference to self
     */
    auto cosh_() -> Tensor &;

    /**
     * Performs element-wise hyperbolic tangent of the input Tensor
     * @return reference to self
     */
    auto tanh_() -> Tensor &;

    /**
     * Performs element-wise inverse hyperbolic sine of the input Tensor
     * @return reference to self
     */
    auto asinh_() -> Tensor &;

    /**
     * Performs element-wise inverse hyperbolic cosine of the input Tensor
     * @return reference to self
     */
    auto acosh_() -> Tensor &;

    /**
     * Performs element-wise inverse hyperbolic tangent of the input Tensor
     * @return reference to self
     */
    auto atanh_() -> Tensor &;

    /**
     * Performs element-wise error function of the input Tensor
     * @note https://en.wikipedia.org/wiki/Error_function
     * @return reference to self
     */
    auto erf_() -> Tensor &;

    /**
     * Performs element-wise complementary error function of the input Tensor
     * @note https://en.wikipedia.org/wiki/Error_function
     * @return reference to self
     */
    auto erfc_() -> Tensor &;

    /**
     * Performs element-wise gamma function of the input Tensor
     * @note https://en.wikipedia.org/wiki/Gamma_function
     * @return reference to self
     */
    auto tgamma_() -> Tensor &;

    /**
     * Performs element-wise logarithm of the gamma function of the input Tensor
     * @note https://en.wikipedia.org/wiki/Gamma_function
     * @return reference to self
     */
    auto lgamma_() -> Tensor &;

    /**
     * Performs element-wise digamma function (derivative of log gamma) of the
     * input Tensor
     * @note https://en.wikipedia.org/wiki/Digamma_function
     * @note This function does not currently have autograd support
     * @return reference to self
     */
    auto digamma_() -> Tensor &;

    /**
     * Performs element-wise ceiling of the input Tensor
     * @note This function does not have autograd support
     * @return reference to self
     */
    auto ceil_() -> Tensor &;

    /**
     * Performs element-wise floor of the input Tensor
     * @note This function does not have autograd support
     * @return reference to self
     */
    auto floor_() -> Tensor &;

    /**
     * Performs element-wise rounding to nearest integer of the input Tensor
     * @note Rounding halfway cases away from zero
     * @note This function does not have autograd support
     * @return reference to self
     */
    auto round_() -> Tensor &;

    // ------------------------------------------------
    // Inplace Activation functions
    // ------------------------------------------------

    /**
     * Performs element-wise sigmoid of the input Tensor
     * @note https://en.wikipedia.org/wiki/Sigmoid_function
     * @return reference to self
     */
    auto sigmoid_() -> Tensor &;

    /**
     * Performs element-wise log-sigmoid of the input Tensor
     * @note https://en.wikipedia.org/wiki/Sigmoid_function
     * @return reference to self
     */
    auto log_sigmoid_() -> Tensor &;

    /**
     * Performs element-wise Hardsigmoid of the input Tensor
     * @note https://pytorch.org/docs/stable/generated/torch.nn.Hardsigmoid.html
     * @return reference to self
     */
    auto hardsigmoid_() -> Tensor &;

    /**
     * Performs element-wise softplus of the input Tensor
     * @note Softplus is a smooth approximation of the ReLU function (always
     * positive)
     * @note https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html
     * @param beta The beta value for Softplus
     * @param threshold Values above this revert to a linear function
     * @return reference to self
     */
    auto softplus_(double beta = 1, double threshold = 20) -> Tensor &;

    /**
     * Performs element-wise rectified linear unit function of the input Tensor
     * @note https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
     * @return reference to self
     */
    auto relu_() -> Tensor &;

    /**
     * Performs element-wise ReLU6 of the input Tensor
     * @note https://pytorch.org/docs/stable/generated/torch.nn.ReLU6.html
     * @return reference to self
     */
    auto relu6_() -> Tensor &;

    /**
     * Performs element-wise LeakyReLU of the input Tensor
     * @note https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html
     * @param negative_slop The cangle of the negative slope, used for negative
     * inputs
     * @return reference to self
     */
    auto leaky_relu_(double negative_slope = 0.01) -> Tensor &;

    /**
     * Performs element-wise Exponential Linear Unit (ELU) of the input Tensor
     * @note https://pytorch.org/docs/stable/generated/torch.nn.ELU.html
     * @param alpha The alpha value for the ELU formulation
     * @return reference to self
     */
    auto elu_(double alpha = 1) -> Tensor &;

    /**
     * Performs element-wise SELU of the input Tensor
     * @note https://pytorch.org/docs/stable/generated/torch.nn.SELU.html
     * @return reference to self
     */
    auto selu_() -> Tensor &;

    /**
     * Performs element-wise Sigmoid Linear Unit (SiLU) of the input Tensor
     * @note https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html
     * @return reference to self
     */
    auto silu_() -> Tensor &;

    /**
     * Performs element-wise HardTanh of the input Tensor
     * @note https://pytorch.org/docs/stable/generated/torch.nn.Hardtanh.html
     * @param min Minimum value of the linear region range
     * @param max Maximum value of the linear region range
     * @return reference to self
     */
    auto hardtanh_(double min = -1, double max = 1) -> Tensor &;

    /**
     * Performs element-wise Softsign of the input Tensor
     * @note https://pytorch.org/docs/stable/generated/torch.nn.Softsign.html
     * @return reference to self
     */
    auto softsign_() -> Tensor &;

    /**
     * Performs element-wise softmax over a dimension of the input Tensor
     * @note The elements along the given dimension are rescaled so that they lie
     * in the range [0,1] and sum to 1
     * @note https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
     * @param dim The dimension along which the Softmax will be computed
     * @return reference to self
     */
    auto softmax_(int dim) -> Tensor &;

    /**
     * Performs element-wise log-softmax over a dimension of the input Tensor
     * @note https://pytorch.org/docs/stable/generated/torch.nn.LogSoftmax.html
     * @param dim The dimension along which the log-softmax will be computed
     * @return reference to self
     */
    auto log_softmax_(int dim) -> Tensor &;

    // ------------------------------------------------
    // Inplace Binary operations
    // ------------------------------------------------

    /**
     * Performs element-wise addition
     * @param rhs The other tensor
     * @return reference to self
     */
    auto add_(const Tensor &rhs) -> Tensor &;

    /**
     * Performs element-wise addition
     * @param rhs The other scalar value
     * @return reference to self
     */
    template <IsScalarType T>
    auto add_(T rhs) -> Tensor & {
        return add_(Scalar(rhs, scalar_type_));
    }

    /**
     * Performs element-wise addition
     * @param rhs The other scalar value
     * @return reference to self
     */
    auto add_(const Scalar &rhs) -> Tensor &;

    /**
     * Performs element-wise subtraction
     * @param rhs The other tensor
     * @return reference to self
     */
    auto sub_(const Tensor &rhs) -> Tensor &;

    /**
     * Performs element-wise subtraction
     * @param rhs The other scalar value
     * @return reference to self
     */
    template <IsScalarType T>
    auto sub_(T rhs) -> Tensor & {
        return sub_(Scalar(rhs, scalar_type_));
    }

    /**
     * Performs element-wise subtraction
     * @param rhs The other scalar value
     * @return reference to self
     */
    auto sub_(const Scalar &rhs) -> Tensor &;

    /**
     * Performs element-wise multiplication
     * @param rhs The other tensor
     * @return reference to self
     */
    auto mul_(const Tensor &rhs) -> Tensor &;

    /**
     * Performs element-wise multiplication
     * @param rhs The other scalar value
     * @return reference to self
     */
    template <IsScalarType T>
    auto mul_(T rhs) -> Tensor & {
        return mul_(Scalar(rhs, scalar_type_));
    }

    /**
     * Performs element-wise multiplication
     * @param rhs The other scalar value
     * @return reference to self
     */
    auto mul_(const Scalar &rhs) -> Tensor &;

    /**
     * Performs element-wise division
     * @param rhs The other tensor
     * @return reference to self
     */
    auto div_(const Tensor &rhs) -> Tensor &;

    /**
     * Performs element-wise addition
     * @param rhs The other scalar value
     * @return reference to self
     */
    template <IsScalarType T>
    auto div_(T rhs) -> Tensor & {
        return div_(Scalar(rhs, scalar_type_));
    }

    /**
     * Performs element-wise division
     * @param rhs The other scalar value
     * @return reference to self
     */
    auto div_(const Scalar &rhs) -> Tensor &;

    // ------------------------------------------------
    // Inplace Distribution operations
    // ------------------------------------------------

    /**
     * Set the Tensor with values sampled from a uniform real distribution over
     * the interval [low, high)
     * @note requires low < high
     * @note The current Tensor must be of floating point type
     * @param low The inclusive left end of the interval
     * @param high The exclusive right end of the interval
     * @param gen The generator source of randomness
     */
    auto uniform_real_(double low, double high, Generator &gen = get_default_generator()) -> Tensor &;

    /**
     * Set the Tensor with values sampled from a uniform real distribution over
     * the interval [low, high), where the shape and type are from the the same
     * values of low and high
     * @note requires low < high for each element in the Tensors
     * @note The current Tensor must be of floating point type
     * @param low The inclusive left end of the interval
     * @param high The exclusive right end of the interval
     * @param gen The generator source of randomness
     */
    auto uniform_real_(const Tensor &low, const Tensor &high, Generator &gen = get_default_generator()) -> Tensor &;

    /**
     * Set the Tensor with values sampled from a uniform real distribution over
     * the interval [low, high)
     * @note requires low < high
     * @note The current Tensor must be of floating point type
     * @param low The inclusive left end of the interval
     * @param high The exclusive right end of the interval
     * @param gen The generator source of randomness
     */
    auto uniform_int_(int64_t low, int64_t high, Generator &gen = get_default_generator()) -> Tensor &;

    /**
     * Set the Tensor with values sampled from a uniform int distribution over the
     * interval [low, high)
     * @note requires low < high for each element in the Tensors
     * @note The current Tensor must be of floating point type
     * @param low The inclusive left end of the interval
     * @param high The exclusive right end of the interval
     * @param gen The generator source of randomness
     */
    auto uniform_int_(const Tensor &low, const Tensor &high, Generator &gen = get_default_generator()) -> Tensor &;

    /**
     * Set the Tensor with values sampled from a Bernoulli distribution
     * @note p must be in range [0, 1]
     * @note The current Tensor must be of floating point type
     * @param p The probability of success
     * @param gen The generator source of randomness
     */
    auto bernoulli_(double p, Generator &gen = get_default_generator()) -> Tensor &;

    /**
     * Set the Tensor with values sampled from a Bernoulli distribution
     * @note p must be in range [0, 1]
     * @note The current Tensor must be of floating point type
     * @param p The probability of success
     * @param gen The generator source of randomness
     */
    auto bernoulli_(const Tensor &p, Generator &gen = get_default_generator()) -> Tensor &;

    /**
     * Set the Tensor with values sampled from a Binomial distribution
     * @note p must be in range [0, 1]
     * @param p The probability of success
     * @param num_draws Number of Bernoulli draws
     * @param gen The generator source of randomness
     */
    auto binomial_(double p, int num_draws, Generator &gen = get_default_generator()) -> Tensor &;

    /**
     * Set the Tensor with values sampled from a Binomial distribution
     * @note p must be in range [0, 1]
     * @param p The probability of success
     * @param num_draws Number of Bernoulli draws
     * @param gen The generator source of randomness
     */
    auto binomial_(const Tensor &p, const Tensor &num_draws, Generator &gen = get_default_generator()) -> Tensor &;

    /**
     * Set the Tensor with values sampled from a Geometric distribution
     * @note Uses representation of X Bernoulli trials to get one success, support
     * = {1, 2, 3, ...}
     * @note p must be in range (0, 1]
     * @param p The probability of success
     * @param gen The generator source of randomness
     */
    auto geometric_(double p, Generator &gen = get_default_generator()) -> Tensor &;

    /**
     * Set the Tensor with values sampled from a Geometric distribution
     * @note Uses representation of X Bernoulli trials to get one success, support
     * = {1, 2, 3, ...}
     * @note p must be in range (0, 1]
     * @param p The probability of success
     * @param gen The generator source of randomness
     */
    auto geometric_(const Tensor &p, Generator &gen = get_default_generator()) -> Tensor &;

    /**
     * Set the Tensor with values sampled from a Poisson distribution
     * @param lambda rate parameter, must be > 0
     * @param gen The generator source of randomness
     */
    auto poisson_(double lambda, Generator &gen = get_default_generator()) -> Tensor &;

    /**
     * Set the Tensor with values sampled from a Exponential distribution
     * @param lambda rate parameter, must be > 0
     * @param gen The generator source of randomness
     */
    auto poisson_(const Tensor &lambda, Generator &gen = get_default_generator()) -> Tensor &;

    /**
     * Set the Tensor with values sampled from a Poisson distribution
     * @param lambda rate parameter, must be > 0
     * @param gen The generator source of randomness
     */
    auto exponential_(double lambda, Generator &gen = get_default_generator()) -> Tensor &;

    /**
     * Set the Tensor with values sampled from a Exponential distribution
     * @param lambda rate parameter, must be > 0
     * @param gen The generator source of randomness
     */
    auto exponential_(const Tensor &lambda, Generator &gen = get_default_generator()) -> Tensor &;

    /**
     * Set the Tensor with values sampled from a Normal distribution
     * @param mu The mean of the distribution
     * @param std The standard deviation, must be > 0
     * @param gen The generator source of randomness
     */
    auto normal_(double mu, double std, Generator &gen = get_default_generator()) -> Tensor &;

    /**
     * Set the Tensor with values sampled from a Normal distribution
     * @param mu The mean of the distribution
     * @param std The standard deviation, must be > 0
     * @param gen The generator source of randomness
     */
    auto normal_(const Tensor &mu, const Tensor &std, Generator &gen = get_default_generator()) -> Tensor &;

    /**
     * Set the Tensor with values sampled from a Cauchy distribution
     * @note See https://en.wikipedia.org/wiki/Cauchy_distribution
     * @param loc The mode/median of the distribution
     * @param scale The scale parameter, which is the half-width at maximum, must
     * be > 0
     * @param gen The generator source of randomness
     */
    auto cauchy_(double loc, double scale, Generator &gen = get_default_generator()) -> Tensor &;

    /**
     * Set the Tensor with values sampled from a Cauchy distribution, where the
     * shape is determined by loc/scale and the type is the promoted type between
     * loc, scale, and kDefaultInt
     * @note See https://en.wikipedia.org/wiki/Cauchy_distribution
     * @param loc The mode/median of the distribution
     * @param scale The scale parameter, which is the half-width at maximum, must
     * be > 0
     * @param gen The generator source of randomness
     */
    auto cauchy_(const Tensor &loc, const Tensor &scale, Generator &gen = get_default_generator()) -> Tensor &;

    /**
     * Set the Tensor with values sampled from a Log-Normal distribution
     * @param mu The mean of the underlying normal distribution
     * @param std The standard deviation of the underlying normal distribution,
     * must be > 0
     * @param gen The generator source of randomness
     */
    auto lognormal_(double mu, double std, Generator &gen = get_default_generator()) -> Tensor &;

    /**
     * Set the Tensor with values sampled from a Log-Normal distribution
     * @param mu The mean of the underlying normal distribution
     * @param std The standard deviation of the underlying normal distribution,
     * must be > 0
     * @param gen The generator source of randomness
     */
    auto lognormal_(const Tensor &mu, const Tensor &std, Generator &gen = get_default_generator()) -> Tensor &;

    /**
     * Set the Tensor with values sampled from a Weibull distribution
     * @note See https://en.wikipedia.org/wiki/Weibull_distribution
     * @param lambda The scale parameter, must be > 0
     * @param k The shape parameter, must be > 0
     * @param gen The generator source of randomness
     */
    auto weibull_(double lambda, double k, Generator &gen = get_default_generator()) -> Tensor &;

    /**
     * Set the Tensor with values sampled from a Weibull distribution
     * @note See https://en.wikipedia.org/wiki/Weibull_distribution
     * @param lambda The scale parameter, must be > 0
     * @param k The shape parameter, must be > 0
     * @param gen The generator source of randomness
     */
    auto weibull_(const Tensor &lambda, const Tensor &k, Generator &gen = get_default_generator()) -> Tensor &;

    // ------------------------------------------------
    // Reduction operations
    // ------------------------------------------------

    /**
     * Performs the minimum of all elements of the input Tensor
     * @return reference to self
     * @return Resulting Tensor
     */
    [[nodiscard]] auto min() const -> Tensor;

    /**
     * Performs the minimum of each element over a dimension of the input Tensor
     * @param dim The dimension to take the minimum over
     * @param keep_dim Whether the result has the reduction dimension retained or
     * not
     * @return Resulting Tensor
     */
    [[nodiscard]] auto min(int dim, bool keep_dim = false) const -> Tensor;

    /**
     * Performs the minimum of each element over a dimension of the input Tensor
     * @param dims The dimensions to take the minimum over
     * @param keep_dim Whether the result has the reduction dimension retained or
     * not
     * @return Resulting Tensor
     */
    [[nodiscard]] auto min(const std::vector<int> &dims, bool keep_dim = false) const -> Tensor;

    /**
     * Performs the index of the minimum of all elements of the input Tensor
     * @return Resulting Tensor
     */
    [[nodiscard]] auto argmin() const -> Tensor;

    /**
     * Performs the indices of the minimum elements over a dimension of the input
     * Tensor
     * @param dim The dimension to take the index of minimums over
     * @param keep_dim Whether the result has the reduction dimension retained or
     * not
     * @return Resulting Tensor
     */
    [[nodiscard]] auto argmin(int dim, bool keep_dim = false) const -> Tensor;

    /**
     * Performs the indices of the minimum elements over a dimension of the input
     * Tensor
     * @param dims The dimensions to take the index of minimums over
     * @param keep_dim Whether the result has the reduction dimension retained or
     * not
     * @return Resulting Tensor
     */
    [[nodiscard]] auto argmin(const std::vector<int> &dims, bool keep_dim = false) const -> Tensor;

    /**
     * Performs the maximum of all elements of the input Tensor
     * @return Resulting Tensor
     */
    [[nodiscard]] auto max() const -> Tensor;

    /**
     * Performs the maximum of each element over a dimension of the input Tensor
     * @param dim The dimension to take the maximum over
     * @param keep_dim Whether the result has the reduction dimension retained or
     * not
     * @return Resulting Tensor
     */
    [[nodiscard]] auto max(int dim, bool keep_dim = false) const -> Tensor;

    /**
     * Performs the maximum of each element over a dimension of the input Tensor
     * @param dims The dimensions to take the maximum over
     * @param keep_dim Whether the result has the reduction dimension retained or
     * not
     * @return Resulting Tensor
     */
    [[nodiscard]] auto max(const std::vector<int> &dims, bool keep_dim = false) const -> Tensor;

    /**
     * Performs the index of the maximum of all elements of the input Tensor
     * @return Resulting Tensor
     */
    [[nodiscard]] auto argmax() const -> Tensor;

    /**
     * Performs the indices of the maximum elements over a dimension of the input
     * Tensor
     * @param dim The dimension to take the index of maximums over
     * @param keep_dim Whether the result has the reduction dimension retained or
     * not
     * @return Resulting Tensor
     */
    [[nodiscard]] auto argmax(int dim, bool keep_dim = false) const -> Tensor;

    /**
     * Performs the indices of the maximum elements over a dimension of the input
     * Tensor
     * @param dims The dimensions to take the index of maximums over
     * @param keep_dim Whether the result has the reduction dimension retained or
     * not
     * @return Resulting Tensor
     */
    [[nodiscard]] auto argmax(const std::vector<int> &dims, bool keep_dim = false) const -> Tensor;

    /**
     * Performs the sum of all elements of the input Tensor
     * @return Resulting Tensor
     */
    [[nodiscard]] auto sum() const -> Tensor;

    /**
     * Performs the sum of each element over a dimension of the input Tensor
     * @param dim The dimension to take the sum over
     * @param keep_dim Whether the result has the reduction dimension retained or
     * not
     * @return Resulting Tensor
     */
    [[nodiscard]] auto sum(int dim, bool keep_dim = false) const -> Tensor;

    /**
     * Performs the sum of each element over a dimension of the input Tensor
     * @param dims The dimensions to take the sum over
     * @param keep_dim Whether the result has the reduction dimension retained or
     * not
     * @return Resulting Tensor
     */
    [[nodiscard]] auto sum(const std::vector<int> &dims, bool keep_dim = false) const -> Tensor;

    /**
     * Performs the mean of all elements of the input Tensor
     * @return Resulting Tensor
     */
    [[nodiscard]] auto mean() const -> Tensor;

    /**
     * Performs the mean of each element over a dimension of the input Tensor
     * @param dim The dimension to take the mean over
     * @param keep_dim Whether the result has the reduction dimension retained or
     * not
     * @return Resulting Tensor
     */
    [[nodiscard]] auto mean(int dim, bool keep_dim = false) const -> Tensor;

    /**
     * Performs the mean of each element over a dimension of the input Tensor
     * @param dims The dimensions to take the mean over
     * @param keep_dim Whether the result has the reduction dimension retained or
     * not
     * @return Resulting Tensor
     */
    [[nodiscard]] auto mean(const std::vector<int> &dims, bool keep_dim = false) const -> Tensor;

    /**
     * Compute the variance of the input
     * @param keep_dim Whether the result has the reduction dimension retained or
     * not
     * @param correction Difference between sample size and sampel degrees of
     * freedom
     * @return the resulting Tensor
     */
    [[nodiscard]] auto var(bool keep_dim = false, int correction = 1) const -> Tensor;

    /**
     * Compute the variance of the input over a given dimension
     * @param dim The dimension to reduce over
     * @param keep_dim Whether the result has the reduction dimension retained or
     * not
     * @param correction Difference between sample size and sampel degrees of
     * freedom
     * @return the resulting Tensor
     */
    [[nodiscard]] auto var(int dim, bool keep_dim = false, int correction = 1) const -> Tensor;

    /**
     * Compute the variance of the input over a given dimension
     * @param dims The dimensions to reduce over
     * @param keep_dim Whether the result has the reduction dimension retained or
     * not
     * @param correction Difference between sample size and sampel degrees of
     * freedom
     * @return the resulting Tensor
     */
    [[nodiscard]] auto var(const std::vector<int> &dims, bool keep_dim = false, int correction = 1) const -> Tensor;

    /**
     * Tests if all elements evaluate to true in the input Tensor
     */
    [[nodiscard]] auto all() const -> bool;

    /**
     * Tests if all elements evaluate to true over a dimension of the input Tensor
     * @note The type of the returned Tensor is kBool
     * @param dim The dimension to take the reduce over
     * @param keep_dim Whether the result has the reduction dimension retained or
     * not
     * @return Resulting Tensor
     */
    [[nodiscard]] auto all(int dim, bool keep_dim = false) const -> Tensor;

    /**
     * Tests if all elements evaluate to true over a dimension of the input Tensor
     * @note The type of the returned Tensor is kBool
     * @param dims The dimensions to take the reduce over
     * @param keep_dim Whether the result has the reduction dimension retained or
     * not
     * @return Resulting Tensor
     */
    [[nodiscard]] auto all(const std::vector<int> &dims, bool keep_dim = false) const -> Tensor;

    /**
     * Tests if any elements evaluate to true in the input Tensor
     */
    [[nodiscard]] auto any() const -> bool;

    /**
     * Tests if any element evaluates to true over a dimension of the input Tensor
     * @note The type of the returned Tensor is kBool
     * @param dim The dimension to take the reduce over
     * @param keep_dim Whether the result has the reduction dimension retained or
     * not
     * @return Resulting Tensor
     */
    [[nodiscard]] auto any(int dim, bool keep_dim = false) const -> Tensor;

    /**
     * Tests if any element evaluates to true over a dimension of the input Tensor
     * @note The type of the returned Tensor is kBool
     * @param dims The dimensions to take the reduce over
     * @param keep_dim Whether the result has the reduction dimension retained or
     * not
     * @return Resulting Tensor
     */
    [[nodiscard]] auto any(const std::vector<int> &dims, bool keep_dim = false) const -> Tensor;

    // ------------------------------------------------
    // Misc operations
    // ------------------------------------------------

    /**
     * Clamps all elements inplace into the range [min, max] given by the
     * ClampOptions
     * @param options Clamp options holding min/max values
     * @return reference to self
     */
    auto clamp_(const ClampOptions &options = ClampOptions()) -> Tensor &;

    /**
     * Clamps all elements into the range [min, max] given by the ClampOptions
     * @param options Clamp options holding min/max values
     * @return the clamped tensor
     */
    auto clamp(const ClampOptions &options = ClampOptions()) const -> Tensor;

    /**
     * Clamps all elements inplace into the range [min, max]
     * @note min and max must be shape shape as the self Tensor
     * @param min Lower-bound on the range to be clamped to
     * @param max Upper-bound on the range to be clamped to
     * @return reference to self
     */
    auto clamp_(const Tensor &min, const Tensor &max) -> Tensor &;

    /**
     * Clamps all elements into the range [min, max]
     * @note min and max must be shape shape as the self Tensor
     * @param min Lower-bound on the range to be clamped to
     * @param max Upper-bound on the range to be clamped to
     * @return the clamped tensor
     */
    auto clamp(const Tensor &min, const Tensor &max) const -> Tensor;

private:
    // Friends
    friend std::ostream &operator<<(std::ostream &os, const Tensor &tensor);

    template <typename T>
    friend struct autograd::TensorFunction;

    friend auto autograd::build_dag(const Tensor &tensor) -> TensorList;
    friend void autograd::backward(Tensor &tensor, const Tensor &grad, bool retain_graph);
    friend void autograd::calc_grad_input(Tensor &tensor, bool retain_graph);
    friend auto make_dot(const Tensor &tensor) -> std::string;

    // private autograd related method
    void apply_grad_hook();

    Device device_;
    ScalarType scalar_type_;
    int offset_;
    Shape shape_;
    Shape stride_;
    std::shared_ptr<StorageBase> storage_;         // Underlying storage
    std::shared_ptr<autograd::SharedGrad> ctx_;    // Gradient related shared data
    mutable int version_count_ = 0;
};

// Extension for vector of tensors, supports negative indexing and bounds
// checking by default
template <typename T>
class TINYTENSOR_EXPORT CheckedVec {
    std::vector<T> data_;

public:
    using value_type = decltype(data_)::value_type;
    using Iterator = decltype(data_)::iterator;
    using ConstIterator = decltype(data_)::const_iterator;

    CheckedVec() = default;

    template <typename InputIt>
    CheckedVec(InputIt first, InputIt last)
        : data_(first, last) {}
    explicit CheckedVec(const std::vector<T> &data)
        : data_(data) {};
    explicit CheckedVec(std::vector<T> &&data)
        : data_(std::move(data)) {};
    CheckedVec(std::initializer_list<T> data)
        : data_(data) {};

    auto to_vec() const -> std::vector<T> {
        return data_;
    }

    void push_back(const T &item) {
        data_.push_back(item);
    }
    void push_back(T &&item) {
        data_.push_back(std::move(item));
    }
    template <typename... Args>
    void emplace_back(Args &&...args) {
        push_back(T(std::forward<Args>(args)...));
    }
    void clear() {
        data_.clear();
    }
    [[nodiscard]] auto empty() const -> bool {
        return data_.empty();
    }
    [[nodiscard]] auto size() const -> int {
        return static_cast<int>(data_.size());
    }
    [[nodiscard]] auto operator[](int idx) -> T & {
        int N = static_cast<int>(data_.size());
        if (idx >= N || idx < -N) {
            TT_EXCEPTION(std::format("Invalid idx, expected to be in range[{}, {}]", -N, N - 1));
        }
        idx = (idx < 0) ? (idx + N) % N : idx;
        return data_[static_cast<std::size_t>(idx)];
    }
    [[nodiscard]] auto operator[](int idx) const -> const T & {
        int N = static_cast<int>(data_.size());
        if (idx >= N || idx < -N) {
            TT_EXCEPTION(std::format("Invalid dim, expected to be in range[{}, {}]", -N, N - 1));
        }
        idx = (idx < 0) ? (idx + N) % N : idx;
        return data_[static_cast<std::size_t>(idx)];
    }
    [[nodiscard]] auto begin() -> Iterator {
        return std::begin(data_);
    }
    [[nodiscard]] auto begin() const -> ConstIterator {
        return std::begin(data_);
    }
    [[nodiscard]] auto end() -> Iterator {
        return std::end(data_);
    }
    [[nodiscard]] auto end() const -> ConstIterator {
        return std::end(data_);
    }
};

// ------------------------------------------------
// Tensor Creation
// ------------------------------------------------

/**
 * Create a Tensor of a given size, filled with a given value
 * @note The type of the resulting Tensor is the scalar type of the given value
 * @param value The value to fill
 * @param shape The shape of the desired Tensor
 * @param device The device of the desried Tensor
 * @param requires_grad Flag if autograd operations should be recorded
 * @return The value filled tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto full(Scalar value, Shape shape, Device device, bool requires_grad = false)
    -> Tensor;

/**
 * Create a Tensor of a given size, filled with a given value
 * @param value The value to fill
 * @param shape The shape of the desired Tensor
 * @param dtype The scalar type of the desired Tensor
 * @param device The device of the desried Tensor
 * @param requires_grad Flag if autograd operations should be recorded
 * @return The value filled tensor
 */
template <typename T>
    requires(IsScalarType<T> || std::is_same_v<T, bool>)
[[nodiscard]] TINYTENSOR_EXPORT auto
    full(T value, Shape shape, ScalarType dtype, Device device, bool requires_grad = false) -> Tensor {
    return full(Scalar(value, dtype), shape, device, requires_grad);
}

/**
 * Create a Tensor of a given size, filled with a given value
 * @param value The value to fill
 * @param shape The shape of the desired Tensor
 * @param device The device of the desried Tensor
 * @param requires_grad Flag if autograd operations should be recorded
 * @return The value filled tensor
 */
template <typename T>
    requires(IsScalarType<T> || std::is_same_v<T, bool>)
[[nodiscard]] TINYTENSOR_EXPORT auto full(T value, Shape shape, Device device, bool requires_grad = false) -> Tensor {
    return full(Scalar(value), shape, device, requires_grad);
}

/**
 * Create a Tensor of a given size, filled with a given value
 * @param value The value to fill
 * @param shape The shape of the desired Tensor
 * @param options The Tensor options of scalar type and device of the desired
 * Tensor
 * @return The value filled tensor
 */
template <typename T>
    requires(IsScalarType<T> || std::is_same_v<T, bool>)
[[nodiscard]] TINYTENSOR_EXPORT auto full(T value, Shape shape, const TensorOptions &options) -> Tensor {
    return full(Scalar(value, options.dtype()), shape, options.device(), options.requires_grad());
}

/**
 * Create a Tensor filled with the value 0
 * @param shape The shape of the desired Tensor
 * @param dtype The scalar type of the desired Tensor
 * @param device The device of the desried Tensor
 * @param requires_grad Flag if autograd operations should be recorded
 * @return The zero filled Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto zeros(Shape shape, ScalarType dtype, Device device, bool requires_grad = false)
    -> Tensor;

/**
 * Create a Tensor filled with the value 0
 * @param shape The shape of the desired Tensor
 * @param options The Tensor options of scalar type and device of the desired
 * Tensor
 * @return The zero filled Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto zeros(Shape shape, const TensorOptions &options = TensorOptions()) -> Tensor;

/**
 * Create a Tensor filled with the value 0 with the same shape, scalar type, and
 * device of the given Tensor
 * @param tensor The reference Tensor
 * @param requires_grad Flag if autograd operations should be recorded
 * @return The zero filled Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto zeros_like(const Tensor &tensor, bool requires_grad = false) -> Tensor;

/**
 * Create a Tensor filled with the value 1
 * @param shape The shape of the desired Tensor
 * @param dtype The scalar type of the desired Tensor
 * @param device The device of the desried Tensor
 * @param requires_grad Flag if autograd operations should be recorded
 * @return The ones filled Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto ones(Shape shape, ScalarType dtype, Device device, bool requires_grad = false)
    -> Tensor;

/**
 * Create a Tensor filled with the value 1
 * @param shape The shape of the desired Tensor
 * @param options The tensor options of scalar type and device of the desired
 * Tensor
 * @return The ones filled Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto ones(Shape shape, const TensorOptions &options = TensorOptions()) -> Tensor;

/**
 * Create a Tensor filled with the value 1 with the same shape, scalar type, and
 * device of the given Tensor
 * @param tensor The reference Tensor
 * @param requires_grad Flag if autograd operations should be recorded
 * @return The ones filled Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto ones_like(const Tensor &tensor, bool requires_grad = false) -> Tensor;

/**
 * Create a Tensor filled with values [0, n), where n is determined by the
 * number of elements in the given shape
 * @param shape The shape of the desired Tensor
 * @param dtype The scalar type of the desired Tensor
 * @param device The device of the desried Tensor
 * @param requires_grad Flag if autograd operations should be recorded
 * @return The Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto arange(Shape shape, ScalarType dtype, Device device, bool requires_grad = false)
    -> Tensor;

/**
 * Create a Tensor filled with values [0, n), where n is determined by the
 * number of elements in the given shape
 * @param shape The shape of the desired Tensor
 * @param options The tensor options of scalar type and device of the desired
 * Tensor
 * @return The Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto arange(Shape shape, const TensorOptions &options = TensorOptions(kDefaultInt))
    -> Tensor;

/**
 * Create a Tensor filled with N values over an evenly-spaced interval, where N
 * is determined by the number of elements in the given shape
 * @param start The start of the interval
 * @param stop The end of the interval
 * @param endpoint True for stop to be the last sample, false for it to not be
 * included
 * @param shape The shape of the desired Tensor
 * @param dtype The scalar type of the desired Tensor
 * @param device The device of the desried Tensor
 * @param requires_grad Flag if autograd operations should be recorded
 * @return The Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto linspace(
    double start,
    double stop,
    bool endpoint,
    Shape shape,
    ScalarType dtype,
    Device device,
    bool requires_grad = false
) -> Tensor;

/**
 * Create a Tensor filled with N values over an evenly-spaced interval, where N
 * is determined by the number of elements in the given shape
 * @param start The start of the interval
 * @param stop The end of the interval
 * @param endpoint True for stop to be the last sample, false for it to not be
 * included
 * @param shape The shape of the desired Tensor
 * @param options The tensor options of scalar type and device of the desired
 * Tensor
 * @return The Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto linspace(
    double start,
    double end,
    bool endpoint,
    Shape shape,
    const TensorOptions &options = TensorOptions(kDefaultFloat)
) -> Tensor;

/**
 * Create a 2D Tensor ones on the main diagonal, zeros elsewhere
 * @param rows The number of rows
 * @param cols The number of columns
 * @param dtype The scalar type of the desired Tensor
 * @param device The device of the desried Tensor
 * @param requires_grad Flag if autograd operations should be recorded
 * @return The Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto
    eye(int rows, int cols, ScalarType dtype, Device device, bool requires_grad = false) -> Tensor;

/**
 * Create a 2D Tensor ones on the main diagonal, zeros elsewhere
 * @param rows The number of rows
 * @param cols The number of columns
 * @param options The tensor options of scalar type and device of the desired
 * Tensor
 * @return The Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto eye(int rows, int cols, const TensorOptions &options = TensorOptions(kDefaultInt))
    -> Tensor;

/**
 * Create a 2D Tensor with zeros everywhere except where the index of the
 * innermost dim matches the corresponding value
 * @note The given indices tensor should be of integral scalar type
 * @param num_classes The maximum number of classes (i.e. the number of
 * columns), use -1 to use the maximum of the passed indices
 * @return The Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto one_hot(Tensor indices, int num_classes) -> Tensor;

// ------------------------------------------------------------
// Tensor Joining
// ------------------------------------------------------------

/**
 * Concatenates the list of Tensors into the given dimension
 * All Tensors must have the same shape, except for the concatenating dimension
 * @note This will allocate new memory for the combined elements
 * @note See https://pytorch.org/docs/main/generated/torch.cat.html
 * @param tensors The list of tensors to concatenate
 * @param dim The dimension to concatenate into
 * @return The concatenated Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto cat(const TensorList &tensors, int dim) -> Tensor;

/**
 * Concatenates the list of Tensors into the given dimension into an existing
 * Tensor All Tensors must have the same shape, except for the concatenating
 * dimension
 * @note See https://pytorch.org/docs/main/generated/torch.cat.html
 * @param tensors The list of tensors to concatenate
 * @param dim The dimension to concatenate into
 * @param out The output Tensor, which must have the same number of elements as
 * the sum of elements in the given list
 */
void TINYTENSOR_EXPORT cat(const TensorList &tensors, int dim, Tensor &out);

/**
 * Concatenates the list of Tensors into a new given dimension
 * All Tensors must have the same shape
 * @note This is equivalent to unsqueezing all Tensors using the given dim then
 * concatenating
 * @note This will allocate new memory for the combined elements
 * @note See
 * https://pytorch.org/docs/main/generated/torch.stack.html#torch.stack
 * @param tensors The list of tensors to concatenate
 * @param dim The dimension to insert, between 0 and number of dimensions of the
 * concatenated Tensors
 * @return The concatenated Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto stack(const TensorList &tensors, int dim) -> Tensor;

/**
 * Concatenates the list of Tensors into a new given dimension into an existing
 * Tensor All Tensors must have the same shape
 * @note This is equivalent to unsqueezing all Tensors using the given dim then
 * concatenating
 * @note See
 * https://pytorch.org/docs/main/generated/torch.stack.html#torch.stack
 * @param tensors The list of tensors to concatenate
 * @param dim The dimension to insert, between 0 and number of dimensions of the
 * concatenated Tensors
 * @param out The output Tensor, which must have the same number of elements as
 * the sum of elements in the given list
 */
TINYTENSOR_EXPORT void stack(const TensorList &tensors, int dim, Tensor &out);

// ------------------------------------------------------------
// Uniform Distributions
// ------------------------------------------------------------

/**
 * Generate a Tensor with values sampled from a uniform real distribution over
 * the interval [low, high), where the shape and type are from the the same
 * values of low and high
 * @note requires low < high for each element in the Tensors
 * @param low The inclusive left end of the interval
 * @param high The exclusive right end of the interval
 * @param requires_grad Flag if autograd operations should be recorded
 * @param gen The generator source of randomness
 * @return The generated Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto uniform_real(
    const Tensor &low,
    const Tensor &high,
    bool requires_grad = false,
    Generator &gen = get_default_generator()
) -> Tensor;

/**
 * Generate a Tensor with values sampled from a uniform real distribution over
 * the interval [low, high)
 * @note requires low < high
 * @param low The inclusive left end of the interval
 * @param high The exclusive right end of the interval
 * @param shape The shape of the Tensor to generate
 * @param options The tensor options of scalar type and device of the desired
 * Tensor
 * @param gen The generator source of randomness
 * @return The generated Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto uniform_real(
    double low,
    double high,
    Shape shape,
    const TensorOptions &options = TensorOptions(),
    Generator &gen = get_default_generator()
) -> Tensor;

/**
 * Generate a Tensor with values sampled from a uniform int distribution over
 * the interval [low, high), where the shape is determined by low/high and the
 * type is the promoted type between low, high, and kDefaultInt
 * @note requires low < high for each element in the Tensors
 * @param low The inclusive left end of the interval
 * @param high The exclusive right end of the interval
 * @param gen The generator source of randomness
 * @return The generated Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto
    uniform_int(const Tensor &low, const Tensor &high, Generator &gen = get_default_generator()) -> Tensor;

/**
 * Generate a Tensor with values sampled from a uniform real distribution over
 * the interval [low, high)
 * @note requires low < high
 * @param low The inclusive left end of the interval
 * @param high The exclusive right end of the interval
 * @param shape The shape of the Tensor to generate
 * @param options The tensor options of scalar type and device of the desired
 * Tensor
 * @param gen The generator source of randomness
 * @return The generated Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto uniform_int(
    int64_t low,
    int64_t high,
    Shape shape,
    const TensorOptions &options = TensorOptions(),
    Generator &gen = get_default_generator()
) -> Tensor;

// ------------------------------------------------------------
// Bernoulli Distributions
// ------------------------------------------------------------

/**
 * Generate a Tensor with values sampled from a Bernoulli distribution, where
 * the shape is determined by p and the type is the promoted type between p and
 * kDefaultFloat
 * @note p must be in range [0, 1]
 * @param p The probability of success
 * @param requires_grad Flag if autograd operations should be recorded
 * @param gen The generator source of randomness
 * @return The generated Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto
    bernoulli(const Tensor &p, bool requires_grad = false, Generator &gen = get_default_generator()) -> Tensor;

/**
 * Generate a Tensor with values sampled from a Bernoulli distribution, where
 * the type is kDefaultFloat if options.dtype is integral
 * @note p must be in range [0, 1]
 * @param p The probability of success
 * @param shape The shape of the Tensor to generate
 * @param options The tensor options of scalar type and device of the desired
 * Tensor
 * @param gen The generator source of randomness
 * @return The generated Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto bernoulli(
    double p,
    Shape shape,
    const TensorOptions &options = TensorOptions(),
    Generator &gen = get_default_generator()
) -> Tensor;

/**
 * Generate a Tensor with values sampled from a Binomial distribution, where the
 * shape is determined by p and the type is the promoted type between p,
 * num_draws, and kDefaultFloat
 * @note p must be in range [0, 1]
 * @param p The probability of success
 * @param num_draws Number of Bernoulli draws
 * @param requires_grad Flag if autograd operations should be recorded
 * @param gen The generator source of randomness
 * @return The generated Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto binomial(
    const Tensor &p,
    const Tensor &num_draws,
    bool requires_grad = false,
    Generator &gen = get_default_generator()
) -> Tensor;

/**
 * Generate a Tensor with values sampled from a Binomial distribution, where the
 * type is kDefaultFloat if options.dtype is integral
 * @note p must be in range [0, 1]
 * @param p The probability of success
 * @param num_draws Number of Bernoulli draws
 * @param shape The shape of the Tensor to generate
 * @param options The tensor options of scalar type and device of the desired
 * Tensor
 * @param gen The generator source of randomness
 * @return The generated Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto binomial(
    double p,
    int num_draws,
    Shape shape,
    const TensorOptions &options = TensorOptions(),
    Generator &gen = get_default_generator()
) -> Tensor;

/**
 * Generate a Tensor with values sampled from a Geometric distribution, where
 * the shape is determined by p and the type is the promoted type between p and
 * kDefaultFloat
 * @note Uses representation of X Bernoulli trials to get one success, support =
 * {1, 2, 3, ...}
 * @note p must be in range (0, 1]
 * @param p The probability of success
 * @param requires_grad Flag if autograd operations should be recorded
 * @param gen The generator source of randomness
 * @return The generated Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto
    geometric(const Tensor &p, bool requires_grad = false, Generator &gen = get_default_generator()) -> Tensor;

/**
 * Generate a Tensor with values sampled from a Geometric distribution, where
 * the type is kDefaultFloat if options.dtype is integral
 * @note Uses representation of X Bernoulli trials to get one success, support =
 * {1, 2, 3, ...}
 * @note p must be in range (0, 1]
 * @param p The probability of success
 * @param shape The shape of the Tensor to generate
 * @param options The tensor options of scalar type and device of the desired
 * Tensor
 * @param gen The generator source of randomness
 * @return The generated Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto geometric(
    double p,
    Shape shape,
    const TensorOptions &options = TensorOptions(),
    Generator &gen = get_default_generator()
) -> Tensor;

// ------------------------------------------------------------
// Exponential Distributions
// ------------------------------------------------------------

/**
 * Generate a Tensor with values sampled from a Poisson distribution, where the
 * shape is determined by lambda and the type is the promoted type between
 * lambda and kDefaultFloat
 * @param lambda rate parameter, must be > 0
 * @param requires_grad Flag if autograd operations should be recorded
 * @param gen The generator source of randomness
 * @return The generated Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto
    poisson(const Tensor &lambda, bool requires_grad = false, Generator &gen = get_default_generator()) -> Tensor;

/**
 * Generate a Tensor with values sampled from a Poisson distribution, where the
 * type is kDefaultFloat if options.dtype is integral
 * @param lambda rate parameter, must be > 0
 * @param shape The shape of the Tensor to generate
 * @param options The tensor options of scalar type and device of the desired
 * Tensor
 * @param gen The generator source of randomness
 * @return The generated Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto poisson(
    double lambda,
    Shape shape,
    const TensorOptions &options = TensorOptions(),
    Generator &gen = get_default_generator()
) -> Tensor;

/**
 * Generate a Tensor with values sampled from a Exponential distribution, where
 * the shape is determined by lambda and the type is the promoted type between
 * lambda and kDefaultFloat
 * @param lambda rate parameter, must be > 0
 * @param requires_grad Flag if autograd operations should be recorded
 * @param gen The generator source of randomness
 * @return The generated Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto
    exponential(const Tensor &lambda, bool requires_grad = false, Generator &gen = get_default_generator()) -> Tensor;

/**
 * Generate a Tensor with values sampled from a Exponential distribution, where
 * the type is kDefaultFloat if options.dtype is integral
 * @param lambda rate parameter, must be > 0
 * @param shape The shape of the Tensor to generate
 * @param options The tensor options of scalar type and device of the desired
 * Tensor
 * @param gen The generator source of randomness
 * @return The generated Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto exponential(
    double lambda,
    Shape shape,
    const TensorOptions &options = TensorOptions(),
    Generator &gen = get_default_generator()
) -> Tensor;

// ------------------------------------------------------------
// Normal Distributions
// ------------------------------------------------------------

/**
 * Generate a Tensor with values sampled from a Normal distribution, where the
 * shape is determined by mu/std and the type is the promoted type between mu,
 * std, and kDefaultInt
 * @param mu The mean of the distribution
 * @param std The standard deviation, must be > 0
 * @param requires_grad Flag if autograd operations should be recorded
 * @param gen The generator source of randomness
 * @return The generated Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto
    normal(const Tensor &mu, const Tensor &std, bool requires_grad = false, Generator &gen = get_default_generator())
        -> Tensor;

/**
 * Generate a Tensor with values sampled from a Normal distribution, where the
 * type is kDefaultFloat if options.dtype is integral
 * @param mu The mean of the distribution
 * @param std The standard deviation, must be > 0
 * @param shape The shape of the Tensor to generate
 * @param options The tensor options of scalar type and device of the desired
 * Tensor
 * @param gen The generator source of randomness
 * @return The generated Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto normal(
    double mu,
    double std,
    Shape shape,
    const TensorOptions &options = TensorOptions(),
    Generator &gen = get_default_generator()
) -> Tensor;

/**
 * Generate a Tensor with values sampled from a Cauchy distribution, where the
 * shape is determined by loc/scale and the type is the promoted type between
 * loc, scale, and kDefaultInt
 * @note See https://en.wikipedia.org/wiki/Cauchy_distribution
 * @param loc The mode/median of the distribution
 * @param scale The scale parameter, which is the half-width at maximum, must be
 * > 0
 * @param requires_grad Flag if autograd operations should be recorded
 * @param gen The generator source of randomness
 * @return The generated Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto
    cauchy(const Tensor &loc, const Tensor &scale, bool requires_grad = false, Generator &gen = get_default_generator())
        -> Tensor;

/**
 * Generate a Tensor with values sampled from a Cauchy distribution, where the
 * type is kDefaultFloat if options.dtype is integral
 * @note See https://en.wikipedia.org/wiki/Cauchy_distribution
 * @param loc The mode/median of the distribution
 * @param scale The scale parameter, which is the half-width at maximum, must be
 * > 0
 * @param shape The shape of the Tensor to generate
 * @param options The tensor options of scalar type and device of the desired
 * Tensor
 * @param gen The generator source of randomness
 * @return The generated Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto cauchy(
    double loc,
    double scale,
    Shape shape,
    const TensorOptions &options = TensorOptions(),
    Generator &gen = get_default_generator()
) -> Tensor;

/**
 * Generate a Tensor with values sampled from a Log-Normal distribution, where
 * the shape is determined by mu/std and the type is the promoted type between
 * mu, std, and kDefaultInt
 * @param mu The mean of the underlying normal distribution
 * @param std The standard deviation of the underlying normal distribution, must
 * be > 0
 * @param requires_grad Flag if autograd operations should be recorded
 * @param gen The generator source of randomness
 * @return The generated Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto
    lognormal(const Tensor &mu, const Tensor &std, bool requires_grad = false, Generator &gen = get_default_generator())
        -> Tensor;

/**
 * Generate a Tensor with values sampled from a Log-Normal distribution, where
 * the type is kDefaultFloat if options.dtype is integral
 * @param mu The mean of the underlying normal distribution
 * @param std The standard deviation of the underlying normal distribution, must
 * be > 0
 * @param shape The shape of the Tensor to generate
 * @param options The tensor options of scalar type and device of the desired
 * Tensor
 * @param gen The generator source of randomness
 * @return The generated Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto lognormal(
    double mu,
    double std,
    Shape shape,
    const TensorOptions &options = TensorOptions(),
    Generator &gen = get_default_generator()
) -> Tensor;

/**
 * Generate a Tensor with values sampled from a Weibull distribution, where the
 * shape is determined by lambda/k and the type is the promoted type between
 * lambda, k, and kDefaultInt
 * @note See https://en.wikipedia.org/wiki/Weibull_distribution
 * @param lambda The scale parameter, must be > 0
 * @param k The shape parameter, must be > 0
 * @param requires_grad Flag if autograd operations should be recorded
 * @param gen The generator source of randomness
 * @return The generated Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto
    weibull(const Tensor &lambda, const Tensor &k, bool requires_grad = false, Generator &gen = get_default_generator())
        -> Tensor;

/**
 * Generate a Tensor with values sampled from a Weibull distribution, where the
 * type is kDefaultFloat if options.dtype is integral
 * @note See https://en.wikipedia.org/wiki/Weibull_distribution
 * @param lambda The scale parameter, must be > 0
 * @param k The shape parameter, must be > 0
 * @param shape The shape of the Tensor to generate
 * @param options The tensor options of scalar type and device of the desired
 * Tensor
 * @param gen The generator source of randomness
 * @return The generated Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto weibull(
    double lambda,
    double k,
    Shape shape,
    const TensorOptions &options = TensorOptions(),
    Generator &gen = get_default_generator()
) -> Tensor;

// ------------------------------------------------
// Shape Modification operations
// ------------------------------------------------

/**
 * If two shapes are broadcastable, finds the resulting Shape
 * If dimensions not same length, 1s are prepended to smaller of the two. Then
 * result is the max size along each dim
 * @note: See https://pytorch.org/docs/stable/notes/broadcasting.html
 * @param lhs The first shape
 * @param rhs The second shape
 * @return The resulting shape
 */
[[nodiscard]] TINYTENSOR_EXPORT auto broadcast_result_shape(const Shape lhs, const Shape rhs) -> Shape;

/**
 * Check if the input shape can be broadcasted to the target shape
 * Two shapes are broadcastable if when iterating over the dimension sizes,
 * starting at the trailing dimension, the dimension sizes must be equal, one of
 * them is 1, or one of them does not exist
 * @note See https://pytorch.org/docs/stable/notes/broadcasting.html
 * @param shape The current shape
 * @param target_shape The target shape to broadcast to
 * @return True if the input shape can broadcast to the target shape, false
 * otherwise
 */
TINYTENSOR_EXPORT auto can_broadcast_to(const Shape &shape, const Shape &target_shape) -> bool;

/**
 * Check if two shapes are broadcastable to a common shape
 * Two shapes are broadcastable if when iterating over the dimension sizes,
 * starting at the trailing dimension, the dimension sizes must be equal, one of
 * them is 1, or one of them does not exist
 * @note See https://pytorch.org/docs/stable/notes/broadcasting.html
 * @param lhs LHS shape
 * @param rhs RHS shape
 * @return True if LHS and RHS can broadcast to common shape, false otherwise
 */
TINYTENSOR_EXPORT auto are_broadcastable(const Shape &lhs, const Shape &rhs) -> bool;

/**
 * Expands a Tensor to the given shape, which can include a larger number of
 * dimensions New dimensions are appended to the front, and any non-singleton
 * existing dimension must match the expanded-to dimension
 * @note: This will always return a view of the input Tensor, sharing the
 * underlying storage
 * @note: This is equivalent to calling expand(tensor, shape)
 * @note: See https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html
 * @param tensor The input Tensor
 * @param shape The shape to expand to
 * @return View of the expanded Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto broadcast_to(const Tensor &tensor, const Shape &shape) -> Tensor;

/**
 * Expands a Tensor to the given shape, which can include a larger number of
 * dimensions New dimensions are appended to the front, and any non-singleton
 * existing dimension must match the expanded-to dimension
 * @note: This will always return a view of the input Tensor, sharing the
 * underlying storage
 * @note: See https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html
 * @param tensor The input Tensor
 * @param shape The shape to expand to
 * @return View of the expanded Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto expand(const Tensor &tensor, const Shape &shape) -> Tensor;

/**
 * Removes a dimension of size one at a given dimension location to the input
 * Tensor
 * @note: This has the same view or copy properties as reshape
 * @note: If the given dimension to squeeze is not of size 1, the Tensor is
 * unchanged
 * @param tensor The input Tensor
 * @param dim The index to remove the singleton dimension
 * @return View of the Tensor with removed dimension
 */
[[nodiscard]] TINYTENSOR_EXPORT auto squeeze(const Tensor &tensor, int dim) -> Tensor;

/**
 * Inserts a dimension of size one at a given dimension location to the input
 * Tensor
 * @note: This has the same view or copy properties as reshape
 * @param tensor The input Tensor
 * @param dim The index to insert the singleton dimension
 * @return View of the Tensor with inserted dimension
 */
[[nodiscard]] TINYTENSOR_EXPORT auto unsqueeze(const Tensor &tensor, int dim) -> Tensor;

/**
 * Reshapes the input Tensor to the specified shape, keeping the same underlying
 * data and number of elements
 * @note: This will return a view of the input Tensor if possible (i.e
 * contiguous), or a copy of the underlying storage
 * @param tensor The input Tensor
 * @param shape The shape of the resulting Tensor to reshape to
 * @return The reshaped Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto reshape(const Tensor &tensor, const Shape &shape) -> Tensor;

/**
 * Flattens the input Tensor into a one-dimensional Tensor, equivalent to
 * reshape({tensor.numel()})
 * @note: If start_dim and end_dim are given, only dimensions between start/end
 * are flattened
 * @note: This has the same view or copy properties as reshape
 * @param tensor The input Tensor
 * @param start_dim Starting dim to flatten
 * @param end_dim Ending dim to flatten
 * @return View of flattened Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto flatten(const Tensor &tensor, int start_dim = 0, int end_dim = -1) -> Tensor;

/**
 * Permutes the input Tensor along its dimensions
 * @note: This will always return a view of the input Tensor, sharing the
 * underlying storage
 * @param tensor The input Tensor
 * @param dims The permutation of the input tensor dimensions the new Tensor
 * should take
 * @return View of the permuted Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto permute(const Tensor &tensor, const std::vector<int> &dims) -> Tensor;

/**
 * Repeat elements consecutively along a given dimension
 * @note See
 * https://pytorch.org/docs/stable/generated/torch.repeat_interleave.html
 * @note This returns a new Tensor with separate underlying storage
 * @param tensor The Tensor to repeat
 * @param repeats The number of times to repeat
 * @param dim The dimension along which to repeat
 * @return The repeated Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto repeat_interleave(const Tensor &tensor, int repeats, int dim) -> Tensor;

/**
 * Repeats the entire Tensor for each amount given over each dimension
 * @note The length of repeats match the number of dimensions of the given
 * Tensor
 * @note See https://pytorch.org/docs/stable/generated/torch.Tensor.repeat.html
 * @note This returns a new Tensor with separate underlying storage
 * @param tensor The Tensor to repeat
 * @param repeats The number of times to repeat along each dimension
 * @return The repeated Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto repeat(const Tensor &tensor, const Tensor &repeats) -> Tensor;

/**
 * Repeats the entire Tensor for each amount given over each dimension
 * @note The length of repeats match the number of dimensions of the given
 * Tensor
 * @note See https://pytorch.org/docs/stable/generated/torch.Tensor.repeat.html
 * @note This returns a new Tensor with separate underlying storage
 * @param tensor The Tensor to repeat
 * @param repeats The number of times to repeat along each dimension
 * @return The repeated Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto repeat(const Tensor &tensor, const std::vector<int> &repeats) -> Tensor;

/**
 * Gather the input Tensor values using the indices along a given dimension
 * @note The number of dimensions of the input tensor and indices must match
 * @param tensor The input Tensor
 * @param indices The indices with respect to the dim to select from
 * @param dim The dimension along which to select
 * @return the gathered tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto gather(const Tensor &tensor, const Tensor &indices, int dim) -> Tensor;

// ------------------------------------------------
// Indexing Operators
// ------------------------------------------------

/**
 * Get the Tensor values at locations given by indices if RHS is an integral
 * type, or where RHS is true if it has kBoolean type.
 * @note This returns a new Tensor with separate underlying storage
 * @note The result is a flat Tensor
 * @param tensor The Tensor to index into
 * @param rhs An tensor of indices or a boolean mask
 * @return The indexed Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto index(const Tensor &tensor, const Tensor &rhs) -> Tensor;

/**
 * Get the resulting Tensor from indexing along a given dim
 * @note This returns a new Tensor with separate underlying storage
 * @param tensor The Tensor to index into
 * @param indices The indices to index
 * @param dim The dimension along which to index
 * @return The indexed Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto index_select(const Tensor &tensor, const Tensor &indices, int dim) -> Tensor;

/**
 * Get the resulting Tensor from indexing along a given dim
 * @note This returns a new Tensor with separate underlying storage
 * @param tensor The Tensor to index into
 * @param indices The indices to index
 * @param dim The dimension along which to index
 * @return The indexed Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto index_select(const Tensor &tensor, const std::vector<int> &indices, int dim)
    -> Tensor;

/**
 * Set the Tensor values in-place at locations given by indices if RHS is an
 * integral type, or where RHS is true if it has kBoolean type
 * @param self The self tensor to update the data
 * @param rhs An tensor of indices or a boolean mask
 * @param value The value to set
 * @return Tensor with updated storage values
 */
[[nodiscard]] TINYTENSOR_EXPORT auto index_put(const Tensor &self, const Tensor &rhs, const Scalar &value) -> Tensor;

/**
 * Set the Tensor values in-place at locations given by indices if RHS is an
 * integral type, or where RHS is true if it has kBoolean type
 * @note The number of elements of values must match the number of elements in
 * rhs
 * @param self The self tensor to update the data
 * @param rhs An tensor of indices or a boolean mask
 * @param values The values to set
 * @return Tensor with updated storage values
 */
[[nodiscard]] TINYTENSOR_EXPORT auto index_put(const Tensor &self, const Tensor &rhs, const Tensor &values) -> Tensor;

/**
 * Set the Tensor values in-place at locations given by indices if RHS is an
 * integral type, or where RHS is true if it has kBoolean type
 * @param rhs An tensor of indices or a boolean mask
 * @param value The value to set
 */
template <IsScalarType T>
[[nodiscard]] TINYTENSOR_EXPORT auto index_put(const Tensor &self, const Tensor &rhs, T value) -> Tensor {
    return index_put(self, rhs, Scalar(value, self.dtype()));
}

// ------------------------------------------------
// Binary Operators
// ------------------------------------------------

// @NOTE: An operator taking a T and a tensor probably wants to cast T to the
// tensors type i.e. tensor(float) + double -> tensor(float) But, we often want
// to do something like T * mask(bool), which shouldn't cast T to bool So we
// handle the bool as a special case

// NOLINTNEXTLINE(*-macro-usage)
#define DECLARE_BINARY_OPS(FUNC)                                                               \
    [[nodiscard]] TINYTENSOR_EXPORT auto FUNC(const Tensor &lhs, const Tensor &rhs) -> Tensor; \
    [[nodiscard]] TINYTENSOR_EXPORT auto FUNC(const Tensor &lhs, Scalar rhs) -> Tensor;        \
    [[nodiscard]] TINYTENSOR_EXPORT auto FUNC(Scalar lhs, const Tensor &rhs) -> Tensor;        \
    template <IsScalarType T>                                                                  \
    [[nodiscard]] TINYTENSOR_EXPORT inline auto FUNC(const Tensor &lhs, T rhs) -> Tensor {     \
        Scalar s = lhs.dtype() == kBool ? cast_to_default(rhs) : Scalar(rhs, lhs.dtype());     \
        return FUNC(lhs, s);                                                                   \
    }                                                                                          \
    template <IsScalarType T>                                                                  \
    [[nodiscard]] TINYTENSOR_EXPORT auto FUNC(T lhs, const Tensor &rhs) -> Tensor {            \
        Scalar s = rhs.dtype() == kBool ? cast_to_default(lhs) : Scalar(lhs, rhs.dtype());     \
        return FUNC(s, rhs);                                                                   \
    }

// NOLINTNEXTLINE(*-macro-usage)
#define DECLARE_BINARY_GLOBAL_OPS(OP, FUNC)                                                                 \
    DECLARE_BINARY_OPS(FUNC)                                                                                \
    template <IsScalarType T>                                                                               \
    [[nodiscard]] TINYTENSOR_EXPORT auto operator OP(const Tensor &lhs, T rhs)->Tensor {                    \
        return FUNC(lhs, rhs);                                                                              \
    }                                                                                                       \
    template <IsScalarType T>                                                                               \
    [[nodiscard]] TINYTENSOR_EXPORT auto operator OP(T lhs, const Tensor &rhs)->Tensor {                    \
        return FUNC(lhs, rhs);                                                                              \
    }                                                                                                       \
    [[nodiscard]] TINYTENSOR_EXPORT inline auto operator OP(const Tensor &lhs, Scalar rhs)->Tensor {        \
        return FUNC(lhs, rhs);                                                                              \
    }                                                                                                       \
    [[nodiscard]] TINYTENSOR_EXPORT inline auto operator OP(Scalar lhs, const Tensor &rhs)->Tensor {        \
        return FUNC(lhs, rhs);                                                                              \
    }                                                                                                       \
    [[nodiscard]] TINYTENSOR_EXPORT inline auto operator OP(const Tensor &lhs, const Tensor &rhs)->Tensor { \
        return FUNC(lhs, rhs);                                                                              \
    }

// Global binary operators
DECLARE_BINARY_GLOBAL_OPS(==, eq);
DECLARE_BINARY_GLOBAL_OPS(!=, ne);
DECLARE_BINARY_GLOBAL_OPS(<, lt);
DECLARE_BINARY_GLOBAL_OPS(<=, le);
DECLARE_BINARY_GLOBAL_OPS(>, gt);
DECLARE_BINARY_GLOBAL_OPS(>=, ge);
DECLARE_BINARY_GLOBAL_OPS(||, logical_or);
DECLARE_BINARY_GLOBAL_OPS(&&, logical_and);
DECLARE_BINARY_GLOBAL_OPS(|, bitwise_or);
DECLARE_BINARY_GLOBAL_OPS(&, bitwise_and);
DECLARE_BINARY_GLOBAL_OPS(^, bitwise_xor);
DECLARE_BINARY_GLOBAL_OPS(<<, bitwise_left_shift);
DECLARE_BINARY_GLOBAL_OPS(>>, bitwise_right_shift);
DECLARE_BINARY_GLOBAL_OPS(%, modulo);
DECLARE_BINARY_GLOBAL_OPS(+, add);
DECLARE_BINARY_GLOBAL_OPS(-, sub);
DECLARE_BINARY_GLOBAL_OPS(*, mul);
DECLARE_BINARY_GLOBAL_OPS(/, div);
#undef DECLARE_BINARY_GLOBAL_OPS

// Additional binary operators
DECLARE_BINARY_OPS(maximum);
DECLARE_BINARY_OPS(minimum);
DECLARE_BINARY_OPS(pow);
#undef DECLARE_BINARY_OPS

/**
 * If both lhs and rhs are 1-dimensional, result is the dot product
 * If lhs is 1-dimensional and rhs is 2-dimensional, result is scalar-matrix
 * product If lhs is 2-dimensional and rhs is 1-dimensional, result is
 * matrix-scalar product If both lhs and rhs are 2-dimensional, result is
 * matrix-matrix product If both lhs and rhs are 3-dimensional, result is
 * batched matrix-matrix product
 * @param lhs LHS Tensor
 * @param rhs RHS Tensor
 * @return Result of matmul operation
 */
[[nodiscard]] TINYTENSOR_EXPORT auto matmul(const Tensor &lhs, const Tensor &rhs) -> Tensor;

// ------------------------------------------------
// Reduction operations
// ------------------------------------------------

/**
 * Returns the minimum of all elements of the input Tensor
 * @param tensor The input Tensor
 * @return the resulting minimum as a Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto min(const Tensor &tensor) -> Tensor;

/**
 * Returns the minimum of each element over a dimension of the input Tensor
 * @param tensor The input Tensor
 * @param dim The dimension to take the minimum over
 * @param keep_dim Whether the result has the reduction dimension retained or
 * not
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto min(const Tensor &tensor, int dim, bool keep_dim = false) -> Tensor;

/**
 * Returns the minimum of each element over a dimension of the input Tensor
 * @param tensor The input Tensor
 * @param dims The dimensions to take the minimum over
 * @param keep_dim Whether the result has the reduction dimension retained or
 * not
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto min(const Tensor &tensor, const std::vector<int> &dims, bool keep_dim = false)
    -> Tensor;

/**
 * Returns the index of the minimum of all elements of the input Tensor
 * @param tensor The input Tensor
 * @return the resulting index of the minimum as a Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto argmin(const Tensor &tensor) -> Tensor;

/**
 * Returns the indices of the minimum elements over a dimension of the input
 * Tensor
 * @param tensor The input Tensor
 * @param dim The dimension to take the index of minimums over
 * @param keep_dim Whether the result has the reduction dimension retained or
 * not
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto argmin(const Tensor &tensor, int dim, bool keep_dim = false) -> Tensor;

/**
 * Returns the indices of the minimum elements over a dimension of the input
 * Tensor
 * @param tensor The input Tensor
 * @param dims The dimensions to take the index of minimums over
 * @param keep_dim Whether the result has the reduction dimension retained or
 * not
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto argmin(const Tensor &tensor, const std::vector<int> &dims, bool keep_dim = false)
    -> Tensor;

/**
 * Returns the maximum of all elements of the input Tensor
 * @param tensor The input Tensor
 * @return the resulting maximum as a Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto max(const Tensor &tensor) -> Tensor;

/**
 * Returns the maximum of each element over a dimension of the input Tensor
 * @param tensor The input Tensor
 * @param dim The dimension to take the maximum over
 * @param keep_dim Whether the result has the reduction dimension retained or
 * not
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto max(const Tensor &tensor, int dim, bool keep_dim = false) -> Tensor;

/**
 * Returns the maximum of each element over a dimension of the input Tensor
 * @param tensor The input Tensor
 * @param dims The dimensions to take the maximum over
 * @param keep_dim Whether the result has the reduction dimension retained or
 * not
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto max(const Tensor &tensor, const std::vector<int> &dims, bool keep_dim = false)
    -> Tensor;

/**
 * Returns the index of the maximum of all elements of the input Tensor
 * @param tensor The input Tensor
 * @return the resulting index of the maximum as a Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto argmax(const Tensor &tensor) -> Tensor;

/**
 * Returns the indices of the maximum elements over a dimension of the input
 * Tensor
 * @param tensor The input Tensor
 * @param dim The dimension to take the index of maximums over
 * @param keep_dim Whether the result has the reduction dimension retained or
 * not
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto argmax(const Tensor &tensor, int dim, bool keep_dim = false) -> Tensor;

/**
 * Returns the indices of the maximum elements over a dimension of the input
 * Tensor
 * @param tensor The input Tensor
 * @param dims The dimensions to take the index of maximums over
 * @param keep_dim Whether the result has the reduction dimension retained or
 * not
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto argmax(const Tensor &tensor, const std::vector<int> &dims, bool keep_dim = false)
    -> Tensor;

/**
 * Returns the sum of all elements of the input Tensor
 * @param tensor The input Tensor
 * @return the resulting sum as a Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto sum(const Tensor &tensor) -> Tensor;

/**
 * Returns the sum of each element over a dimension of the input Tensor
 * @param tensor The input Tensor
 * @param dim The dimension to take the sum over
 * @param keep_dim Whether the result has the reduction dimension retained or
 * not
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto sum(const Tensor &tensor, int dim, bool keep_dim = false) -> Tensor;

/**
 * Returns the sum of each element over a dimension of the input Tensor
 * @param tensor The input Tensor
 * @param dims The dimensions to take the sum over
 * @param keep_dim Whether the result has the reduction dimension retained or
 * not
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto sum(const Tensor &tensor, const std::vector<int> &dims, bool keep_dim = false)
    -> Tensor;

/**
 * Returns the mean of all elements of the input Tensor
 * @param tensor The input Tensor
 * @return the resulting mean as a Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto mean(const Tensor &tensor) -> Tensor;

/**
 * Returns the mean of each element over a dimension of the input Tensor
 * @param tensor The input Tensor
 * @param dim The dimension to take the mean over
 * @param keep_dim Whether the result has the reduction dimension retained or
 * not
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto mean(const Tensor &tensor, int dim, bool keep_dim = false) -> Tensor;

/**
 * Returns the mean of each element over a dimension of the input Tensor
 * @param tensor The input Tensor
 * @param dims The dimensions to take the mean over
 * @param keep_dim Whether the result has the reduction dimension retained or
 * not
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto mean(const Tensor &tensor, const std::vector<int> &dims, bool keep_dim = false)
    -> Tensor;

/**
 * Tests if all elements evaluate to true in the input Tensor
 * @param tensor The input Tensor
 * @return True if all elements evaluate to True, False otherwise
 */
[[nodiscard]] TINYTENSOR_EXPORT auto all(const Tensor &input) -> bool;

/**
 * Tests if all elements evaluate to true over a dimension of the input Tensor
 * @note The type of the returned Tensor is kBool
 * @param tensor The input Tensor
 * @param dim The dimension to take the reduce over
 * @param keep_dim Whether the result has the reduction dimension retained or
 * not
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto all(const Tensor &tensor, int dim, bool keep_dim = false) -> Tensor;

/**
 * Tests if all elements evaluate to true over a dimension of the input Tensor
 * @note The type of the returned Tensor is kBool
 * @param tensor The input Tensor
 * @param dims The dimensions to take the reduce over
 * @param keep_dim Whether the result has the reduction dimension retained or
 * not
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto all(const Tensor &tensor, const std::vector<int> &dims, bool keep_dim = false)
    -> Tensor;

/**
 * Tests if any elements evaluate to true in the input Tensor
 * @param tensor The input Tensor
 * @return True if any element evaluates to True, False otherwise
 */
[[nodiscard]] TINYTENSOR_EXPORT auto any(const Tensor &input) -> bool;

/**
 * Tests if any element evaluates to true over a dimension of the input Tensor
 * @note The type of the returned Tensor is kBool
 * @param tensor The input Tensor
 * @param dim The dimension to take the reduce over
 * @param keep_dim Whether the result has the reduction dimension retained or
 * not
 * @param correction Difference between sample size and sampel degrees of
 * freedom
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto any(const Tensor &tensor, int dim, bool keep_dim = false) -> Tensor;

/**
 * Tests if any element evaluates to true over a dimension of the input Tensor
 * @note The type of the returned Tensor is kBool
 * @param tensor The input Tensor
 * @param dims The dimensions to take the reduce over
 * @param keep_dim Whether the result has the reduction dimension retained or
 * not
 * @param correction Difference between sample size and sampel degrees of
 * freedom
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto any(const Tensor &tensor, const std::vector<int> &dims, bool keep_dim = false)
    -> Tensor;

/**
 * Compute the variance of the input
 * @param input The input Tensor
 * @param keep_dim Whether the result has the reduction dimension retained or
 * not
 * @param correction Difference between sample size and sampel degrees of
 * freedom
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto var(const Tensor &input, bool keep_dim = false, int correction = 1) -> Tensor;

/**
 * Compute the variance of the input over a given dimension
 * @param input The input Tensor
 * @param dim The dimension to reduce over
 * @param keep_dim Whether the result has the reduction dimension retained or
 * not
 * @param correction Difference between sample size and sampel degrees of
 * freedom
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto var(const Tensor &input, int dim, bool keep_dim = false, int correction = 1)
    -> Tensor;

/**
 * Compute the variance of the input over a given dimension
 * @param input The input Tensor
 * @param dims The dimensions to reduce over
 * @param keep_dim Whether the result has the reduction dimension retained or
 * not
 * @param correction Difference between sample size and sampel degrees of
 * freedom
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto
    var(const Tensor &input, const std::vector<int> &dims, bool keep_dim = false, int correction = 1) -> Tensor;

// ------------------------------------------------
// Unary operations
// ------------------------------------------------

/**
 * Returns the element-wise absolute of the input Tensor
 * @param tensor The input Tensor
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto abs(const Tensor &tensor) -> Tensor;

/**
 * Returns the element-wise negation of the input Tensor
 * @param tensor The input Tensor
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto negate(const Tensor &tensor) -> Tensor;

/**
 * Returns the element-wise logical not of the input Tensor
 * @note The returned type of the tensor is boolean
 * @note This function does not have autograd support
 * @param tensor The input Tensor
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto logical_not(const Tensor &tensor) -> Tensor;

/**
 * Returns the element-wise sign of the input Tensor
 * @note see https://en.wikipedia.org/wiki/Sign_function
 * @param tensor The input Tensor
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto sign(const Tensor &tensor) -> Tensor;

/**
 * Returns the element-wise natural logarithm of the input Tensor
 * @param tensor The input Tensor
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto log(const Tensor &tensor) -> Tensor;

/**
 * Returns the element-wise base 10 logarithm of the input Tensor
 * @param tensor The input Tensor
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto log10(const Tensor &tensor) -> Tensor;

/**
 * Returns the element-wise base 2 logarithm of the input Tensor
 * @param tensor The input Tensor
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto log2(const Tensor &tensor) -> Tensor;

/**
 * Returns the element-wise natural logarithm of one plus the input Tensor
 * @param tensor The input Tensor
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto log1p(const Tensor &tensor) -> Tensor;

/**
 * Returns the element-wise exponential of the input Tensor
 * @param tensor The input Tensor
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto exp(const Tensor &tensor) -> Tensor;

/**
 * Returns the element-wise base 2 exponential of the input Tensor
 * @param tensor The input Tensor
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto exp2(const Tensor &tensor) -> Tensor;

/**
 * Returns the element-wise exp(x)-1 of the input Tensor
 * @param tensor The input Tensor
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto expm1(const Tensor &tensor) -> Tensor;

/**
 * Returns the element-wise square root of the input Tensor
 * @param tensor The input Tensor
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto sqrt(const Tensor &tensor) -> Tensor;

/**
 * Returns the element-wise sine of the input Tensor
 * @param tensor The input Tensor
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto sin(const Tensor &tensor) -> Tensor;

/**
 * Returns the element-wise cosine of the input Tensor
 * @param tensor The input Tensor
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto cos(const Tensor &tensor) -> Tensor;

/**
 * Returns the element-wise tangent of the input Tensor
 * @param tensor The input Tensor
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto tan(const Tensor &tensor) -> Tensor;

/**
 * Returns the element-wise trigonometric inverse sine of the input Tensor
 * @param tensor The input Tensor
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto asin(const Tensor &tensor) -> Tensor;

/**
 * Returns the element-wise trigonometric inverse cosine of the input Tensor
 * @param tensor The input Tensor
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto acos(const Tensor &tensor) -> Tensor;

/**
 * Returns the element-wise trigonometric inverse tangent of the input Tensor
 * @param tensor The input Tensor
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto atan(const Tensor &tensor) -> Tensor;

/**
 * Returns the element-wise hyperbolic sine of the input Tensor
 * @param tensor The input Tensor
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto sinh(const Tensor &tensor) -> Tensor;

/**
 * Returns the element-wise hyperbolic cosine of the input Tensor
 * @param tensor The input Tensor
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto cosh(const Tensor &tensor) -> Tensor;

/**
 * Returns the element-wise hyperbolic tangent of the input Tensor
 * @param tensor The input Tensor
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto tanh(const Tensor &tensor) -> Tensor;

/**
 * Returns the element-wise inverse hyperbolic sine of the input Tensor
 * @param tensor The input Tensor
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto asinh(const Tensor &tensor) -> Tensor;

/**
 * Returns the element-wise inverse hyperbolic cosine of the input Tensor
 * @param tensor The input Tensor
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto acosh(const Tensor &tensor) -> Tensor;

/**
 * Returns the element-wise inverse hyperbolic tangent of the input Tensor
 * @param tensor The input Tensor
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto atanh(const Tensor &tensor) -> Tensor;

/**
 * Returns the element-wise error function of the input Tensor
 * @note https://en.wikipedia.org/wiki/Error_function
 * @param tensor The input Tensor
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto erf(const Tensor &tensor) -> Tensor;

/**
 * Returns the element-wise complementary error function of the input Tensor
 * @note https://en.wikipedia.org/wiki/Error_function
 * @param tensor The input Tensor
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto erfc(const Tensor &tensor) -> Tensor;

/**
 * Returns the element-wise gamma function of the input Tensor
 * @note https://en.wikipedia.org/wiki/Gamma_function
 * @param tensor The input Tensor
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto tgamma(const Tensor &tensor) -> Tensor;

/**
 * Returns the element-wise logarithm of the gamma function of the input Tensor
 * @note https://en.wikipedia.org/wiki/Gamma_function
 * @param tensor The input Tensor
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto lgamma(const Tensor &tensor) -> Tensor;

/**
 * Returns the element-wise digamma function (derivative of log gamma) of the
 * input Tensor
 * @note https://en.wikipedia.org/wiki/Digamma_function
 * @note This function does not have autograd support
 * @param tensor The input Tensor
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto digamma(const Tensor &tensor) -> Tensor;

/**
 * Returns the element-wise ceiling of the input Tensor
 * @note This function does not have autograd support
 * @param tensor The input Tensor
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto ceil(const Tensor &tensor) -> Tensor;

/**
 * Returns the element-wise floor of the input Tensor
 * @note This function does not have autograd support
 * @param tensor The input Tensor
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto floor(const Tensor &tensor) -> Tensor;

/**
 * Returns the element-wise rounding to nearest integer of the input Tensor
 * @note Rounding halfway cases away from zero
 * @note This function does not have autograd support
 * @param tensor The input Tensor
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto round(const Tensor &tensor) -> Tensor;

/**
 * Returns the element-wise boolean check if value is positive or negative
 * infinity of the input Tensor
 * @note Result Tensor is of type kBool
 * @param tensor The input Tensor
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto isinf(const Tensor &tensor) -> Tensor;

/**
 * Returns the element-wise boolean check if value is not-a-number of the input
 * Tensor
 * @note This function does not have autograd support
 * @note Result Tensor is of type kBool
 * @param tensor The input Tensor
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto isnan(const Tensor &tensor) -> Tensor;

/**
 * Returns the element-wise boolean check if value is finite of the input Tensor
 * @note This function does not have autograd support
 * @note Values are true except for positive/negative infinity or NaNs
 * @note Result Tensor is of type kBool
 * @param tensor The input Tensor
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto isfinite(const Tensor &tensor) -> Tensor;

// Special unary ops
TINYTENSOR_EXPORT auto operator-(const Tensor &tensor) -> Tensor;
TINYTENSOR_EXPORT auto operator!(const Tensor &tensor) -> Tensor;

// ------------------------------------------------
// Activation Functions
// ------------------------------------------------

/**
 * Returns the element-wise sigmoid of the input Tensor
 * @note https://en.wikipedia.org/wiki/Sigmoid_function
 * @param tensor The input Tensor
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto sigmoid(const Tensor &tensor) -> Tensor;

/**
 * Returns the element-wise log-sigmoid of the input Tensor
 * @note https://en.wikipedia.org/wiki/Sigmoid_function
 * @param tensor The input Tensor
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto log_sigmoid(const Tensor &tensor) -> Tensor;

/**
 * Returns the element-wise Hardsigmoid of the input Tensor
 * @note https://pytorch.org/docs/stable/generated/torch.nn.Hardsigmoid.html
 * @param tensor The input Tensor
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto hardsigmoid(const Tensor &tensor) -> Tensor;

/**
 * Returns the element-wise softplus of the input Tensor
 * @note Softplus is a smooth approximation of the ReLU function (always
 * positive)
 * @note https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html
 * @param tensor The input Tensor
 * @param beta The beta value for Softplus
 * @param threshold Values above this revert to a linear function
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto softplus(const Tensor &tensor, double beta = 1, double threshold = 20) -> Tensor;

/**
 * Returns the element-wise rectified linear unit function of the input Tensor
 * @note https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
 * @param tensor The input Tensor
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto relu(const Tensor &tensor) -> Tensor;

/**
 * Returns the element-wise ReLU6 of the input Tensor
 * @note https://pytorch.org/docs/stable/generated/torch.nn.ReLU6.html
 * @param tensor The input Tensor
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto relu6(const Tensor &tensor) -> Tensor;

/**
 * Returns the element-wise LeakyReLU of the input Tensor
 * @note https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html
 * @param tensor The input Tensor
 * @param negative_slop The cangle of the negative slope, used for negative
 * inputs
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto leaky_relu(const Tensor &tensor, double negative_slope = 0.01) -> Tensor;

/**
 * Returns the element-wise Exponential Linear Unit (ELU) of the input Tensor
 * @note https://pytorch.org/docs/stable/generated/torch.nn.ELU.html
 * @param tensor The input Tensor
 * @param alpha The alpha value for the ELU formulation
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto elu(const Tensor &tensor, double alpha = 1) -> Tensor;

/**
 * Returns the element-wise SELU of the input Tensor
 * @note https://pytorch.org/docs/stable/generated/torch.nn.SELU.html
 * @param tensor The input Tensor
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto selu(const Tensor &tensor) -> Tensor;

/**
 * Returns the element-wise Sigmoid Linear Unit (SiLU) of the input Tensor
 * @note https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html
 * @param tensor The input Tensor
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto silu(const Tensor &tensor) -> Tensor;

/**
 * Returns the element-wise HardTanh of the input Tensor
 * @note https://pytorch.org/docs/stable/generated/torch.nn.Hardtanh.html
 * @param tensor The input Tensor
 * @param min Minimum value of the linear region range
 * @param max Maximum value of the linear region range
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto hardtanh(const Tensor &tensor, double min = -1, double max = 1) -> Tensor;

/**
 * Returns the element-wise Softsign of the input Tensor
 * @note https://pytorch.org/docs/stable/generated/torch.nn.Softsign.html
 * @param tensor The input Tensor
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto softsign(const Tensor &tensor) -> Tensor;

/**
 * Returns the element-wise softmax over a dimension of the input Tensor
 * @note The elements along the given dimension are rescaled so that they lie in
 * the range [0,1] and sum to 1
 * @note https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
 * @param tensor The input Tensor
 * @param dim The dimension along which the softmax will be computed
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto softmax(const Tensor &tensor, int dim) -> Tensor;

/**
 * Returns the element-wise log-softmax over a dimension of the input Tensor
 * @note https://pytorch.org/docs/stable/generated/torch.nn.LogSoftmax.html
 * @param tensor The input Tensor
 * @param dim The dimension along which the log softmax will be computed
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto log_softmax(const Tensor &tensor, int dim) -> Tensor;

// ------------------------------------------------
// Util/misc
// ------------------------------------------------

/**
 * Save the given tensor
 * @param path The full path for the file to save into
 * @param tensor The tensor to save
 */
TINYTENSOR_EXPORT void save(const std::string &path, const Tensor &tensor);

/**
 * Load a tensor from a given path
 * @note This will always load the tensor on the CPU, regardless of what device
 * the saved tensor was on
 * @param path The full path of the file containing the serialized tensor
 * @return The loaded tensor
 */
TINYTENSOR_EXPORT auto load(const std::string &path) -> Tensor;

/**
 * Returns a tensor where elements are from lhs if cond is true, rhs otherwise
 * element-wise
 * @param cond A boolean tensor for the condition
 * @param lhs Values to use if the cond is true
 * @param rhs Values to use if the cond is false
 * @return the resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto where(const Tensor &cond, const Tensor &lhs, const Tensor &rhs) -> Tensor;
[[nodiscard]] TINYTENSOR_EXPORT auto where(const Tensor &cond, Scalar lhs, Scalar rhs) -> Tensor;
[[nodiscard]] TINYTENSOR_EXPORT auto where(const Tensor &cond, const Tensor &lhs, Scalar rhs) -> Tensor;
[[nodiscard]] TINYTENSOR_EXPORT auto where(const Tensor &cond, Scalar lhs, const Tensor &rhs) -> Tensor;

/**
 * Returns a tensor where elements are from lhs if cond is true, rhs otherwise
 * element-wise
 * @param cond A boolean tensor for the condition
 * @param lhs Values to use if the cond is true
 * @param rhs Values to use if the cond is false
 * @return the resulting Tensor
 */
template <IsScalarType T>
[[nodiscard]] TINYTENSOR_EXPORT auto where(const Tensor &cond, T lhs, T rhs) -> Tensor {
    return where(cond, Scalar(lhs), Scalar(rhs));
}
template <IsScalarType T>
[[nodiscard]] TINYTENSOR_EXPORT auto where(const Tensor &cond, const Tensor &lhs, T rhs) -> Tensor {
    return where(cond, lhs, Scalar(rhs));
}
template <IsScalarType T>
[[nodiscard]] TINYTENSOR_EXPORT auto where(const Tensor &cond, T lhs, const Tensor &rhs) -> Tensor {
    return where(cond, Scalar(lhs), rhs);
}

// Auxiliary options for close checks
class TINYTENSOR_EXPORT CloseOptions {
public:
    auto rtol(double tol) -> CloseOptions &;
    auto atol(double tol) -> CloseOptions &;
    auto equal_nan() -> CloseOptions &;
    [[nodiscard]] inline auto rtol() const -> double {
        return _rtol;
    }
    [[nodiscard]] inline auto atol() const -> double {
        return _atol;
    }
    [[nodiscard]] inline auto equal_nan() const -> bool {
        return _equal_nan;
    }

private:
    double _rtol = 1e-5;
    double _atol = 1e-8;
    bool _equal_nan = false;
};

/**
 * Returns a Boolea Tensor where each element represents if the corresponding
 * element in lhs is close to rhs NaNs are considered equal to each other when
 * equal_nan is true
 * @note Shape and device of Tensors must match
 * @note Close is defined as |lhs - rhs| <= atol + rtol x |other|
 * @note See https://pytorch.org/docs/stable/generated/torch.isclose.html
 * @param lhs First Tensor to compare
 * @param rhs Second Tensor to compare
 * @options Contains values for atol, rtol, and equal_nan
 * @return Resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto
    isclose(const Tensor &lhs, const Tensor &rhs, const CloseOptions &options = CloseOptions()) -> Tensor;

/**
 * Returns a Boolea Tensor where each element represents if the corresponding
 * element in lhs is close to rhs NaNs are considered equal to each other when
 * equal_nan is true
 * @note Close is defined as |lhs - rhs| <= atol + rtol x |other|
 * @note See https://pytorch.org/docs/stable/generated/torch.isclose.html
 * @param lhs Tensor to compare
 * @param rhs Value to compare against lhs
 * @options Contains values for atol, rtol, and equal_nan
 * @return Resulting Tensor
 */
template <IsScalarType T>
[[nodiscard]] TINYTENSOR_EXPORT auto isclose(const Tensor &lhs, T rhs, const CloseOptions &options = CloseOptions())
    -> Tensor {
    return isclose(lhs, Tensor(Scalar(rhs), lhs.device()).expand(lhs.shape()), options);
}

/**
 * Returns a Boolea Tensor where each element represents if the corresponding
 * element in lhs is close to rhs NaNs are considered equal to each other when
 * equal_nan is true
 * @note Close is defined as |lhs - rhs| <= atol + rtol x |other|
 * @note See https://pytorch.org/docs/stable/generated/torch.isclose.html
 * @param lhs Value against rhs to compare
 * @param rhs Tensor to compare
 * @options Contains values for atol, rtol, and equal_nan
 * @return Resulting Tensor
 */
template <IsScalarType T>
[[nodiscard]] TINYTENSOR_EXPORT auto isclose(T lhs, const Tensor &rhs, const CloseOptions &options = CloseOptions())
    -> Tensor {
    return isclose(Tensor(Scalar(lhs), rhs.device()).expand(rhs.shape()), rhs, options);
}

/**
 * Returns true if each element in input is close to other, equivalent to
 * boolean reduction of isclose NaNs are considered equal to each other when
 * equal_nan is true
 * @note Close is defined as |lhs - rhs| <= atol + rtol x |other|
 * @note See https://pytorch.org/docs/stable/generated/torch.isclose.html
 * @param input First Tensor to compare
 * @param other Second Tensor to compare
 * @options Contains values for atol, rtol, and equal_nan
 * @return Resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto
    allclose(const Tensor &input, const Tensor &other, const CloseOptions &options = CloseOptions()) -> bool;

/**
 * Clamps all elements into the range [min, max] given by the ClampOptions
 * @param input The input Tensor
 * @param options Clamp options holding min/max values
 * @return Resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto clamp(const Tensor &input, const ClampOptions &options = ClampOptions()) -> Tensor;

/**
 * Clamps element-wise into the range [min, max]
 * @param input The input Tensor
 * @param min The Lower-bound for each element
 * @param max The Upper-bound for each element
 * @return Resulting Tensor
 */
[[nodiscard]] TINYTENSOR_EXPORT auto clamp(const Tensor &input, const Tensor &min, const Tensor &max) -> Tensor;

/**
 * Performs a 2D convolution using filters of size (K, K) over the input. The
 * kernel size is determined by the shape of the given weight Tensor
 * @param input The input 4D image (batch, in_channels, height, width)
 * @param weight The filters of shape (out_channels, in_channels, K, K)
 * @param bias An optional bias of shape (out_channels) to apply to each channel
 * of the output
 * @param stride The stide to apply of the convolving kernel
 * @param padding The mount of zero padding to apply around all side of the
 * input before applying the convolution
 * @return The result of the convolution
 */
[[nodiscard]] TINYTENSOR_EXPORT auto conv2d(
    const Tensor &input,
    const Tensor &weight,
    const std::optional<Tensor> &bias = {},
    int stride = 1,
    int padding = 0
) -> Tensor;

/**
 * Performs a 2D max-pooling opreation in (kernel_size, kernel_size) regions by
 * (stride, stride) steps. The number of output features is equal to the number
 * of input features
 * @param input The input 4d image (batch, in_channels, heigh, width)
 * @param kernel_size The size of the pooling region
 * @param stride The stride of the pooling opreation
 * @param padding The mount of zero padding to apply around all side of the
 * input before applying the pooling
 * @return The result of the pooling
 */
[[nodiscard]] TINYTENSOR_EXPORT auto max_pool2d(const Tensor &input, int kernel_size, int stride, int padding = 0)
    -> Tensor;

/**
 * Performs a 2D min-pooling opreation in (kernel_size, kernel_size) regions by
 * (stride, stride) steps. The number of output features is equal to the number
 * of input features
 * @param input The input 4d image (batch, in_channels, heigh, width)
 * @param kernel_size The size of the pooling region
 * @param stride The stride of the pooling opreation
 * @param padding The mount of zero padding to apply around all side of the
 * input before applying the pooling
 * @return The result of the pooling
 */
[[nodiscard]] TINYTENSOR_EXPORT auto min_pool2d(const Tensor &input, int kernel_size, int stride, int padding = 0)
    -> Tensor;

/**
 * Performs a 2D average-pooling opreation in (kernel_size, kernel_size) regions
 * by (stride, stride) steps. The number of output features is equal to the
 * number of input features
 * @param input The input 4d image (batch, in_channels, heigh, width)
 * @param kernel_size The size of the pooling region
 * @param stride The stride of the pooling opreation
 * @param padding The mount of zero padding to apply around all side of the
 * input before applying the pooling
 * @return The result of the pooling
 */
[[nodiscard]] TINYTENSOR_EXPORT auto avg_pool2d(const Tensor &input, int kernel_size, int stride, int padding = 0)
    -> Tensor;

/**
 * An embedding lookup table call
 * @input input Tensor of indices into the weight table
 * @input weight The embedding matrix of shape (D, embedding_dim) where D is the
 * maximum index size + 1
 * @return Embedding lookup results of (*input_shape, embedding_dim)
 */
[[nodiscard]] TINYTENSOR_EXPORT auto embedding(const Tensor &input, const Tensor &weight) -> Tensor;

/**
 * Get the current memory allocated for a given device
 * @param device The device backend to query
 * @return The current bytes allocated
 */
[[nodiscard]] TINYTENSOR_EXPORT auto current_memory_allocated(const Device &device) -> uint64_t;

/**
 * Get the total memory allocated throughout the program lifetime for a given
 * device
 * @param device The device backend to query
 * @return The total bytes allocated
 */
[[nodiscard]] TINYTENSOR_EXPORT auto total_memory_allocated(const Device &device) -> uint64_t;

/**
 * Create a dot graphviz of the ops and tensors for the computation graph up to
 * and including this tensor
 * @tensor The Tensor to build the computation graph from
 */
[[nodiscard]] TINYTENSOR_EXPORT auto make_dot(const Tensor &tensor) -> std::string;

}    // namespace tinytensor

#endif    // TINYTENSOR_TENSOR_H_
