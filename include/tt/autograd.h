// autograd.h
// Types for the autograd engine

#ifndef TINYTENSOR_AUTOGRAD_TYPES_H_
#define TINYTENSOR_AUTOGRAD_TYPES_H_

#include <tt/device.h>
#include <tt/export.h>
#include <tt/grad_mode.h>
#include <tt/index.h>
#include <tt/scalar.h>
#include <tt/shape.h>
#include <tt/tensor.h>

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace tinytensor::autograd {

namespace detail {
// CRTP class to iterate over a variadic collection and apply a function to each argument
// Derived classes should overload operator() for each type it wants to handle, along with the behaviour
template <typename T>
struct TINYTENSOR_EXPORT IterateApply {
    // Base case reaching end of collection
    template <typename... Args>
    T &apply() {
        return self();
    }

    // Call overloaded operator() on element arg then continue with rest of collection
    template <typename Arg, typename... Args>
    T &apply(Arg &&arg, Args &&...args) {
        self()(std::forward<Arg>(arg));
        return apply(std::forward<Args>(args)...);
    }

private:
    // Cast to dervied CRTP type to call its overloaded operator()'s
    inline T &self() {
        return *static_cast<T *>(this);
    }
};

// Extract tensors from a generic set of arguments, so we can track with autograd
class TINYTENSOR_EXPORT ExtractTensors : public IterateApply<ExtractTensors> {
public:
    ExtractTensors(TensorList &tensors)
        : tensors_(tensors) {}
    ExtractTensors(std::vector<Tensor> &&) = delete;    // prevent rvalue binding then immediately being destroyed

    // If a tensor, save it
    void operator()(const Tensor &tensor) {
        tensors_.push_back(tensor);
    }

    // If a vector of tensors, save all of them
    void operator()(const TensorList &tensors) {
        for (const auto &tensor : tensors) {
            tensors_.push_back(tensor);
        }
    }

    // If an optional tensor, save it if contains payload
    void operator()(const std::optional<Tensor> &tensor) {
        if (tensor) {
            tensors_.push_back(*tensor);
        }
    }

    // We don't care about any other type
    template <typename T>
    void operator()(const T &x) {
        (void)x;
    }

private:
    // Non-owning tensor list, used to save results while iterating over collection
    TensorList &tensors_;    // NOLINT(*avoid-const-or-ref-data-members)
};
}    // namespace detail

// Supported types for storage
using VersionedTensor = std::tuple<Tensor, int>;
using VersionedTensorList = std::vector<std::tuple<Tensor, int>>;
using StorageItem = std::variant<
    bool,
    int,
    float,
    double,
    std::vector<int>,
    ScalarType,
    Device,
    Shape,
    Tensor,
    TensorList,
    indexing::IndexList,
    VersionedTensor,
    VersionedTensorList>;
TINYTENSOR_EXPORT inline auto make_versioned_tensor(const Tensor &tensor) -> VersionedTensor {
    return {tensor, tensor.version_count()};
}
TINYTENSOR_EXPORT inline auto make_versioned_tensor_list(const TensorList &tensors) -> VersionedTensorList {
    VersionedTensorList list;
    for (const auto &tensor : tensors) {
        list.emplace_back(tensor, tensor.version_count());
    }
    return list;
}

// Generic storage mapping for autograd aware functions, used to retrive items for backward pass
using AutogradStorage = std::unordered_map<std::string, StorageItem>;

// Backward pass function for gradient computation, takes storage and incomming gradient
// using GradList = std::vector<std::optional<Tensor>>;
using GradList = CheckedVec<std::optional<Tensor>>;
using GradFunc = std::function<GradList(AutogradStorage &storage, const Tensor &grad_output)>;

// Shared grad info between all reference tensors
struct TINYTENSOR_EXPORT SharedGrad {
    AutogradStorage storage;
    TensorList parents;
    std::optional<Tensor> grad = std::nullopt;
    GradFunc grad_func;
    GradHook grad_hook;
    std::string grad_func_name;
    bool requires_grad = false;
};

/**
 * CRTP struct to facilitate autograd functions by tracking the DAG of operations and storage for backward passes
 * @note Autograd aware functions should extend this, and implement a forward() and backward() method
 * Tensor functions can only return tensors
 */
template <typename T>
struct TINYTENSOR_EXPORT TensorFunction {
    // Internal friend getters
    static auto get_shape(Tensor &tensor) -> Shape & {
        return tensor.shape_;
    }
    static auto get_stride(Tensor &tensor) -> Shape & {
        return tensor.stride_;
    }
    static auto get_offset(Tensor &tensor) -> int & {
        return tensor.offset_;
    }
    static auto get_ctx(Tensor &tensor) -> std::shared_ptr<SharedGrad> & {
        return tensor.ctx_;
    }
    static void bump_version(const Tensor &tensor) {
        ++tensor.version_count_;
    }

    // Need type T to get forward_t type, but when TensorFunction<T> is instantiated, T::forward is not yet declared
    template <typename... Args>
    static auto apply(Args &&...args) -> Tensor {
        // autograd functions can have generic arguments, need to extract tensors
        TensorList input_tensors;
        detail::ExtractTensors(input_tensors).apply(std::forward<Args>(args)...);
        bool is_grad_required =
            GradMode::is_enabled() && std::any_of(input_tensors.begin(), input_tensors.end(), [](const Tensor &t) {
                return t.requires_grad();
            });

        AutogradStorage storage;
        std::optional<Tensor> output;
        {
            // Ensure we do not re-enter and create cycles from arithmetic in the forward/backward passes
            const NoGradGuard guard;
            output = T::forward(storage, is_grad_required, std::forward<Args>(args)...);
        }

        // If gradmode enabled and any of the functions arguments requires a gradient, store the necessary context
        // information
        if (is_grad_required) {
            output->ctx_->parents = std::move(input_tensors);
            output->ctx_->storage = std::move(storage);
            output->ctx_->grad_func = T::backward;
            output->ctx_->requires_grad = true;
            output->ctx_->grad_func_name = T::name;
        }
        return *output;
    }
};

}    // namespace tinytensor::autograd

#endif    // TINYTENSOR_AUTOGRAD_TYPES_H_
