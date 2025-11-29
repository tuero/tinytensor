// module.h
// Base module

#ifndef TINYTENSOR_NN_MODULE_H_
#define TINYTENSOR_NN_MODULE_H_

#include <tt/device.h>
#include <tt/exception.h>
#include <tt/export.h>
#include <tt/tensor.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace tinytensor::nn {

class TINYTENSOR_EXPORT Module {
public:
    virtual ~Module() = default;
    Module() = default;

    // @NOTE: Revisit these
    Module(const Module &) = delete;
    Module(Module &&) = default;
    auto operator=(const Module &) -> Module & = delete;
    auto operator=(Module &&) -> Module & = delete;

    /**
     * Get the string name of the module
     * @return the string name
     */
    [[nodiscard]] virtual auto name() const -> std::string = 0;

    /**
     * Pretty print the module
     * @param Most of the time this does not need to be overridden, as the default behaviour will call pretty print for
     * each registered module
     */
    virtual void pretty_print(std::ostream &os) const;

    /**
     * Get the underlying parameters of the model
     * @note The order of the given weights is unspecified and should not be relied on
     * @param recursive True to recursively get all parameters, false for only the current module
     */
    [[nodiscard]] auto parameters(bool recursive = true) const -> std::vector<Tensor>;

    /**
     * Get the underlying parameters of the model, for use with optimizers
     * @note We use reference wrappers to avoid having to write another indirection for the underlying storage, so
     * moving the model to/from devices and performings loads are reflected in the optimizers saved references. As a
     * result, the lifetime of the model parameters must outlife the optimizer
     * @note The order of the given weights is unspecified and should not be relied on
     * @param recursive True to recursively get all parameters, false for only the current module
     */
    [[nodiscard]] auto parameters_for_optimizer(bool recursive = true) const
        -> std::vector<std::reference_wrapper<Tensor>>;

    /**
     * Get the total number of registered parameters
     */
    [[nodiscard]] auto num_params() const -> int64_t;

    /**
     * Serialize the modules weights
     * @note This is useful for synchronizing a model's weights from another instance of the same model
     * @return serialized vector of of weight data
     */
    [[nodiscard]] auto serialize() const -> std::vector<std::vector<char>>;

    /**
     * Deserialize and load the weights from the given data vector
     * @param data The serialized model's data
     */
    void deserialize(const std::vector<std::vector<char>> &data);

    /**
     * Save the weights to a file
     * @param path The path to the weights file
     */
    void save(const std::string &path) const;

    /**
     * Load the weights from a given file
     * @param path The path to the weights file
     */
    void load(const std::string &path);

    /**
     * Move the module onto the device
     * @note This will recursively move all registered modules
     * @param device The device to move to
     */
    void to(Device device);

    /**
     * Zero the gradient recursively for all registered modules
     */
    void zero_grad();

    /**
     * Cast the module reference (or pointer) to the given derived module
     */
    template <typename T>
    auto as() -> T * {
        return dynamic_cast<T *>(this);
    }

    /**
     * Cast the module reference (or pointer) to the given derived module
     * @note If the underlying module is not of the given type, an exception is thrown
     */
    template <typename T>
    auto as_checked() -> T & {
        T *p = dynamic_cast<T *>(this);
        if (p) {
            return *p;
        }
        TT_EXCEPTION("Cannot cast underlying moduel to given type");
    }

    /**
     * Apply the given function to all registered modules
     * @param func The function to apply to the modules
     * @param recursive Flag if the function should be applied recursively to all submodules
     */
    void apply(const std::function<void(Module &)> &func, bool recursive = true);

    /**
     * Register a parameter
     * @note Parameters should be registered such that .to(), .zero_grad(), etc. apply correctly
     * @param param The parameter to register
     */
    void register_param(std::shared_ptr<Tensor> param);

    /**
     * Register a module
     * @note Modules should be registered such that .to(), .zero_grad(), etc. apply correctly
     * @param module The module to register
     */
    void register_module(Module &module);

    /**
     * Register a named module, which will show during a pretty print of the module
     * @note Modules should be registered such that .to(), .zero_grad(), etc. apply correctly
     * @param module The module to register
     */
    void register_module(Module &module, const std::string &name);

    /**
     * Set the module to train mode
     * @note Some layers behave differently during the forward pass whether its in training or evaluation mode
     * @param is_train True for training mode, false for evaluation mode
     */
    void train(bool is_train = true);

    /**
     * Set the module to evaluation mode
     * @note This is equivalent to .train(false)
     */
    void eval();

private:
    friend std::ostream &operator<<(std::ostream &os, Module &module);

    void pretty_print_recursive(std::ostream &os, const std::string &indentation) const;
    void get_params(std::vector<Tensor> &params, bool recursive) const;
    void get_params(std::vector<std::reference_wrapper<Tensor>> &params, bool recursive) const;

protected:
    std::vector<std::shared_ptr<Tensor>> params_;
    std::vector<std::pair<std::string, std::reference_wrapper<Module>>> modules_;
    bool is_train_ = true;
};

}    // namespace tinytensor::nn

#endif    // TINYTENSOR_NN_MODULE_H_
