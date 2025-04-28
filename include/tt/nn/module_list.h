// module_list.h
// List of arbitrary type modules

#ifndef TINYTENSOR_NN_MODULE_LIST_H_
#define TINYTENSOR_NN_MODULE_LIST_H_

#include <tt/device.h>
#include <tt/nn/module.h>
#include <tt/tensor.h>

#include <concepts>
#include <memory>
#include <string>
#include <type_traits>

namespace tinytensor::nn {

class ModuleList : public Module {
private:
    CheckedVec<std::shared_ptr<nn::Module>> modules;
    using Iterator = decltype(modules)::Iterator;
    using ConstIterator = decltype(modules)::ConstIterator;

public:
    template <typename M>
        requires(std::derived_from<M, nn::Module> && !std::is_lvalue_reference_v<M>)
    void push_back(M &&module) {
        using T = std::remove_reference_t<M>;
        modules.push_back(std::make_shared<T>(std::forward<M>(module)));
        register_module(*modules[-1]);
    }

    auto operator[](int idx) -> nn::Module & {
        return *modules[idx];
    }

    auto operator[](int idx) const -> const nn::Module & {
        return *modules[idx];
    }

    [[nodiscard]] auto name() const -> std::string override {
        return "ModuleList";
    }

    [[nodiscard]] auto begin() -> Iterator {
        return modules.begin();
    }
    [[nodiscard]] auto begin() const -> ConstIterator {
        return modules.begin();
    }
    [[nodiscard]] auto end() -> Iterator {
        return modules.end();
    }
    [[nodiscard]] auto end() const -> ConstIterator {
        return modules.end();
    }
};

}    // namespace tinytensor::nn

#endif    // TINYTENSOR_NN_MODULE_LIST_H_
