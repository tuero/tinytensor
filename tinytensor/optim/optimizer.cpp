// optimizer.cpp
// Base optimizer

#include <tt/optim/optimizer.h>
#include <tt/tensor.h>

#include <functional>
#include <vector>

namespace tinytensor::optim {

void Optimizer::zero_grad() {
    for (auto &t : params_) {
        t.get().clear_grad();
    }
}

void Optimizer::add_parameters(const std::vector<std::reference_wrapper<Tensor>> &params) {
    params_.insert(params_.end(), params.begin(), params.end());
}

}    // namespace tinytensor::optim
