// dag.h
// Autograd DAG for correct gradient computation ordering

#include <tt/autograd.h>
#include <tt/tensor.h>

#include <functional>
#include <unordered_set>

namespace tinytensor::autograd {

auto build_dag(const Tensor &tensor) -> TensorList {
    std::unordered_set<SharedGrad *> cache;
    std::function<void(const Tensor &)> recurse;
    TensorList dag;

    recurse = [&](const Tensor &t) {
        auto id = t.ctx_.get();
        if (cache.find(id) != cache.end()) {
            return;
        }

        for (const auto &parent : t.ctx_->parents) {
            recurse(parent);
        }
        cache.insert(id);
        dag.push_back(t);
    };

    recurse(tensor);
    return dag;
}

}    // namespace tinytensor::autograd
