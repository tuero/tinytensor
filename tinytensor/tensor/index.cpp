// index.cpp
// Int and Slice indexing

#include <tt/exception.h>
#include <tt/index.h>
#include <tt/util.h>

#include <algorithm>
#include <sstream>
#include <string>
#include <variant>

namespace tinytensor::indexing {

namespace {
constexpr auto _to_string(const Slice::SliceIdx &slice) -> std::string {
    return slice.has_value() ? std::to_string(slice.value()) : "";
}
}    // namespace

Slice::Slice(SliceIdx start_idx, SliceIdx end_idx, SliceIdx step)
    : start_(start_idx), end_(end_idx), stride_(step) {
    if (stride() <= 0) {
        TT_EXCEPTION("Slice stride must be greater than 0.");
    }
}
auto Slice::to_size(int dim) const -> int {
    int start_idx = std::max(start(), 0);
    int end_idx = std::min(end(), dim);
    return ceil_div(end_idx - start_idx, stride_.value_or(1));
}

auto Slice::to_string() const -> std::string {
    std::stringstream ss;
    ss << _to_string(start_) << ":" << _to_string(end_) << ":" << _to_string(stride_);
    return ss.str();
}

auto index_list_to_string(const IndexList &indices) -> std::string {
    std::stringstream ss;
    std::string delim = "";
    ss << "[";
    for (const auto &index : indices) {
        std::visit(
            overloaded{[&](int idx) { ss << delim << idx; }, [&](Slice slice) { ss << delim << slice.to_string(); }},
            index.get_index()
        );
        delim = ", ";
    }
    ss << "]";
    return ss.str();
}

}    // namespace tinytensor::indexing
