// index.h
// Int and Slice indexing

#ifndef TINYTENSOR_INDEX_H_
#define TINYTENSOR_INDEX_H_

#include <tt/exception.h>
#include <tt/export.h>

#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace tinytensor::indexing {

constexpr std::nullopt_t None = std::nullopt;

// A range slice (start, end, stride)
class TINYTENSOR_EXPORT Slice {
public:
    using SliceIdx = std::optional<int>;
    Slice(SliceIdx start_idx = None, SliceIdx end_idx = None, SliceIdx step = None);

    [[nodiscard]] constexpr auto start() const -> int {
        return start_.value_or(0);
    }

    [[nodiscard]] constexpr auto end() const -> int {
        return end_.value_or(std::numeric_limits<int>::max());
    }

    [[nodiscard]] constexpr auto end(int dim) const -> int {
        return end_.value_or(dim - 1);
    }

    [[nodiscard]] constexpr auto stride() const -> int {
        return stride_.value_or(1);
    }

    [[nodiscard]] auto to_string() const -> std::string;

    [[nodiscard]] auto to_size(int dim) const -> int;

private:
    SliceIdx start_;
    SliceIdx end_;
    SliceIdx stride_;
};

// Index can be either an int or Slice
class TINYTENSOR_EXPORT Index {
public:
    using IndexT = std::variant<int, Slice>;
    Index(int index)
        : index_(index) {}
    Index(Slice slice)
        : index_(std::move(slice)) {}

    [[nodiscard]] constexpr auto get_index() const -> const IndexT & {
        return index_;
    }

private:
    IndexT index_;
};
using IndexList = std::vector<Index>;

/**
 * Utility function to convert a list of indices to a pretty-print string
 */
TINYTENSOR_EXPORT auto index_list_to_string(const IndexList &indices) -> std::string;

}    // namespace tinytensor::indexing

#endif    // TINYTENSOR_INDEX_H_
