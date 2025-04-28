#include <tt/print_options.h>

#include <algorithm>

namespace tinytensor {

namespace detail {
int precision = 4;        // NOLINT
int width = 11;           // NOLINT
int line_width = 10;      // NOLINT
int max_lines = 100;      // NOLINT
bool suppress = false;    // NOLINT
}    // namespace detail

void set_print_precision(int precision) {
    detail::precision = std::max(precision, 0);
}
auto get_print_precision() -> int {
    return detail::precision;
}

void set_print_width(int width) {
    detail::width = std::max(width, 1);
}
auto get_print_width() -> int {
    return detail::width;
}

void set_max_lines(int max_lines) {
    detail::max_lines = std::max(max_lines, 1);
}
auto get_max_lines() -> int {
    return detail::max_lines;
}

void set_print_line_width(int line_width) {
    detail::line_width = std::max(line_width, 4);
}
auto get_print_line_width() -> int {
    return detail::line_width;
}

void set_print_suppression(bool suppress) {
    detail::suppress = suppress;
}
auto get_print_suppression() -> bool {
    return detail::suppress;
}

}    // namespace tinytensor
