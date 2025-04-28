// print.h
// Printing functionality, follows the logic used by Torch/Numpy

#ifndef TINYTENSOR_PRINT_H_
#define TINYTENSOR_PRINT_H_

#include <tt/print_options.h>
#include <tt/scalar.h>
#include <tt/shape.h>

#include "backend/common/span.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <format>
#include <iomanip>
#include <iostream>
#include <string>

namespace tinytensor {
namespace detail {

constexpr double SUPPRESS_MAX = 1e3;
constexpr double SUPPRESS_MIN = 1e-4;

template <typename T>
std::ostream &print_data(
    const HostSpan<const T> &data,
    int idx,
    int dim,
    const Shape &shape,
    const Shape &stride,
    int offset,
    bool print_fixed,
    bool is_bool,
    std::ostream &os
) {
    auto s = static_cast<std::size_t>(stride[shape.size() - dim - 1]);
    auto d = static_cast<std::size_t>(shape[shape.size() - dim - 1]);
    static std::string prefix = "";
    static std::string postfix = "";
    if (dim == 0) {
        os << postfix;
        for (int i = 0; i < static_cast<int>(postfix.size()); ++i) {
            os << std::endl;
        }
        prefix += "[";
        os << std::setw(static_cast<int>(shape.size())) << prefix << " ";
        prefix = "";
        for (std::size_t i = 0, j = 0; i < d; ++i, j += s) {
            // Check for printing long rows
            if (i == 3 && d > static_cast<std::size_t>(get_print_line_width())) {
                i = d - 4;
                j = i * s;
                os << std::format("{:>5}", "...,");
                continue;
            }
            // Check for bool specialization
            const auto value = data[static_cast<std::size_t>(idx + offset) + j];
            if (is_bool) {
                os << std::format("{:>5}", static_cast<bool>(value));
            } else {
                int precision = get_print_precision();
                int width = get_print_width();
                if (print_fixed) {
                    if constexpr (IsScalarIntType<T>) {
                        width -= std::min(precision, 5);
                        precision = 0;
                    }
                    os << std::format("{:>{}.{}f}", static_cast<double>(value), width, precision);
                } else {
                    os << std::format("{:>{}.{}e}", static_cast<double>(value), width, precision);
                }
            }
            os << (i < d - 1 ? ", " : " ");
        }
        postfix = "";
    } else {
        prefix += "[";
        for (std::size_t i = 0; i < d; ++i) {
            if (i == 3 && d > static_cast<std::size_t>(get_max_lines())) {
                os << postfix << std::endl;
                // Leading indent for ...
                for (int j = 0; j < static_cast<int>(postfix.size()) - 2; ++j) {
                    os << std::endl;
                }
                for (int j = 0; j < static_cast<int>(shape.size()); ++j) {
                    os << " ";
                }
                os << "..., " << std::endl;
                for (int j = 0; j < static_cast<int>(postfix.size()) - 2; ++j) {
                    os << std::endl;
                }
                i = d - 3;
                postfix = "";
            }
            print_data(data, idx, dim - 1, shape, stride, offset, print_fixed, is_bool, os);
            postfix += "]";
            idx += static_cast<int>(s);
        }
    }
    // Missing final closing brace at end
    if (dim == stride.size() - 1) {
        for (int i = 0; i < stride.size(); ++i) {
            os << "]";
        }
        postfix = "";
    }
    return os;
}

}    // namespace detail

template <typename T>
std::ostream &print_data(
    const HostSpan<const T> &data,
    const Shape &shape,
    const Shape &stride,
    int offset,
    bool fixed_formatting,
    bool is_bool,
    std::ostream &os
) {
    bool print_fixed = get_print_suppression() || fixed_formatting;
    return detail::print_data<T>(data, 0, shape.size() - 1, shape, stride, offset, print_fixed, is_bool, os);
}

}    // namespace tinytensor

#endif    // TINYTENSOR_STORAGE_BASE_H_
