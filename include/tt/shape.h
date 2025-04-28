// shape.h
// Array shape type definition

#ifndef TINYTENSOR_SHAPE_H_
#define TINYTENSOR_SHAPE_H_

#include <format>
#include <initializer_list>
#include <iostream>
#include <string>
#include <vector>

namespace tinytensor {

/**
 * Defines the Tensor's shape, and operations between them
 */
class Shape {
public:
    Shape() = default;
    explicit Shape(const std::vector<int> &dims);
    explicit Shape(std::vector<int> &&dims);
    Shape(std::initializer_list<int> dims);

    /**
     * Get the total number of elements the shape represents
     * @return Total number of elements
     */
    [[nodiscard]] auto numel() const noexcept -> int;

    /**
     * Get the total number of dimensions the shape represents
     * @return Total number of dimensions
     */
    [[nodiscard]] auto ndim() const noexcept -> int;

    /**
     * Get the total number of dimensions the shape represents
     * @return Total number of dimensions
     */
    [[nodiscard]] auto size() const noexcept -> int;

    // Specific Accessors

    /**
     * Get the size of the given dimension
     * @note Throws exception if dimension is invalid
     * @param index The dimension to query
     * @return Size of the given dimension
     */
    auto operator[](int index) -> int &;

    /**
     * Get the size of the given dimension
     * @note Throws exception if dimension is invalid
     * @param index The dimension to query
     * @return Size of the given dimension
     */
    auto operator[](int index) const -> int;

    // Equality operators

    /**
     * Determines equality of two shape
     * @return True if shapes are equal, false otherwise
     */
    auto operator==(const Shape &other) const noexcept -> bool;

    /**
     * Determines inequality of two shape
     * @return True if shapes are equal, false otherwise
     */
    auto operator!=(const Shape &other) const noexcept -> bool;

    /**
     * Determines equality of shape and initializer list
     * @return True if shapes are equal, false otherwise
     */
    auto operator==(const std::initializer_list<int> &other) const noexcept -> bool;

    /**
     * Determines inequality of shape and initializer list
     * @return True if shapes are equal, false otherwise
     */
    auto operator!=(const std::initializer_list<int> &other) const noexcept -> bool;

    /**
     * Remove size at the given index, modifying shape in place
     * @param dim The dim to remove
     * @return shape at index which was removed
     */
    auto pop(int dim = -1) -> int;

    /**
     * Insert size at the given index, modifying shape in place
     * @param size The size of the new dimension
     * @param dim Dimension index to insert into
     */
    void insert(int size, int dim = -1);

    /**
     * Convert Shape to string
     * @return String representation of the shape
     */
    [[nodiscard]] auto to_string() const -> std::string;

    /**
     * Compute the strides for the shape
     * @return The stride
     */
    [[nodiscard]] auto to_stride() const -> Shape;

    /**
     * Get underlying vector representation of shape
     * @return The shape as a vector
     */
    [[nodiscard]] auto to_vec() const -> const std::vector<int> &;

private:
    std::vector<int> dims;
};

auto operator<<(std::ostream &os, const Shape &shape) -> std::ostream &;

auto to_string(const Shape &shape) -> std::string;

}    // namespace tinytensor

template <>
struct std::formatter<tinytensor::Shape> : std::formatter<std::string> {
    auto format(const tinytensor::Shape &shape, format_context &ctx) const {
        return formatter<string>::format(std::format("{}", to_string(shape)), ctx);
    }
};

#endif    // TINYTENSOR_SHAPE_H_
