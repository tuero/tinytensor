// concepts.h
// Custom concepts and type traits

#ifndef TINYTENSOR_CONCEPTS_H_
#define TINYTENSOR_CONCEPTS_H_

#include <concepts>
#include <type_traits>

namespace tinytensor {

namespace detail {

// Get nested template type
template <typename T>
struct template_parameter;

template <template <typename...> class C, typename T>
struct template_parameter<C<T>> {
    using type = T;
};

// https://stackoverflow.com/a/54191646
template <typename T, template <typename...> class Z>
struct is_specialization_of : std::false_type {};

template <typename... Args, template <typename...> class Z>
struct is_specialization_of<Z<Args...>, Z> : std::true_type {};

template <typename T, template <typename...> class Z>
inline constexpr bool is_specialization_of_v = is_specialization_of<T, Z>::value;

}    // namespace detail

/**
 * Concept if type T is in list U
 */
template <typename T, typename... U>
concept IsAnyOf = (std::same_as<T, U> || ...);

/**
 * Concept if type list U are all of same type T
 */
template <typename T, typename... U>
concept IsAllOf = (std::same_as<T, U> && ...);

/**
 * Extract inner template type
 */
template <typename T>
using template_parameter_t = typename detail::template_parameter<T>::type;

/**
 * Check for if T is a template specialization of Z
 */
template <typename T, template <typename...> class Z>
concept IsSpecialization = detail::is_specialization_of_v<T, Z>;

}    // namespace tinytensor

#endif    // TINYTENSOR_CONCEPTS_H_
