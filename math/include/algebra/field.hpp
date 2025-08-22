/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include <complex>
#include <type_traits>

// Forward declarations
namespace topology {

/** \struct Topology
 *  \brief Structure that deals with topological sets
 *  \details See Topology struct definition for more details
 *  \ingroup topology
 */
template <typename T, size_t N>
struct Topology;

/** \struct OpenSet
 *  \brief Structure that deals with open sets
 *  \details See open set struct definition for more details
 *  \ingroup topology
 */
template <typename T, size_t N>
struct OpenSet;

}  // namespace topology

namespace algebra {

/** \struct Vector
 *  \brief Structure that deals with vector types \in vector spaces
 *  \details See vector struct definition for more details
 *  \ingroup algebra
 */
template <typename T, size_t N>
struct Vector;

/** \struct scalar_type
 *  \brief Structure that deals in type_trait deduction for a scalar type
 *  \details scalar_type is an overloaded type that strips away meta-templates
 * to the final primitive scalar type
 *  \ingroup algebra
 */
template <typename T>
struct scalar_type {
  using value_type = T;
};

/// @brief True if type is of a std::array<T, N>
/// @tparam value_type is \equiv to the deduced scalar type value
template <typename T, size_t N>
struct scalar_type<std::array<T, N>> {
  using value_type = typename scalar_type<T>::value_type;
};

/// @brief True if type is of a std::complex<T>
/// @tparam value_type is \equiv to the deduced scalar type value
template <typename T>
struct scalar_type<std::complex<T>> {
  using value_type = typename scalar_type<T>::value_type;
};

/// @brief True if type is of a OpenSet<T, N>
/// @tparam value_type is \equiv to the deduced scalar type value
template <typename T, size_t N>
struct scalar_type<topology::OpenSet<T, N>> {
  using value_type = typename scalar_type<T>::value_type;
};

/// @brief True if type is of a Topology<OpenSets<T, N>, M>
/// @tparam value_type is \equiv to the deduced scalar type value
template <typename T, size_t N, size_t M>
struct scalar_type<topology::Topology<topology::OpenSet<T, N>, M>> {
  using value_type = typename scalar_type<T>::value_type;
};

/// @brief True if type is of a Vector<T, N>
/// @tparam value_type is \equiv to the deduced scalar type value
template <typename T, size_t N>
struct scalar_type<Vector<T, N>> {
  using value_type = typename scalar_type<T>::value_type;
};

/// @brief True if type is of a Vector<Vector<T, N>, M>
/// @tparam value_type is \equiv to the deduced scalar type value
template <typename T, size_t M, size_t N>
struct scalar_type<Vector<Vector<T, M>, N>> {
  using value_type = typename scalar_type<T>::value_type;
};

/// @brief scalar_type_t alias for deducing scalar type of templated argument
template <typename T>
using scalar_type_t = typename scalar_type<T>::value_type;

/// @brief FloatingPoint concept requires underlying scalar type to be floating
/// point
template <typename T>
concept FloatingPoint = std::floating_point<scalar_type_t<T>>;

/** \struct is_complex_t
 *  \brief Structure that deals in type_trait deduction for checking if type is
 * complex (false initially)
 *  \details scalar_type is an overloaded type that strips away meta-templates
 * to see if the type is complex
 *  \ingroup algebra
 */
template <typename T>
struct is_complex_t : public std::false_type {};

/// @brief True if type is of a complex<T>
template <typename T>
struct is_complex_t<std::complex<T>> : public std::true_type {};

/// @brief True if type is of a complex<T> otherwise false (default)
template <typename T, size_t N>
struct is_complex_t<Vector<T, N>> {
  static const bool value = is_complex_t<T>::value;
};

/// @brief True if type is of a complex<T> otherwise false (default)
template <typename T, size_t N, size_t M>
struct is_complex_t<Vector<Vector<T, M>, N>> {
  static const bool value = is_complex_t<T>::value;
};

/// @brief True if type is of a complex<T> otherwise false (default)
template <typename T, size_t N>
struct is_complex_t<topology::OpenSet<T, N>> {
  static const bool value = is_complex_t<T>::value;
};

/// @brief True if type is of a complex<T> otherwise false (default)
template <typename T, size_t N, size_t M>
struct is_complex_t<topology::Topology<topology::OpenSet<T, M>, N>> {
  static const bool value = is_complex_t<T>::value;
};

/// @brief is_complex_v is the variable holding the value type of is_complex_t
template <typename T>
constexpr bool is_complex_v = is_complex_t<T>::value;

/**
 * @brief Checks if all elements in the set are 1
 *
 * @param set is an open set of type T with N elements
 *
 * @return T/F
 */
template <typename T, size_t N>
constexpr bool is_ones_like(const topology::OpenSet<T, N>& set) {
  return std::all_of(std::begin(set), std::end(set),
                     [](const auto& x) { return x == 1; });
}

/// @brief GroupLike concept requires the type to fullfill group axioms
template <typename T>
concept GroupLike = requires(T a, T b, T c) {
  { a + b } -> std::same_as<T>;
  { (a + b) + c == a + (a + c) };
  { -a } -> std::same_as<T>;
  { a + b == b + a };
};

/// @brief RingLike concept requires the type to fullfill ring axioms
template <typename T>
concept RingLike = GroupLike<T> && requires(T a, T b, T c) {
  { a * b } -> std::same_as<T>;
  { (a * b) * c == a * (b * c) };
  { a * (b + c) == a * b + a * c };
  { a * b == b * a };
};

/// @brief FieldLike concept requires the type to fullfill field axioms
template <typename T>
concept FieldLike = FloatingPoint<T> || (RingLike<T> && requires(T a, T b) {
                      { is_ones_like(a / a) };
                      { a + T{0} } -> std::same_as<T>;
                      { a * T{1} } -> std::same_as<T>;
                    });

}  // namespace algebra
