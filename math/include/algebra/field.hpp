/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include <algorithm>
#include <complex>
#include <type_traits>

// Forward declarations
namespace topology {

struct Space;

/** \struct Topology
 *  \brief Structure that deals with topological sets
 *  \details See Topology struct definition for more details
 *  \ingroup topology
 */
template <typename T, size_t N, typename Derived = Space>
struct Topology;

template <typename T, size_t N, typename Derived = Space>
struct TopologicalSpace;

template <typename T, size_t N, typename DistanceFn>
struct MetricSpace;

/** \struct OpenSet
 *  \brief Structure that deals with open sets
 *  \details See open set struct definition for more details
 *  \ingroup topology
 */
template <typename T, size_t N>
struct OpenSet;

}  // namespace topology

namespace linalg {

/** \struct Vector
 *  \brief Structure that deals with vector types \in vector spaces
 *  \details See vector struct definition for more details
 *  \ingroup linalg
 */
template <typename T, size_t N>
struct Vector;

/** \struct scalar_type
 *  \brief Structure that deals in type_trait deduction for a scalar type
 *  \details scalar_type is an overloaded type that strips away meta-templates
 * to the final primitive scalar type
 *  \ingroup linalg
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

template <typename T, size_t N, size_t M, typename Derived>
struct scalar_type<topology::Topology<topology::OpenSet<T, N>, M, Derived>> {
  using value_type = typename scalar_type<T>::value_type;  // e.g., double
};

template <typename T, size_t N, typename Derived>
struct scalar_type<
    topology::TopologicalSpace<topology::OpenSet<T, N>, N, Derived>> {
  using value_type = typename scalar_type<T>::value_type;
};

template <typename T, size_t N, size_t M, typename DistanceFn>
struct scalar_type<
    topology::MetricSpace<topology::OpenSet<T, N>, M, DistanceFn>> {
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
 *  \ingroup linalg
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

/**
 * @brief FieldLike concept is obtained by floating point values
 *
 * ** Group axioms **
 * 0. a + b \exists
 * 1. (a + b) + c == a + (b + c)
 * 2. -1 * a \exists
 * 3. a + b == b + a
 *
 * ** Ring axioms **
 * Group plus the following
 * 0. a * b \exists
 * 1. (a * b) * c == a * (b * c)
 * 2. a * (b + c) == a * b + a * c
 * 3. a * b == b * a
 *
 * ** Field axioms **
 * 0. a / a == 1
 * 1. a + 0 == a
 * 2. a * 1 == a
 */
template <typename T>
concept FieldLike = FloatingPoint<T>;

}  // namespace linalg
