/* Copyright © 2025 Akreon LLC. All rights reserved.
 *
 * Proprietary and Confidential.
 * This source code is the property of Akreon LLC and constitutes a trade secret
 * protected by U.S. and international law.
 *
 * Unauthorized copying, modification, distribution, or disclosure of this file,
 * via any medium, is strictly prohibited without express written permission
 * from Akreon LLC.
 */

#pragma once

#include <cstddef>
#include <utility>

#include "linalg/scalar/scalar_traits.hpp"

namespace linalg {

template <typename T, std::size_t Dim>
struct Vector;

template <class Vec>
struct vector_traits;

template <typename T, std::size_t Dim>
struct vector_traits<Vector<T, Dim>> {
  using element_type = T;
  static constexpr std::size_t dim = Dim;

  template <typename... Ts>
  static constexpr auto operator()(Ts&&... xs) {
    static_assert(sizeof...(xs) == Dim,
                  "vector_traits::operator() requires exactly Dim coordinates");
    Vector<T, Dim> vector;
    std::size_t i = 0;
    auto set_element = [&](const T& x) { vector(i++) = x; };
    ((void)set_element(std::forward<Ts>(xs)), ...);
    return vector;
  }

  static constexpr auto fill(const T& x) {
    Vector<T, Dim> vector;
    for (std::size_t i = 0; i < Dim; ++i) vector(i) = x;
    return vector;
  }
};

template <class Vec>
using vector_traits_t = vector_traits<remove_cvref_t<Vec>>;

template <class Vec>
using vector_element_t = typename vector_traits_t<Vec>::element_type;

template <class Vec>
inline constexpr std::size_t vector_dim_v = vector_traits_t<Vec>::dim;

template <class T>
struct vector_scalar {
  using type = remove_cvref_t<T>;
};

template <typename T, std::size_t Dim>
struct vector_scalar<Vector<T, Dim>> : vector_scalar<T> {};

template <class Vec>
using vector_scalar_t = typename vector_scalar<remove_cvref_t<Vec>>::type;

} // namespace linalg
