/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "../linalg/vector_ops.hpp"

namespace linalg {

/**
 * @brief compute the L2 norm of a vector
 *
 * || X ||_2 or sqrt(X^T X)
 *
 * @param vector X with N elements of type T
 *
 * @return scalar
 */
template <typename T, size_t N>
  requires(FieldLike<T>)
constexpr auto L2(const Vector<T, N>& X) {
  scalar_type_t<T> result = 0;
  if constexpr (!is_complex_v<T>)
    result = dot(X, X);
  else {
    const auto tmp = dot(conjugate(X), X);
    assert(is_real(tmp));
    result = tmp.real();
  }
  return constexpr_sqrt(result);
}

/**
 * @brief compute the L1 norm of a vector
 *
 * || X ||_1 or sum_i^N(|x_i|) \forall i
 *
 * @param vector X with N elements of type T
 *
 * @return scalar
 */
template <typename T, size_t N>
  requires(FieldLike<T>)
constexpr auto L1(const Vector<T, N>& X) {
  T sum = 0;
  for (const auto& x : X) sum += constexpr_abs(x);
  return sum;
}

/**
 * @brief compute the max norm of a vector
 *
 * ||X||_\inf or max value of X (abs)
 *
 * @param vector X with N elements of type T
 *
 * @return scalar
 */
template <typename T, size_t N>
  requires(FieldLike<T>)
constexpr auto max_norm(const Vector<T, N>& X) {
  T mx = 0;
  for (const auto& x : X) mx = std::max(mx, constexpr_abs(x));
  return mx;
}

}  // namespace linalg
