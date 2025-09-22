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
struct L2 {
  constexpr auto operator()(const Vector<T, N>& x) const {
    scalar_type_t<T> result = 0;
    if constexpr (!is_complex_v<T>)
      result = dot(x, x);
    else {
      const auto tmp = dot(conjugate(x), x);
      assert(is_real(tmp));
      result = tmp.real();
    }
    return constexpr_sqrt(result);
  }
};

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
struct L1 {
  constexpr T operator()(const Vector<T, N>& x) const {
    T sum = 0;
    for (const auto& _x : x) sum += constexpr_abs(_x);
    return sum;
  }
};

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
struct Linf {
  constexpr T operator()(const Vector<T, N>& x) const {
    T mx = 0;
    for (const auto& _x : x) mx = std::max(mx, constexpr_abs(_x));
    return mx;
  }
};

}  // namespace linalg
