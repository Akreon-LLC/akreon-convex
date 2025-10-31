/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "linalg/vector_ops.hpp"
#include "norm_traits.hpp"

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
template <VectorLike Vec>
struct L2 {
  using T = vec_type_t<Vec>;
  using ScalarT = scalar_type_t<vec_type_t<Vec>>;
  constexpr auto operator()(const Vec& x) const {
    ScalarT result = 0;
    if constexpr (RealLike<T>)
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
template <VectorLike Vec>
struct L1 {
  using T = vec_type_t<Vec>;
  using ScalarT = scalar_type_t<vec_type_t<Vec>>;
  constexpr auto operator()(const Vec& x) const {
    ScalarT sum = 0;
    if constexpr (RealLike<T>)
      for (const auto& _x : x) sum += constexpr_abs(_x);
    else
      for (const auto& _x : x) sum += constexpr_absc(_x);
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
template <VectorLike Vec>
struct Linf {
  using T = vec_type_t<Vec>;
  using ScalarT = scalar_type_t<vec_type_t<Vec>>;
  constexpr auto operator()(const Vec& x) const {
    ScalarT mx = 0;
    if constexpr (RealLike<T>)
      for (const auto& _x : x) mx = std::max(mx, constexpr_abs(_x));
    else
      for (const auto& _x : x) mx = std::max(mx, constexpr_absc(_x));
    return mx;
  }
};

}  // namespace linalg
