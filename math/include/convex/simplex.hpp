/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "convex_set.hpp"

namespace convex {

/**
 * @brief Simplex definition
 *
 * S = { x | x >= 0, \sum x_i = 1 }
 *
 * Useful in probabilities
 */
template <linalg::VectorSpace Space>
  requires(linalg::space_dim_v<Space> ==
           linalg::vec_size_v<linalg::space_point_t<Space>>)
class Simplex : public ConvexSet<Space, Simplex<Space>> {
 public:
  /// @brief bring into scope the type traits from the base class
  using ConvexBase = ConvexSet<Space, Simplex<Space>>;
  using typename ConvexBase::Point;
  using typename ConvexBase::ScalarT;
  using typename ConvexBase::T;

  /// @brief bring into scope the const data members
  using ConvexBase::Dim;

  /// @brief helper to get n elements in point
  static constexpr size_t N = linalg::vec_size_v<Point>;

 protected:
  /// @brief allow base class to invoke impl funcs
  friend ConvexBase;

  /**
   * @brief contains function to see if x is in the simplex
   *
   * @param point x
   *
   * @return T/F
   */
  constexpr bool contains_impl(const Point& x, const T& tol) const {
    T sum(0);
    for (size_t i = 0; i < N; ++i) {
      if (x(i) < T(0)) return false;
      sum += x(i);
    }
    return linalg::constexpr_abs(sum - T(1)) <= tol;
  }

  /**
   * @brief is_well_formed impl check
   *
   * @return T/F
   */
  constexpr bool is_well_formed_impl() const { return true; }

  /**
   * @brief transform simplex to polyhedron
   *
   * S = { x | x >= 0, sum_i x_i = 1 }
   * f(x) = U x + t
   * S' =  { y = f(x) | x \in S }
   * y = U x + t \to x = U^{-1} (y - t)
   *
   * Affine Inequality
   * A x <= b... A x <= 0... A U^{-1} y <= U^{-1} t
   * A = [ -U^{-1} ]
   * b = -U^{-1} t
   *
   * Linear Eqaulity
   * C x = d ... C = I and d = 1 in simplex
   * C U^{-1} y - U^{-1} t = d
   * C U^{-1} y = d + U^{-1} t
   *
   * C = [ 1^T U^{-1} ]
   * d = 1 + 1^T U^{-1} t
   *
   * @param AffineMap
   *
   * @return polyhedron
   */
  template <linalg::MatrixLike Mat>
    requires(std::same_as<T, linalg::mat_type_t<Mat>>)
  constexpr auto transform_impl(const AffineMap<Mat>& map) const {
    const auto& U = map.linear_part();
    const auto& t = map.translation();
    const auto U_inv = linalg::inverse(U);

    linalg::Vector<T, N> ones;
    for (size_t i = 0; i < N; ++i) ones[i] = T(1);

    const auto one_T_U_inv = as_row_vector(ones) * U_inv;

    const auto A = T(-1) * U_inv;
    const auto b = T(-1) * U_inv * t;
    const auto C = one_T_U_inv;
    const auto d = T(1) + (one_T_U_inv * t)(0);

    return Polyhedron<Space, HalfspaceType::CLH, decltype(A), decltype(C)>(
        A, b, C, d);
  }

  /**
   * @brief equality comparison
   *
   * @param convex set
   *
   * @return T/F
   */
  constexpr bool is_equal_impl(const ConvexBase& /*other*/) const {
    return true;
  }
};

}  // namespace convex
