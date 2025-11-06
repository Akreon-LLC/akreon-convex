/*
 * Copyright © 2025 Akreon LLC. All rights reserved.
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

#include "convex_set.hpp"

namespace convex {

/**
 * @brief Simplex definition
 *
 * S = { x | x >= 0, \sum x_i = 1 }
 *
 * Useful in probabilities
 */
template <linalg::RnSpace Space, ViewTag Tag = ViewTag::ConvexSet>
  requires(linalg::space_dim_v<Space> ==
           linalg::vec_size_v<linalg::space_point_t<Space>>)
class Simplex : public ConvexSet<Space, Simplex<Space, Tag>> {
 public:
  /// @brief bring into scope the type traits from the base class
  using ConvexBase = ConvexSet<Space, Simplex<Space, Tag>>;
  using typename ConvexBase::Point;
  using typename ConvexBase::T;

  /// @brief bring into scope the const data members
  using ConvexBase::Dim;

  /// @brief helper to get n elements in point
  static constexpr size_t N = linalg::vec_size_v<Point>;

  /**
   * @brief contains function to see if x is in the simplex
   *
   * @param point x
   * @param tolerance (scalar)
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
   * @brief contains function to see if x is in the simplex
   *
   * C' = { (x, t) \in R^n x R_++ | 1^T x == t }
   *
   * @param (x, t) \in R^n x R_++
   * @param tolerance (scalar)
   *
   * @return T/F
   */
  constexpr bool contains_impl(const std::tuple<Point, T>& x_in,
                               const T& tol) const {
    const auto x = std::get<0>(x_in);
    auto t = std::get<1>(x_in);
    if (t < T(0)) return false;

    T sum(0);
    for (size_t i = 0; i < N; ++i) {
      if (x(i) < T(0)) return false;
      sum += x(i);
    }
    return linalg::constexpr_abs(sum - t) <= tol;
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
  constexpr auto transform_impl(const affine::AffineMap<Mat>& map) const {
    const auto& U = map.linear_part();
    const auto& t = map.translation();
    const auto U_inv = linalg::inverse(U);

    linalg::vec_traits<linalg::Vector<T, N>> vec;
    const auto ones = vec.fill(T(1));

    const auto one_T_U_inv = as_row_vector(ones) * U_inv;

    const auto A = T(-1) * U_inv;
    const auto b = T(-1) * U_inv * t;
    const auto C = one_T_U_inv;
    const auto d = T(1) + (one_T_U_inv * t)(0);

    return Polyhedron<Space, HalfspaceType::CLH, decltype(A), decltype(C), Tag>(
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

  /**
   * @brief inverse perspective func from convex set to cone
   *
   * @return cone
   */
  constexpr auto cone_view_impl() const {
    return ConeView(Simplex<Space, ViewTag::LiftedCone>());
  }
};

}  // namespace convex
