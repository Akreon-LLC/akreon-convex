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
 * @brief Nonnegative orthant R^n_+
 */
template <linalg::RnSpace Space, ViewTag Tag = ViewTag::ConvexSet>
  requires(linalg::space_dim_v<Space> ==
           linalg::vec_size_v<linalg::space_point_t<Space>>)
class NonnegOrthant : public ConvexSet<Space, NonnegOrthant<Space, Tag>> {
 public:
  /// @brief bring into scope the type traits from the base class
  using ConvexBase = ConvexSet<Space, NonnegOrthant<Space, Tag>>;
  using typename ConvexBase::Point;
  using typename ConvexBase::T;

  /// @brief dimension
  using ConvexBase::Dim;

  /// @brief helper to get N elements in point
  static constexpr size_t N = linalg::vec_size_v<Point>;

  /**
   * @brief contains function to see if point x lies in the R^n_+ set
   *
   * @param point x of type T (n-dim)
   * @param tolerance
   *
   * @return T/F
   */
  constexpr bool contains_impl(const Point& x, const T& /* tol */) const {
    for (size_t i = 0; i < N; ++i)
      if (x(i) < T(0)) return false;
    return true;
  }

  /**
   * @brief contains function to see if point x lies in the R^n_+ set
   *
   * @param (x, t) \in R^n_+ x R
   * @param tolerance
   *
   * @return T/F
   */
  constexpr bool contains_impl(const std::tuple<Point, T>& x_in,
                               const T& tol) const {
    const auto x = std::get<0>(x_in);
    return contains_impl(x, tol);
  }

  /**
   * @brief is well formed check
   *
   * @return T/F
   */
  constexpr bool is_well_formed_impl() const { return true; }

  /**
   * @brief is equal implementation
   *
   * @return T/F
   */
  constexpr bool is_equal_impl(const ConvexBase& /* other */) const {
    return true;
  }

  /**
   * @brief transform impl
   *
   * @return self
   */
  template <linalg::MatrixLike Mat>
  constexpr auto transform_impl(const affine::AffineMap<Mat>& /* map */) const {
    return *this;
  }

  /**
   * @brief to cone
   *
   * @return cone
   */
  constexpr auto cone_view_impl() const {
    if constexpr (Tag == ViewTag::ConvexSet)
      return ConeView(NonnegOrthant<Space, ViewTag::LiftedCone>());
    else
      return ConeView(NonnegOrthant<Space, Tag>());
  }
};

}  // namespace convex
