/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "../convex_traits.hpp"

namespace convex {

/**
 * @brief this is an owning cone view of the underlying convex set
 */
template <ConvexConeLike Cone>
class ConeView {
 public:
  /// @brief alias to extract underlying scalar type of convex set
  using T = set_type_t<Cone>;

  /// @brief alias for point
  using Point = set_point_t<Cone>;

  /**
   * @brief CTOR
   *
   * @param ConvexSet argument
   */
  constexpr explicit ConeView(const Cone& cone_in) : cone(cone_in) {}

  /**
   * @brief contains func (wrapped around cone impl)
   *
   * @param point in space
   * @param tolerance
   *
   * @return T/F
   */
  constexpr bool contains(const Point& x, T tol = 1e-8) const {
    return cone.contains_impl(x, tol);
  }

 private:
  Cone cone;
};

}  // namespace convex
