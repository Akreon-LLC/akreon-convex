/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "cone_view_base.hpp"

namespace convex {

/**
 * @brief Dual Cone View is another owning view onto the underlying cone K
 * but in the dual sense.
 *
 * K^* = { y | x^T y >= 0, x \in K }
 */
template <ConvexConeLike Cone>
struct DualConeView : ConeViewBase<DualConeView<Cone>> {
  /// @helpers from base class
  using ConeViewBase = ConeViewBase<DualConeView<Cone>>;
  using typename ConeViewBase::Point;
  using typename ConeViewBase::T;

  /**
   * @brief CTOR
   *
   * @param ConvexSet argument
   */
  constexpr explicit DualConeView(const Cone& cone_in)
      : ConeViewBase(cone_in) {}

  /**
   * @brief contains func (wrapped around cone impl)
   *
   * @param point in space
   * @param tolerance
   *
   * @return T/F
   */
  constexpr bool contains_impl(const Point& x, T tol = 1e-8) const {
    return this->cone.contains_dual_impl(x, tol);
  }

  /**
   * @brief get dual view of cone
   *
   * K^** = K
   *
   * @return cone view
   */
  constexpr auto dual_impl(const Cone& cone) const {
    return ConeView<Cone>(cone);
  }
};

}  // namespace convex
