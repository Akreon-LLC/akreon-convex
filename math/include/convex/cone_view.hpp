/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "cone_view_base.hpp"

namespace convex {

/**
 * @brief Cone View is an owning cone view of the underlying convex set
 */
template <ConvexConeLike Cone>
struct ConeView : ConeViewBase<ConeView<Cone>> {
  /// @helpers from base class
  using ConeViewBase = ConeViewBase<ConeView<Cone>>;
  using typename ConeViewBase::Point;
  using typename ConeViewBase::T;

  /**
   * @brief CTOR
   *
   * @param ConvexSet argument
   */
  constexpr explicit ConeView(const Cone& cone_in) : ConeViewBase(cone_in) {}

  /**
   * @brief contains func (wrapped around cone impl)
   *
   * @param point in space
   * @param tolerance
   *
   * @return T/F
   */
  constexpr bool contains_impl(const Point& x, T tol = 1e-8) const {
    return this->cone.contains_impl(x, tol);
  }

  /**
   * @brief get dual view of cone
   *
   * @return dual cone view
   */
  constexpr auto dual_view_impl(const Cone& cone) const {
    return DualConeView<Cone>(cone);
  }
};

}  // namespace convex
