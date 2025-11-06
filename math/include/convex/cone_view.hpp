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

#include "cone_view_base.hpp"

namespace convex {

/**
 * @brief Cone View is an owning cone view of the underlying convex set
 */
template <ConvexConeLike Cone>
struct ConeView : ConeViewBase<ConeView<Cone>> {
  /// @helpers from base class
  using ConeBase = ConeViewBase<ConeView<Cone>>;
  using ConeBase::N;
  using typename ConeBase::Point;
  using typename ConeBase::T;

  /// @brief alias for mat
  using Mat = linalg::Matrix<T, N, N>;
  using LiftedMat = linalg::Matrix<T, N + 1, N + 1>;

  /**
   * @brief CTOR
   *
   * @param ConvexSet argument
   */
  constexpr explicit ConeView(const Cone& cone_in) : ConeBase(cone_in) {}

  /**
   * @brief contains func (wrapped around cone impl)
   *
   * @param (x, t) \in R^n x R
   * @param tolerance
   *
   * @return T/F
   */
  constexpr bool contains(const std::tuple<Point, T>& x, T tol = 1e-8) const {
    return this->cone.contains_impl(x, tol);
  }

  /**
   * @brief project (x,t) onto cone coords
   *
   * @param (x,t) \in R^n x R
   *
   * @return (x,s) in cone coords
   */
  constexpr auto project(const std::tuple<Point, T>& x) const {
    return this->cone.project(x);
  }

  /**
   * @brief get slack - projected point x onto cone coordinates
   *
   * F z + b = s, s \in K, z = [ x ; t ]
   *
   * @param point (x, t) \in R^n x R
   *
   * @return s \in R
   */
  constexpr auto get_slack(const std::tuple<Point, T>& /*x*/) const {
    // TODO - re-implement
    return false;
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
