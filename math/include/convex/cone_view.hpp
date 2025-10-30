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
   * @brief compute slack - project a point x onto cone coordinates
   *
   * F z + b = s, s \in K, z = [ x ; t ]
   *
   * @param AffineMap (A \in R^{n x n} and g \in R^n)
   * @param point (x, t) \in R^n x R
   *
   * @return s \in R^n x R
   */
  constexpr auto compute_slack(const AffineMap<Mat>& map,
                               const std::tuple<Point, T>& x_in) const {
    const auto x = std::get<0>(x_in);
    const auto t = std::get<1>(x_in);

    const auto x_proj = map(x);
    return std::tuple(x_proj, t);
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
