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

#include "cone.hpp"

namespace conic {

/**
 * @brief rsoc cone class
 *
 * K_rsoc = { (u, v, w) | ||u||_2^2 <= 2 v w, v >= 0, w >= 0 }
 */
template <linalg::VectorLike Slack>
struct RotatedSecondOrderCone
    : public Cone<Slack, RotatedSecondOrderCone<Slack>> {
  /// @brief bring into scope alias from cone base
  using ConeBase = Cone<Slack, RotatedSecondOrderCone<Slack>>;

  /// @brief helpers to get T and M
  using ConeBase::M;
  using typename ConeBase::T;

  static_assert(M == 3, "RSOC must be a triplet");

  /**
   * @brief default interior point
   *
   * @return vector
   */
  constexpr Slack x0_impl() const { return Slack{T(0), T(1), T(1)}; }

  /**
   * @brief in interior of K
   *
   * @param slack vector
   *
   * @return bool
   */
  constexpr bool in_cone_impl(const Slack& x) const {
    return x(0) * x(0) < T(2) * x(1) * x(2);
  }

  /**
   * @brief fraction to boundary dK
   *
   * sigma_x = \frac{1}{sup{alpha | x - alpha * p in K}}
   * p = [ delta(u); delta(v); delta(w) ]
   * new coords = [(u - alpha * delta(u), (v - alpha * delta(v)), (w - alpha *
   * delta(w)]
   *
   * ||u - alpha*delta(u)||_2^2 <= 2 * (v - alpha * delta(v)) * (w - alpha *
   * delta(w))
   *
   * a alpha^2 + b alpha + c <= 0
   *
   * a = ||delta(u)||^2 - 2 * delta(v) * delta(w)... < 0... strictly in int(K)
   * b = -2 u^T delta(u) + 2 (v delta(w) + w delta(v))
   * c = ||u||^2 - 2 * v * w
   *
   * (i) alpha_quad = piecewise{
   *   \frac{-b + \sqrt(b^2 - 4 ac)}{2a}, a > 0
   *   \frac{-c}{b}, a = 0 & b > 0
   *   +inf, a < 0
   * }
   *
   * v - alpha * delta(v) >= 0
   * (ii) alpha_v = v / delta(v) if delta(v) > 0
   *
   * w - alpha * delta(w) >= 0
   * (iii) alpha_w = w / delta(w) if delta(w) > 0
   *
   * @param x \in K
   * @param p \in E
   *
   * @return \alpha_max
   */
  constexpr std::optional<T> alpha_max_impl(
      const Slack& x, const linalg::Vector<T, M>& p) const {
    const T u = x(0), v = x(1), w = x(2);
    const T du = p(0), dv = p(1), dw = p(2);

    const T a = du * du - T(2) * dv * dw;
    const T b = T(-2) * u * du + T(2) * (v * dw + w * dv);
    const T c = u * u - T(2) * v * w;

    std::optional<T> alpha;
    if (a > T(0))
      alpha = a;
    else if (a == T(0) && b > T(0))
      alpha = b;

    if (dv > T(0)) {
      const T d = v / dv;
      alpha = alpha.has_value() ? (alpha.value() > d ? d : alpha.value()) : d;
    }

    if (dw > T(0)) {
      const T d = w / dw;
      alpha = alpha.has_value() ? (alpha.value() > d ? d : alpha.value()) : d;
    }

    return alpha;
  }
};

}  // namespace conic
