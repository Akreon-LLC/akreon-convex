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
  constexpr bool in_K_impl(const Slack& x) const {
    return x(0) * x(0) < T(2) * x(1) * x(2);
  }

  /**
   * @brief fraction to boundary dK
   *
   * @param x \in K
   * @param p \in E
   *
   * @return \alpha_max
   */
  constexpr T alpha_max_impl(const Slack& x,
                             const linalg::Vector<T, M>& p) const {
    (void)x, (void)p;
    return T(0);
  }
};

}  // namespace conic
