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

/** @brief zoc cone class
 *
 * K_zoc = { x | x == 0 }
 */
template <linalg::VectorLike Slack>
struct ZeroOrderCone : public Cone<Slack, ZeroOrderCone<Slack>> {
  /// @brief bring into scope alias from cone base
  using ConeBase = Cone<Slack, ZeroOrderCone<Slack>>;

  /// @brief helpers to get T and M
  using ConeBase::M;
  using typename ConeBase::T;

  /**
   * @brief default interior point
   *
   * @return vector
   */
  constexpr Slack x0_impl() const { return Slack{}; }

  /**
   * @brief in interior of K
   *
   * @param slack vector
   *
   * @return bool
   */
  constexpr bool in_K_impl(const Slack& /*x*/) const {
    /// (dakre) always empty \equiv null set!
    return false;
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
