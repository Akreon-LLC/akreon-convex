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

/// @brief power cone class
template <linalg::VectorLike Slack>
struct PowerCone : public Cone<Slack, PowerCone<Slack>> {
  /// @brief bring into scope alias from cone base
  using ConeBase = Cone<Slack, PowerCone<Slack>>;

  /// @brief helpers to get T and M
  using ConeBase::M;
  using typename ConeBase::T;

  /**
   * @brief default interior point
   *
   * @param vector of indices
   *
   * @return vector
   */
  constexpr Slack x0_impl() const {
    /// (dakre) TODO - acquire interior point
    return Slack{};
  }

  /**
   * @brief in interior of K
   *
   * @param primal vector
   *
   * @return bool
   */
  constexpr bool in_K_impl(const Slack& x) const {
    /// (dakre) TODO - perform is interior check
    (void)x;
    return true;
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
    /// (dakre) TODO - compute alpha max
    return T(0);
  }
};

}  // namespace conic
