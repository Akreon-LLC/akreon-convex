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

/** @brief nonneg cone class
 *
 * R^n_+ = set of all nonneg R
 */
template <linalg::VectorLike Slack>
struct NonnegOrthantCone : public Cone<Slack, NonnegOrthantCone<Slack>> {
  /// @brief bring into scope alias from cone base
  using ConeBase = Cone<Slack, NonnegOrthantCone<Slack>>;

  /// @brief helpers to get T and M
  using ConeBase::M;
  using typename ConeBase::T;

  /**
   * @brief default interior point
   *
   * @return vector
   */
  constexpr Slack x0_impl() const {
    Slack s{};
    for (size_t i = 0; i < M; ++i) s[i] = T(1);
    return s;
  }

  /**
   * @brief in interior of K
   *
   * @param slack vector
   *
   * @return bool
   */
  constexpr bool in_K_impl(const Slack& x) const {
    for (size_t i = 0; i < M; ++i)
      if (x(i) <= T(0)) return false;
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
    return T(0);
  }
};

}  // namespace conic
