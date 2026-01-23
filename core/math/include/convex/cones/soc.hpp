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
 * @brief soc cone class
 *
 * K_soc = { (x, t) | ||x|| <= t }
 */
template <linalg::VectorLike Slack>
struct SecondOrderCone : public Cone<Slack, SecondOrderCone<Slack>> {
  /// @brief bring into scope alias from cone base
  using ConeBase = Cone<Slack, SecondOrderCone<Slack>>;

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
    s[M - 1] = T(1);
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
    T res{};
    for (size_t i = 0; i < M - 1; ++i) res += x(i) * x(i);
    res = linalg::constexpr_sqrt(res);
    return res < x(M - 1);
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
