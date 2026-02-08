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

/** @brief power cone class
 *
 * K_pow = { (x,t) \in R_++ x R | \prod(x_i)^a_i >= |t|, sum(a_i) == 1, a_i >= 0
 * } a \in Q (set of rational numbers)
 */
template <linalg::VectorLike Slack>
struct PowerCone : public Cone<Slack, PowerCone<Slack>> {
  /// @brief bring into scope alias from cone base
  using ConeBase = Cone<Slack, PowerCone<Slack>>;

  /// @brief helpers to get T and M
  using ConeBase::M;
  using typename ConeBase::T;

  /// @brief alpha vec
  using AlphaVec = linalg::Vector<T, M - 1>;

  /**
   * @brief PowerCone
   *
   * @param alphas \in Q^n
   */
  constexpr explicit PowerCone(const AlphaVec& alphas_in) : alphas(alphas_in) {}

  /**
   * @brief default interior point
   *
   * @return vector
   */
  constexpr Slack x0_impl() const {
    Slack s{};
    for (size_t i = 0; i < M - 1; ++i) s(i) = T(1);
    return s;
  }

  /**
   * @brief in interior of K
   *
   * @param slack vector
   *
   * @return bool
   */
  constexpr bool in_cone_impl(const Slack& x) const {
    if (T(x(M - 1)) == T(0)) {
      for (size_t i = 0; i < M - 1; ++i)
        if (x(i) < T(0)) return false;
      return true;
    } else {
      T res{};
      for (size_t i = 0; i < M - 1; ++i) {
        if (x(i) < T(0)) return false;
        res += alphas(i) * linalg::constexpr_ln(x(i)).value();
      }
      return res > linalg::constexpr_ln(linalg::constexpr_abs(x(M - 1)));
    }
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

  /** @brief member vars for power cone */
  AlphaVec alphas;
};

}  // namespace conic
