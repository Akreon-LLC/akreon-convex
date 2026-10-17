/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cone.hpp"

namespace conic {

/** @brief exp cone class
 *
 * K_exp = { (u, v, w) | v exp(u/v) <= w, v > 0 }
 *
 * To support F block being affine in x then aux vars the coords
 * are strided by 3*N nominally. So u coords are the first M/3 rows, v are the
 * next M/3 rows, and w are the final rows
 */
template <linalg::VectorContainer Slack>
struct ExpCone : public Cone<Slack, ExpCone<Slack>, ConeType::ASYMMETRIC> {
  /// @brief bring into scope alias from cone base
  using ConeBase = Cone<Slack, ExpCone<Slack>, ConeType::ASYMMETRIC>;

  /// @brief helpers to get T and M
  using ConeBase::M;
  using typename ConeBase::T;

  static_assert(M % 3 == 0, "Exp cone must be a triplet");

  /**
   * @brief default interior point
   *
   * @return vector
   */
  constexpr Slack x0_impl() const {
    Slack s{};
    for (std::size_t i = 0; i < M / 3; ++i) s(i) = T(0);
    for (std::size_t i = M / 3; i < 2 * M / 3; ++i) s(i) = T(1);
    for (std::size_t i = 2 * M / 3; i < M; ++i) s(i) = T(2);
    return s;
  }

  /**
   * @brief in interior of K
   *
   * @param x in K
   *
   * @return bool
   */
  constexpr bool in_cone_impl(const Slack& x) const {
    for (std::size_t i = 0; i < M / 3; ++i) {
      const auto u = x(i);
      const auto v = x(M / 3 + i);
      const auto w = x(2 * M / 3 + i);
      if (v <= T(0) || v * linalg::constexpr_exp(u / v) >= w) return false;
    }
    return true;
  }

  /**
   * @brief fraction to boundary dK
   *
   * @param x in K
   * @param p in E
   *
   * @return alpha_max
   */
  constexpr T alpha_max_impl(const Slack& x,
                             const linalg::Vector<T, M>& p) const {
    (void)x, (void)p;
    /// (dakre) TODO - compute alpha max
    return T(0);
  }
};

} // namespace conic
