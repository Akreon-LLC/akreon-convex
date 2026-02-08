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
    s(M - 1) = T(1);
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
    T res{};
    for (size_t i = 0; i < M - 1; ++i) res += x(i) * x(i);
    res = linalg::constexpr_sqrt(res);
    return res < x(M - 1);
  }

  /**
   * @brief fraction to boundary dK
   *
   * sigma_x = \frac{1}{sup{alpha | x - alpha p in K}}
   * p = [ delta x; delta t ]
   * ||x - alpha delta(x)||_2 <= (t + alpha delta(t))^2
   * ||x - alpha delta(x)||_2^2 <= (t + alpha delta(t))^2
   * (x - alpha delta(x))^2 - (t + alpha delta(t))^2 <= 0
   *
   * a alpha^2 + b alpha + c <= 0
   *
   * a = ||delta(x)||^2 - delta(t)^2
   * b = 2(x^T delta(x) - t delta(t))
   * c = ||x||^2 - t^2 < 0... strictly in int(K)
   *
   * alpha_max = piecewise{
   *   \frac{-b + \sqrt(b^2 - 4 ac)}{2a}, a > 0
   *   \frac{-c}{b}, a = 0 & b > 0
   *   +inf, a < 0
   * }
   *
   * @param x \in K
   * @param p \in E (delta dir)
   *
   * @return \alpha_max
   */
  constexpr std::optional<T> alpha_max_impl(
      const Slack& x, const linalg::Vector<T, M>& p) const {
    const auto var = linalg::splice<M - 1>(x);
    const auto t = x(M - 1);
    const auto delta_var = linalg::splice<M - 1>(p);
    const auto delta_t = p(M - 1);

    const auto a = linalg::L2(delta_var) - delta_t * delta_t;
    const auto b = T(2) * (linalg::dot(var, delta_var) - t * delta_t);
    const auto c = linalg::L2(var) - t * t;

    if (a > T(0))
      return a;
    else if (a == T(0) && b > T(0))
      return T(-1) * c / b;
    return std::nullopt;
  }
};

}  // namespace conic
