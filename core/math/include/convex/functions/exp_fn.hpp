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

#include "convex_fn.hpp"

namespace convex {
/**
 * @brief exp function
 *
 * f(x) = e^x, x \in R
 */
template <linalg::RnSpace Domain, FnTag Tag = FnTag::Convex>
  requires(linalg::space_dim_v<Domain> == 1)
class ExpFn : public ConvexFn<Domain, ExpFn<Domain, Tag>> {
 public:
  /// @brief bring into scope convex fn alias / helpers
  using ConvexBase = ConvexFn<Domain, ExpFn<Domain, Tag>>;
  using ConvexBase::Dim;
  using ConvexBase::N;
  using typename ConvexBase::Point;
  using typename ConvexBase::T;

  /**
   * @brief eval impl
   *
   * @param x \in R
   *
   * @return exp(x) \in R
   */
  constexpr std::optional<T> eval_impl(const Point& x) const {
    if constexpr (Tag == FnTag::Convex)
      return linalg::constexpr_exp(x(0));
    else
      return T(-1) * linalg::constexpr_exp(x(0));
  }

  /**
   * @brief negate impl
   *
   * @return negated exp f
   */
  constexpr auto negate_impl() const {
    if constexpr (Tag == FnTag::Concave)
      return ExpFn<Domain, FnTag::Convex>();
    else
      return ExpFn<Domain, FnTag::Concave>();
  }

  /**
   * @brief canonicalize_impl
   *
   * (1) f(x) = exp(x)
   * K = { (x,t) | exp(x) <= t }
   * K_exp = { (u,v,w) | v exp(u/v) <= w, v > 0 }
   *
   * let u = x, v = 1, w = t
   *
   * [ 1 0 ] [ x; t ] + [ 0; 1; 0 ] -> (x, 1, t)
   * [ 0 0 ]
   * [ 0 1 ]
   *
   * (2) f(x) = -exp(x)
   * K_exp = { (x, t) | -exp(x) >= t }... { (x, t) | exp(x) <= -t, t <= 0 }
   *
   * [ 1  0 ] [ x; t ] + [ 0; 1; 0 ] -> (x, 1, -t)
   * [ 0  0 ]
   * [ 0 -1 ]
   *
   * @return cone block
   */
  constexpr auto canonicalize_impl() const {
    linalg::Matrix<T, 3, 2> F;
    F[0][0] = T(1);
    linalg::Vector<T, 3> g;
    g[1] = T(1);
    if constexpr (Tag == FnTag::Convex) {
      F[2][1] = T(1);
    } else {
      F[2][1] = T(-1);
    }
    const auto map = affine::AffineMap<decltype(F)>(F, g);
    const auto exp_cone = conic::ExpCone<linalg::Vector<T, 3>>();
    return std::tuple(conic::ConeBlock(map, exp_cone));
  }
};

}  // namespace convex
