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
 * @brief abs function
 *
 * f(x) = |x|, x \in R
 */
template <linalg::RnSpace Domain, FnTag Tag = FnTag::Convex>
  requires(linalg::space_dim_v<Domain> == 1)
class AbsFn : public ConvexFn<Domain, AbsFn<Domain, Tag>> {
 public:
  /// @brief bring into scope convex fn alias / helpers
  using ConvexBase = ConvexFn<Domain, AbsFn<Domain, Tag>>;
  using ConvexBase::Dim;
  using ConvexBase::N;
  using typename ConvexBase::Point;
  using typename ConvexBase::T;

  /**
   * @brief eval impl
   *
   * @param x \in R
   *
   * @return abs(x) \in R
   */
  constexpr std::optional<T> eval_impl(const Point& x) const {
    if constexpr (Tag == FnTag::Convex)
      return linalg::constexpr_abs(x(0));
    else
      return T(-1) * linalg::constexpr_abs(x(0));
  }

  /**
   * @brief negate impl
   *
   * @return negated f
   */
  constexpr auto negate_impl() const {
    if constexpr (Tag == FnTag::Concave) {
      return AbsFn<Domain, FnTag::Convex>();
    } else {
      return AbsFn<Domain, FnTag::Concave>();
    }
  }

  /**
   * @brief canonicalize_impl
   *
   * (1) abs(x) -> epi -> cones
   * f(x) = |x|
   * K_epi = { (x,t) | |x| <= t }
   *
   * let u >= |x|... first convex constraint
   * 0 <= t - u... second convex constraint
   * u >= 0... third convex constraint
   * t >= 0... fourth convex constraint
   *
   * -u <= x <= u
   * 0 <= u + x... nonneg orthant cone
   * 0 <= u - x... nonneg orthant cone
   * 0 <= t - u... nonneg orthant cone
   * u >= 0... nonneg orthant cone
   * t >= 0... nonneg orthant cone
   *
   * [  1  1 0 ] [ x; u; t ]
   * [ -1  1 0 ]
   * [  0 -1 1 ]
   * [  0  1 0 ]
   * [  0  0 1 ]
   *
   * (2) -abs(x) -> hypo -> cones
   * f(x) = - |x|
   * K_hypo = { (x,t) | -|x| >= t, t <= 0}
   *
   * let u >= |x|... first convex constraint
   * let t <= -u... second convex constraint
   * u >= 0... third convex constraint
   * t <= 0... fourth convex constraint
   *
   * 0 <= u + x... nonneg orthant cone
   * 0 <= u - x... nonneg orthant cone
   * 0 <= -u - t... nonneg orthant cone
   * u >= 0... nonneg orthant cone
   * -t >= 0... nonneg orthant cone
   *
   * [  1  1  0 ] [ x; u; t ]
   * [ -1  1  0 ]
   * [  0 -1 -1 ]
   * [  0  1  0 ]
   * [  0  0 -1 ]
   *
   * @return cone block
   */
  constexpr auto canonicalize_impl() const {
    linalg::Matrix<T, 5, 3> F;
    linalg::Vector<T, 5> g;
    if constexpr (Tag == FnTag::Convex) {
      F[0][0] = T(1), F[0][1] = T(1);
      F[1][0] = T(-1), F[1][1] = T(1);
      F[2][1] = T(-1), F[2][2] = T(1);
      F[3][1] = T(1);
      F[4][2] = T(1);
    } else {
      F[0][0] = T(1), F[0][1] = T(1);
      F[1][0] = T(-1), F[1][1] = T(1);
      F[2][1] = T(-1), F[2][2] = T(-1);
      F[3][1] = T(1);
      F[4][2] = T(-1);
    }
    const auto map = affine::AffineMap<decltype(F)>(F, g);
    const auto rplus = conic::NonnegOrthantCone<linalg::Vector<T, 1>>();
    return std::tuple(conic::ConeBlock(map, rplus, rplus, rplus, rplus, rplus));
  }
};

}  // namespace convex
