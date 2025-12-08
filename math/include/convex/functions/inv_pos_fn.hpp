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
 * @brief inv pos function
 *
 * f(x) = 1/x, x in R_++
 */
template <linalg::RnSpace Domain, FnTag Tag = FnTag::Convex>
  requires(linalg::space_dim_v<Domain> == 1)
class InvPosFn : public ConvexFn<Domain, InvPosFn<Domain, Tag>> {
 public:
  /// @brief bring into scope convex fn alias / helpers
  using ConvexBase = ConvexFn<Domain, InvPosFn<Domain, Tag>>;
  using ConvexBase::Dim;
  using ConvexBase::N;
  using typename ConvexBase::Point;
  using typename ConvexBase::T;

  /**
   * @brief eval impl
   *
   * @param x \in R_++
   *
   * @return res \in R_++ or R_-- (depending on convex or concave)
   */
  constexpr std::optional<T> eval_impl(const Point& x) const {
    if (x(0) <= T(0)) return std::nullopt;
    if constexpr (Tag == FnTag::Convex) {
      return T(1) / x(0);
    } else {
      return T(-1) / x(0);
    }
  }

  /**
   * @brief negate impl
   *
   * @return negated f
   */
  constexpr auto negate_impl() const {
    if constexpr (Tag == FnTag::Convex)
      return InvPosFn<Domain, FnTag::Concave>();
    else
      return InvPosFn<Domain, FnTag::Convex>();
  }

  /**
   * @brief canonicalize_impl
   *
   * (1) f(x) = 1/x
   * K = { (x, t) | 1/x <= t, x in R_++ implies t >= 0 }
   * K_rsoc = { (u, v, w) | ||u||_2^2 <= 2 v w, v > 0, w >= 0 }
   *
   * let u = sqrt(2), v = x, w = t
   * [ 0 0 ] [ x; t ] + [ sqrt(2); 0; 0] -> (sqrt(2), x, t)
   * [ 1 0 ]
   * [ 0 1 ]
   *
   * (2) f(x) = -1/x
   * K = { (x, t) | 1/x <= -t, x in R_++ t in R_- }
   *
   * let u = sqrt(2), v = x, w = -t
   * [ 0  0 ] [ x; t ] + [ sqrt(2); 0; 0 ]
   * [ 1  0 ]
   * [ 0 -1 ]
   *
   * @return cone block
   */
  constexpr auto canonicalize_impl() const {
    linalg::Matrix<T, 3, 2> F;
    linalg::Vector<T, 3> g;
    g[0] = linalg::constexpr_sqrt(T(2));
    if constexpr (Tag == FnTag::Convex) {
      F[1][0] = T(1);
      F[2][1] = T(1);
    } else {
      F[1][0] = T(1);
      F[2][1] = T(-1);
    }
    const auto map = affine::AffineMap<decltype(F)>(F, g);
    const auto rsoc = conic::RotatedSecondOrderCone<linalg::Vector<T, 3>>();
    return std::tuple(conic::ConeBlock(map, rsoc));
  }
};

}  // namespace convex
