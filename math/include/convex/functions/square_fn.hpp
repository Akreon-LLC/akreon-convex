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

template <linalg::RnSpace Domain, FnTag Tag = FnTag::Convex>
  requires(linalg::space_dim_v<Domain> == 1)
class SquareFn : public ConvexFn<Domain, SquareFn<Domain, Tag>> {
 public:
  /// @brief bring into scope convex fn alias / helpers
  using ConvexBase = ConvexFn<Domain, SquareFn<Domain, Tag>>;
  using ConvexBase::Dim;
  using ConvexBase::N;
  using typename ConvexBase::Point;
  using typename ConvexBase::T;

  /**
   * @brief eval impl
   *
   * @param x \in R
   *
   * @return x^2 \in R
   */
  constexpr std::optional<T> eval_impl(const Point& x) const {
    if constexpr (Tag == FnTag::Convex)
      return x(0) * x(0);
    else
      return T(-1) * x(0) * x(0);
  }

  /**
   * @brief negate impl
   *
   * @return negated f
   */
  constexpr auto negate_impl() const {
    if constexpr (Tag == FnTag::Concave)
      return SquareFn<Domain, FnTag::Convex>();
    else
      return SquareFn<Domain, FnTag::Concave>();
  }

  /**
   * @brief canonicalize_impl
   *
   * (1) f(x) = x^2
   * K = { (x,t) | x^2 <= t }
   * K_rsoc = { (u, v, w) | ||u||_2^2 <= 2 v w, v > 0, w >= 0 }
   *
   * let u = x, v = 1/2, w = t
   * [ 1 0 ] [ x; t ] + [ 0; 1/2; 0 ] -> (x, 1/2, t)
   * [ 0 0 ]
   * [ 0 1 ]
   *
   * (2) f(x) = -x^2
   * K = { (x,t) | -x^2 >= t }... { (x,t) | x^2 <= -t }
   *
   * let u = x, v = 1/2, w = -t
   * [ 1  0 ] [ x; t ] + [ 0; 1/2; 0 ] -> (x, 1/2, -t)
   * [ 0  0 ]
   * [ 0 -1 ]
   *
   * @return cone block
   */
  constexpr auto canonicalize_impl() const {
    linalg::Matrix<T, 3, 2> F;
    F[0][0] = T(1);
    linalg::Vector<T, 3> g;
    g[1] = T(0.5);
    if constexpr (Tag == FnTag::Convex) {
      F[2][1] = T(1);
    } else {
      F[2][1] = T(-1);
    }
    const auto map = affine::AffineMap<decltype(F)>(F, g);
    const auto rsoc = conic::RotatedSecondOrderCone<linalg::Vector<T, 3>>();
    return std::tuple(conic::ConeBlock(map, rsoc));
  }
};

}  // namespace convex
