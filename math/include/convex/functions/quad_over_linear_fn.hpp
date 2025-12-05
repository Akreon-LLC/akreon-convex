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
 * @brief quadratic over linear function
 *
 * F(x) = x^2 / y, where y > 0 so (x, y) \in R x R_++ or more generally
 * F(x) = x^T x / y, where y > 0 and x \in R^n so R^n x R_++
 */
template <linalg::RnSpace Domain, FnTag Tag = FnTag::Convex>
  requires(linalg::space_dim_v<Domain> >= 2)
class QuadOverLinearFn
    : public ConvexFn<Domain, QuadOverLinearFn<Domain, Tag>> {
 public:
  /// @brief bring into scope convex fn alias / helpers
  using ConvexBase = ConvexFn<Domain, QuadOverLinearFn<Domain, Tag>>;
  using ConvexBase::Dim;
  using ConvexBase::N;
  using typename ConvexBase::Point;
  using typename ConvexBase::T;

  /**
   * @brief eval impl
   *
   * @param point \in domain
   *
   * @return y \in R
   */
  constexpr std::optional<T> eval_impl(const Point& x_in) const {
    const auto y = x_in(N - 1);
    const auto x = std::get<0>(linalg::splice<N - 1>(x_in));
    if (y <= T(0)) return std::nullopt;
    if constexpr (Tag == FnTag::Convex)
      return linalg::dot(x, x) / y;
    else
      return T(-1) * linalg::dot(x, x) / y;
  }

  /**
   * @brief negate impl
   *
   * @return negated f
   */
  constexpr auto negate_impl() const {
    if constexpr (Tag == FnTag::Concave)
      return QuadOverLinearFn<Domain, FnTag::Convex>();
    else
      return QuadOverLinearFn<Domain, FnTag::Concave>();
  }

  /**
   * @brief canonicalize_impl
   *
   * abs(x) -> epi or hypo -> homogenized cone
   *
   * @return cone block
   */
  constexpr auto canonicalize_impl() const {
    if constexpr (Tag == FnTag::Convex) {
    }
  }
};

}  // namespace convex
