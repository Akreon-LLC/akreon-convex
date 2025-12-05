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
 * @brief max convex function CRTP
 *
 * F(x) = max { x_1, x_2, ... x_n }, x \in R^n
 */
template <linalg::RnSpace Domain, FnTag Tag = FnTag::Convex>
class MaxFn : public ConvexFn<Domain, MaxFn<Domain, Tag>> {
 public:
  /// @brief bring into scope convex fn alias / helpers
  using ConvexBase = ConvexFn<Domain, MaxFn<Domain, Tag>>;
  using ConvexBase::Dim;
  using ConvexBase::N;
  using typename ConvexBase::Point;
  using typename ConvexBase::T;

  /**
   * @brief eval impl
   *
   * @param Point \in domain
   *
   * @return y \in R
   */
  constexpr std::optional<T> eval_impl(const Point& x) const {
    T mx(0);
    std::for_each(x.begin(), x.end(), [&mx](const auto& pt) {
      if (mx < pt) mx = pt;
    });
    if constexpr (Tag == FnTag::Convex)
      return mx;
    else
      return T(-1) * mx;
  }

  /**
   * @brief negate impl
   *
   * if concave -> convex
   * else convex -> concave
   *
   * @return negated f
   */
  constexpr auto negate_impl() const {
    if constexpr (Tag == FnTag::Concave)
      return MaxFn<Domain, FnTag::Convex>();
    else
      return MaxFn<Domain, FnTag::Concave>();
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
