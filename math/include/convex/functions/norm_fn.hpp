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
 * @brief norm convex function wrapper
 */
template <linalg::RnSpace Domain, linalg::NormLike Norm,
          FnTag Tag = FnTag::Convex>
  requires(linalg::space_dim_v<Domain> == linalg::norm_size_v<Norm>)
class NormFn : public ConvexFn<Domain, NormFn<Domain, Norm, Tag>> {
 public:
  /// @brief bring into scope the convex fn alias / helpers
  using ConvexBase = ConvexFn<Domain, NormFn<Domain, Norm, Tag>>;
  using ConvexBase::Dim;
  using ConvexBase::N;
  using typename ConvexBase::Point;
  using typename ConvexBase::T;

  /**
   * @brief eval impl
   *
   * @param point x \in domain
   *
   * @return y (scalar \in R)
   */
  constexpr std::optional<T> eval_impl(const Point& x) const {
    if constexpr (Tag == FnTag::Convex)
      return Norm()(x);
    else
      return T(-1) * Norm()(x);
  }

  /**
   * @brief negate impl
   *
   * if concave tag -> convex
   * else convex tag -> concave
   *
   * @return negated f
   */
  constexpr auto negate_impl() const {
    if constexpr (Tag == FnTag::Concave)
      return NormFn<Domain, Norm, FnTag::Convex>();
    else
      return NormFn<Domain, Norm, FnTag::Concave>();
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
