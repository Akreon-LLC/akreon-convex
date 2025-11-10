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

#include "convex/sets/norm_cone.hpp"
#include "convex_fn.hpp"

namespace convex {

/**
 * @brief abs function
 *
 * f(x) = |x|, x \in R
 */
template <linalg::RnSpace Domain, FnTag Tag = FnTag::Convex>
  requires(linalg::space_dim_v<Domain> == 1)
class AbsFn : public std::conditional_t<Tag == FnTag::Convex,
                                        ConvexFn<Domain, AbsFn<Domain, Tag>>,
                                        ConcaveFn<Domain, AbsFn<Domain, Tag>>> {
 public:
  /// @brief bring into scope convex fn alias / helpers
  using ConvexBase = std::conditional_t<Tag == FnTag::Convex,
                                        ConvexFn<Domain, AbsFn<Domain, Tag>>,
                                        ConcaveFn<Domain, AbsFn<Domain, Tag>>>;
  using ConvexBase::Dim;
  using ConvexBase::N;
  using typename ConvexBase::CoordMat;
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
   * @brief epigraph impl
   *
   * C = { (x, t) | |x| <= t }
   * K_exp = { (x, t) | |x| <= t }
   *
   * @return norm cone
   */
  constexpr auto epigraph_impl() const
    requires(Tag == FnTag::Convex)
  {
    return NormCone<Domain, NormConeType::L1CC, ViewTag::LiftedCone>()
        .cone_view();
  }

  /**
   * @brief hypograph impl
   *
   * C = { (x, t) | |x| <= -t }
   *
   * @return norm cone
   */
  constexpr auto hypograph_impl() const
    requires(Tag == FnTag::Concave)
  {
    return NormCone<Domain, NormConeType::L1CC, ViewTag::LiftedCone>(
               default_cone_map<CoordMat>(linalg::Vectord<N>{}, T(-1)))
        .cone_view();
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
};

}  // namespace convex
