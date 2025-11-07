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

#include "convex/sets/exp_cone.hpp"
#include "convex_fn.hpp"

namespace convex {
/**
 * @brief exp function
 *
 * f(x) = e^x, x \in R
 */
template <linalg::RnSpace Domain, FnTag Tag = FnTag::Convex>
  requires(linalg::space_dim_v<Domain> == 1)
class ExpFn : public std::conditional_t<Tag == FnTag::Convex,
                                        ConvexFn<Domain, ExpFn<Domain, Tag>>,
                                        ConcaveFn<Domain, ExpFn<Domain, Tag>>> {
 public:
  /// @brief bring into scope convex fn alias / helpers
  using ConvexBase = std::conditional_t<Tag == FnTag::Convex,
                                        ConvexFn<Domain, ExpFn<Domain, Tag>>,
                                        ConcaveFn<Domain, ExpFn<Domain, Tag>>>;
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
   * @return exp(x) \in R
   */
  constexpr std::optional<T> eval_impl(const Point& x) const {
    if constexpr (Tag == FnTag::Convex)
      return linalg::constexpr_exp(x(0));
    else
      return T(-1) * linalg::constexpr_exp(x(0));
  }

  /**
   * @brief epigraph impl
   *
   * C = { (x, t) | exp(x) <= t }, expressed as exp cone
   * K_exp = { (x, t) | e^{x} <= t }
   * exp coords (x, 1, z=t)
   *
   * @return exp cone
   */
  constexpr auto epigraph_impl() const
    requires(Tag == FnTag::Convex)
  {
    return ExpCone<Domain, ViewTag::LiftedCone>().cone_view();
  }

  /**
   * @brief hypograph impl
   *
   * C = { (x, t) | -exp(x) >= t }, to exp cone
   * K_exp = { (x, -t) | e^{x} <= -t }
   * exp coords (x, 1, z=-t)
   *
   * @return hypo cone
   */
  constexpr auto hypograph_impl() const
    requires(Tag == FnTag::Concave)
  {
    return ExpCone<Domain, ViewTag::LiftedCone>(
               default_exp_cone_map<CoordMat>(T(-1)))
        .cone_view();
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
};
}  // namespace convex
