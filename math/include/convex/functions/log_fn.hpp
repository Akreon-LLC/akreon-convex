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
 * @brief log function
 *
 * f(x) = log(x), x \in R_++
 */
template <linalg::RnSpace Domain, FnTag Tag = FnTag::Concave>
  requires(linalg::space_dim_v<Domain> == 1)
class LogFn : public std::conditional_t<Tag == FnTag::Concave,
                                        ConcaveFn<Domain, LogFn<Domain>>,
                                        ConvexFn<Domain, LogFn<Domain, Tag>>> {
 public:
  /// @brief bring into scope convex fn alias / helpers
  using ConvexBase = std::conditional_t<Tag == FnTag::Concave,
                                        ConcaveFn<Domain, LogFn<Domain, Tag>>,
                                        ConvexFn<Domain, LogFn<Domain, Tag>>>;
  using ConvexBase::Dim;
  using ConvexBase::N;
  using typename ConvexBase::Point;
  using typename ConvexBase::T;

  /**
   * @brief eval impl
   *
   * @param x \in R_++
   *
   * @return log(x) \in R
   */
  constexpr std::optional<Point> eval_impl(const Point& x) const {
    if (x(0) <= T(0)) return std::nullopt;
    return linalg::constexpr_ln(x);
  }

  /**
   * @brief hypograph impl
   *
   * C = { (x, t) | log(x) >= t } \equiv
   * C = { (x, t) | exp(x) >= exp(t) } \equiv
   * C = { (x, t) | exp(x - t) >= 1 }
   *
   * @return cone
   */
  constexpr auto hypograph_impl() const {
    // TODO - impl
    return true;
  }

  /**
   * @brief epigraph impl
   *
   * C = { (x, t) | -log(x) <= t } \equiv
   * C = { (x, t) | log(x) >= -t } \equiv
   * C = { (x, t) | exp(x) >= exp(-t) } \equiv
   * C = { (x, t) | exp(x + t) >= 1 }
   *
   * @return cone
   */
  constexpr auto epigraph_impl() const {
    // TODO -impl
    return true;
  }

  /**
   * @brief negate impl
   *
   * @return negated log
   */
  constexpr auto negate_impl() const {
    if constexpr (Tag == FnTag::Convex)
      return LogFn<Domain, FnTag::Concave>();
    else
      return LogFn<Domain, FnTag::Convex>();
  }
};
}  // namespace convex
