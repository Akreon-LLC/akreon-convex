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

#include "convex/cone_view.hpp"
#include "convex/convex_fn_traits.hpp"

namespace convex {
/**
 * @brief base convex function CRTP
 *
 * Similar space constraints on R^n with convex set domain
 */
template <ConvexSetLike Domain, typename Derived>
class ConvexFn {
 public:
  /// @brief alias to extract type traits from Domain convex set
  using Space = set_space_t<Domain>;
  using T = set_type_t<Domain>;
  using Point = set_point_t<Domain>;

  /// @brief helper to get dimension size of space & size of point
  static constexpr size_t Dim = set_dim_v<Domain>;
  static constexpr size_t N = linalg::vec_size_v<Point>;

  /**
   * @brief operator overload for evaluation fn
   *
   * @param point x \in domain
   *
   * @return y \in R
   */
  constexpr std::optional<T> operator()(const Point& x) const {
    return self().eval_impl(x);
  }

  /**
   * @brief eval
   *
   * F(x) -> y \in R
   *
   * @param point x \in domain
   *
   * @return y (scalar in R)
   */
  constexpr std::optional<T> eval(const Point& x) const {
    return self().eval_impl(x);
  }

  /**
   * @brief epigraph of f
   *
   * Region above the graph of f defined by t
   *
   * f : R^n -> R
   * graph(f) = { (x, f(x)) | x \in dom f } \in R^n
   * epi(F) = { (x, t) | x \in dom f, f(x) <= t } \in R^{n+1}
   *
   * @return epigraph of func
   */
  constexpr auto epigraph() const { return self().epigraph_impl(); }

  /**
   * @brief negate f -> concave function or id
   *
   * @return negated convex fn
   */
  constexpr auto negate() const { return self().negate_impl(); }

  /**
   * @brief self
   *
   * @return derived class instance
   */
  constexpr auto self() const { return static_cast<const Derived&>(*this); }
};
}  // namespace convex
