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

#include "convex/sets/power_cone.hpp"
#include "concave_fn.hpp"

namespace convex {

/**
 * @brief geometric mean f(x) = (\prod_{i=1}^n x_i)^{1/n} is concave on dom f =
 * R^n_++
 */
template <linalg::RnSpace Domain, FnTag Tag = FnTag::Concave>
class GeoMeanFn
    : public std::conditional_t<Tag == FnTag::Concave,
                                ConcaveFn<Domain, GeoMeanFn<Domain, Tag>>,
                                ConvexFn<Domain, GeoMeanFn<Domain, Tag>>> {
 public:
  /// @brief bring into scope the concave fn alias / helpers
  using ConvexBase =
      std::conditional_t<Tag == FnTag::Concave,
                         ConcaveFn<Domain, GeoMeanFn<Domain, Tag>>,
                         ConvexFn<Domain, GeoMeanFn<Domain, Tag>>>;
  using ConvexBase::Dim;
  using ConvexBase::N;
  using typename ConvexBase::Point;
  using typename ConvexBase::T;

  /// @brief alias for affine map (codomain 1-dim)
  using Mat = linalg::Matrix<T, 1, N>;

  /**
   * @brief evaluation impl
   *
   * @param x \in dom f
   *
   * @return y \in R
   */
  constexpr std::optional<T> eval_impl(const Point& x) const {
    T prod(1);
    std::for_each(x.begin(), x.end(), [&](const auto& pt) { prod *= pt; });
    if constexpr (Tag == FnTag::Convex)
      return T(-1) * linalg::constexpr_nrt<T>(prod, N);
    else
      return linalg::constexpr_nrt<T>(prod, N);
  }

  /**
   * @brief hypograph impl
   *
   * F(x) = (\prod_i^n x_i)^{1/n}
   * graph(F(x)) = { (x, \prod_i^n x_i)^{1/n} | x \in dom f}
   * hypo(F(x)) = { (x, t) | (\prod_i^n x_i)^{1/n} >= t, x \in dom f }
   * \equiv = { (x, t) \in R^n_+ x R | \prod_i^n x_i^{1/n} >= |t| } which is the
   * power cone
   *
   * @return hypograph
   */
  constexpr auto hypograph_impl() const {
    return PowerCone<Domain, ViewTag::HypographCone>().cone_view();
  }

  /**
   * @brief epigraph impl
   *
   * C = { (x, t) | -\prod_i^n x_i^{1/n} >= |t| }
   *
   * @return epigraph
   */
  constexpr auto epigraph_impl() const {
    return PowerCone<Domain, ViewTag::EpigraphCone>().cone_view();
  }

  /**
   * @brief negate impl
   *
   * @return negated f
   */
  constexpr auto negate_impl() const {
    if constexpr (Tag == FnTag::Convex)
      return GeoMeanFn<Domain, FnTag::Concave>();
    else
      return GeoMeanFn<Domain, FnTag::Convex>();
  }
};

}  // namespace convex
