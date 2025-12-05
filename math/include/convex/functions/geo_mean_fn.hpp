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
 * @brief geometric mean f(x) = (\prod_{i=1}^n x_i)^{1/n} is concave on dom f =
 * R^n_++
 */
template <linalg::RnSpace Domain, FnTag Tag = FnTag::Concave>
class GeoMeanFn : public ConvexFn<Domain, GeoMeanFn<Domain, Tag>> {
 public:
  /// @brief bring into scope the concave fn alias / helpers
  using ConvexBase = ConvexFn<Domain, GeoMeanFn<Domain, Tag>>;
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

  /**
   * @brief canonicalize_impl
   *
   * (1) f(x) = prod_i^n(x_i)^{1/n}
   * K_geo = { (x, t) | prod_i^n(x_i)^{1/n} >= t }... x in R_+
   * power cone = { (x,t) \in R_+ x R | prod(x_i)^alpha_i >= |t|, sum(a_i) = 1,
   * a_i >= 0 }
   *
   * I [ x; t ]
   *
   * (2) f(x) = -prod_i^n(x_i)^{1/n}
   * K_geo = { (x, t) | prod_i^n(x_i)^{1/n} >= -t }... x in R_+
   *
   * [ 1  0 ] [ x; t ]
   * [ 0 -1 ]
   *
   * @return cone block
   */
  constexpr auto canonicalize_impl() const {
    auto F = linalg::make_identity<T, 2>();
    linalg::Vector<T, 2> g;
    if constexpr (Tag == FnTag::Convex) {
      F[1][1] = T(-1);
    }
    const auto map = affine::AffineMap<decltype(F)>(F, g);
    const auto power_cone = conic::PowerCone<decltype(F)>();
    return std::tuple(conic::ConeBlock(map, power_cone));
  }
};

}  // namespace convex
