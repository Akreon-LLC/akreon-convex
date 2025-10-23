/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "concave_fn.hpp"

namespace convex {

/**
 * @brief geometric mean f(x) = (\prod_{i=1}^n x_i)^{1/n} is concave on dom f =
 * R^n_++
 */
template <linalg::RnSpace Domain>
class GeoMeanFn : public ConcaveFn<Domain, GeoMeanFn<Domain>> {
 public:
  /// @brief bring into scope the concave fn alias / helpers
  using ConvexBase = ConcaveFn<Domain, GeoMeanFn<Domain>>;
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
    return linalg::constexpr_nrt<T, N>(prod);
  }

  /**
   * @brief hypograph impl
   *
   * @return hypograph
   */
  constexpr auto hypograph_impl() const {
    // (dakre) TODO - impl
    return false;
  }
};

}  // namespace convex
