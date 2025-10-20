/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "convex_fn.hpp"

namespace convex {
/**
 * @brief quadratic over linear function
 *
 * (dakre) TODO Fix / Finish
 *
 * F(x) = x^2 / y, where y > 0 so (x, y) \in R x R_++
 */
template <linalg::RnSpace Space>
  requires(linalg::space_dim_v<Space> == 2)
class QuadOverLinearFn : public ConvexFn<Space, QuadOverLinearFn<Space>> {
 public:
  /// @brief bring into scope convex fn alias / helpers
  using ConvexBase = ConvexFn<Space, MaxFn<Space>>;
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
  constexpr std::optional<T> eval_impl(const Point& x) const {
    if (x(1) <= T(0)) return std::nullopt;
    return (x(0) * x(0)) / x(1);
  }

  /**
   * @brief epigraph impl
   *
   * F(x) = x^2 / y, y > 0
   * graph(F(x)) = { (x, y, x^2 / y) | (x, y) \in dom f }
   * epi(F(x)) = { (x, y, t) \in R x R_++ x R_+ | x^2 / y <= t, (x, y) \in dom
   * f, t
   * >= 0 } \equiv { (x, y, t) | t y >= x^2, t >= 0, (x, y) \in dom f }
   *
   * @return L1 cone
   */
  constexpr auto epigraph_impl() const {
    using LiftedSpace = linalg::Matrix<T, 1, N + 1>;
    return false;
  }
};
}  // namespace convex
