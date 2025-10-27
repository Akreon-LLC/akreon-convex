/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "../sets/rsoc.hpp"
#include "convex_fn.hpp"

namespace convex {
/**
 * @brief quadratic over linear function
 *
 * F(x) = x^2 / y, where y > 0 so (x, y) \in R x R_++ or more generally
 * F(x) = x^T x / y, where y > 0 and x \in R^n so R^n x R_++
 */
template <linalg::RnSpace Space, FnTag Tag = FnTag::Convex>
  requires(linalg::space_dim_v<Space> >= 2)
class QuadOverLinearFn : public ConvexFn<Space, QuadOverLinearFn<Space, Tag>> {
 public:
  /// @brief bring into scope convex fn alias / helpers
  using ConvexBase = ConvexFn<Space, QuadOverLinearFn<Space, Tag>>;
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
  constexpr std::optional<T> eval_impl(const Point& x_in) const {
    const auto y = x_in(N - 1);
    const auto x = std::get<0>(linalg::splice<N - 1>(x_in));
    if (y <= T(0)) return std::nullopt;
    return linalg::dot(x, x) / y;
  }

  /**
   * @brief epigraph impl
   *
   * F(x) = ||x||_2^2 / y
   * graph(F(x)) = { (x, y, ||x||_2^2 / y) | x \in R^n, y \in R_++, t \in R_+ }
   * epi(F(x)) = { (x, y, t) | ||x||_2^2 / y, (x, y) \in dom f, t >= 0 }...
   * ||x||_2^2 <= t y
   *
   * rotated SOC = { (x, y, z) | ||x||_2^2 <= y z, y >= 0, z >= 0 }
   * epi(F(x)) as RSOC = { (x, y, z) | ||x||_2^2 <= y t, y > 0, t >= 0 }
   *
   * @return L1 cone
   */
  constexpr auto epigraph_impl() const {
    using LiftedSpace = linalg::Matrix<T, 1, N + 1>;
    return RotatedSOC<LiftedSpace, NormConeType::L2CC>().cone_view();
  }
};
}  // namespace convex
