/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "../sets/norm_cone.hpp"
#include "convex_fn.hpp"

namespace convex {
/**
 * @brief max convex function CRTP
 *
 * F(x) = max { x_1, x_2, ... x_n }, x \in R^n
 */
template <linalg::RnSpace Space>
class MaxFn : public ConvexFn<Space, MaxFn<Space>> {
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
   * @param Point \in domain
   *
   * @return y \in R
   */
  constexpr std::optional<T> eval_impl(const Point& x) const {
    T mx(0);
    std::for_each(x.begin(), x.end(), [&mx](const auto& pt) {
      if (mx < pt) mx = pt;
    });
    return mx;
  }

  /**
   * @brief epigraph impl
   *
   * F(x) = max { x_1, x_2, ... x_n }, x \in R^n
   * graph(F(x)) = { (x, max(x)) | x \in dom f }
   * epi(F(x)) = { (x, t) | max(x) <= t, x \in dom f, t >= 0 }
   *
   * @return Linf cone
   */
  constexpr auto epigraph_impl() const {
    using LiftedSpace = linalg::Matrix<T, 1, N + 1>;
    return NormCone<LiftedSpace, NormConeType::LinfCC, ViewTag::ConvexCone>();
  }
};

}  // namespace convex
