/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "../sets/norm_cone.hpp"
#include "convex_fn.hpp"

namespace convex {
/**
 * @brief norm convex function wrapper
 */
template <linalg::RnSpace Space, linalg::NormLike Norm>
  requires(linalg::space_dim_v<Space> == linalg::norm_size_v<Norm>)
class NormFn : public ConvexFn<Space, NormFn<Space, Norm>> {
 public:
  /// @brief bring into scope the convex fn alias / helpers
  using ConvexBase = ConvexFn<Space, NormFn<Space, Norm>>;
  using ConvexBase::Dim;
  using ConvexBase::N;
  using typename ConvexBase::Point;
  using typename ConvexBase::T;

  /**
   * @brief eval impl
   *
   * @param point x \in domain
   *
   * @return y (scalar \in R)
   */
  constexpr std::optional<T> eval_impl(const Point& x) const {
    return Norm()(x);
  }

  /**
   * @brief epigraph impl
   *
   * F(x) = ||x||
   * graph(F(x)) = { (x, ||x||) | x \in dom f }
   * epi(F(x)) = epi(||x||) = { (x, t) | ||x|| <= t, x \in dom f, t >= 0 }
   * \equiv norm cone
   *
   * @return epigraph = norm cone
   */
  constexpr auto epigraph_impl() const {
    using LiftedSpace = linalg::Matrix<T, 1, N + 1>;
    if constexpr (std::same_as<Norm, linalg::L2<Point>>) {
      return NormCone<LiftedSpace, NormConeType::L2CC, ViewTag::ConvexCone>();
    } else if constexpr (std::same_as<Norm, linalg::L1<Point>>) {
      return NormCone<LiftedSpace, NormConeType::L1CC, ViewTag::ConvexCone>();
    } else {
      return NormCone<LiftedSpace, NormConeType::LinfCC, ViewTag::ConvexCone>();
    }
  }
};
}  // namespace convex
