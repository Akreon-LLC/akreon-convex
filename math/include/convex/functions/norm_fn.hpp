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
class NormFn : public ConvexFn<Space, NormFn<Space, Norm>> {
 public:
  /// @brief bring into scope the convex fn alias / helpers
  using ConvexBase = ConvexFn<Space, NormFn<Space, Norm>>;
  using ConvexBase::Dim;
  using ConvexBase::N;
  using typename ConvexBase::Point;
  using typename ConvexBase::Space;
  using typename ConvexBase::T;

  /**
   * @brief CTOR that wraps a norm
   *
   * @param norm functor (L2, L1, Linf)
   */
  constexpr explicit NormFn(const Norm& norm_in) : norm(norm_in) {}

  /**
   * @brief eval impl
   *
   * @param point x \in domain
   *
   * @return y (scalar \in R)
   */
  constexpr std::optional<T> eval_impl(const Point& x) const { return norm(x); }

  /**
   * @brief epigraph impl
   *
   * F(x) = ||x||
   * graph(F(x)) = { (x, ||x||) | x \in dom f }
   * epi(F(x)) = epi(||x||) = { (x, t) | ||x|| <= t, x \in dom f } \equiv norm
   * cone
   *
   * @return epigraph
   */
  constexpr auto epigraph_impl() const {
    using LiftedSpace = linalg::Matrix<T, 1, N + 1>;
    if constexpr (std::same_as<Norm, linalg::L2<T, N>>) {
      return NormCone<LiftedSpace, NormConeType::L2CC, ViewTag::ConvexCone>();
    } else if constexpr (std::same_as<Norm, linalg::L1<T, N>>) {
      return NormCone<LiftedSpace, NormConeType::L1CC, ViewTag::ConvexCone>();
    } else {
      return NormCone<LiftedSpace, NormConeType::LinfCC, ViewTag::ConvexCone>();
    }
  }

 private:
  /** @brief member vars */
  Norm norm;  //< L1, L2, Linf norm functor
};
}  // namespace convex
