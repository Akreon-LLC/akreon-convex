/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "convex_fn.hpp"

namespace convex {

/**
 * @brief log sum exp = smooth max function
 *
 * lse(x) = log(\sum_i^n(e^{x_i})) which is a smooth mapping from R^n to R
 */
template <linalg::RnSpace Domain>
class LogSumExp : public ConvexFn<Domain, LogSumExp<Domain>> {
 public:
  /// @brief bring into scope the convex fn alias / helpers
  using ConvexBase = ConvexFn<Domain, ConvexFn<Domain>>;
  using ConvexBase::Dim;
  using ConvexBase::N;
  using typename ConvexBase::Point;
  using typename ConvexBase::T;

  /// @brief alias for affine map (codomain 1-dim)
  using Mat = linalg::Matrix<T, 1, N>;

  /**
   * @brief eval impl
   *
   * @param point x \in dom f
   *
   * @return y \in R
   */
  constexpr std::optional<T> eval_impl(const Point& x) const { return T(0); }

  /**
   * @brief epigraph impl
   *
   * @return epigraph
   */
  constexpr auto epigraph_impl() const { return false; }
};

}  // namespace convex
