/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "convex_fn.hpp"

namespace convex {

/**
 * @brief indicator func
 *
 * I(x) = { 0 | x \in c } else \infinity
 */
template <ConvexSetLike Set>
class IndicatorFn : public ConvexFn<Set, IndicatorFn<Set>> {
 public:
  /// @brief bring into scope convex fn alias / helpers
  using ConvexBase = ConvexFn<Set, IndicatorFn<Set>>;
  using ConvexBase::Dim;
  using ConvexBase::N;
  using typename ConvexBase::Point;
  using typename ConvexBase::T;

  /**
   * @brief CTOR
   *
   * convex set in ctor
   */
  constexpr explicit IndicatorFn(const Set& set_in) : set(set_in) {}

  /**
   * @brief eval impl
   *
   * @param Point \in R^n (may or may not be in C)
   *
   * @return (0 or inf)
   */
  constexpr std::optional<T> eval_impl(const Point& x) const {
    if (set.contains(x)) return T(0);
    return std::nullopt;
  }

  /**
   * @brief epigraph impl
   *
   * I(x) = { 0 | x \in C } else \infinity
   * graph(I(x)) = { (x, I(x)) | x \in dom f }
   * epi(I(x)) = { (x, t) | I(x) <= t , t >= 0 }
   *
   * @return cone of C
   */
  constexpr auto epigraph_impl() const { return set.cone_view(); }

 private:
  /** @brief member vars */
  Set set;
};
}  // namespace convex
