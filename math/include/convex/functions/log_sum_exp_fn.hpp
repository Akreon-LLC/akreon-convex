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
class LogSumExpFn : public ConvexFn<Domain, LogSumExpFn<Domain>> {
 public:
  /// @brief bring into scope the convex fn alias / helpers
  using ConvexBase = ConvexFn<Domain, LogSumExpFn<Domain>>;
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
  constexpr std::optional<T> eval_impl(const Point& x) const {
    T sum(0);
    std::for_each(x.begin(), x.end(),
                  [&](const auto& pt) { sum += linalg::constexpr_exp(pt); });
    return linalg::constexpr_ln(sum);
  }

  /**
   * @brief epigraph impl
   *
   * F(x) = log(\sum_i^n exp(x_i))
   * graph(F(x)) = { (x, log(\sum_i^n exp(x_i)) | x \in dom f }
   * epi(F(x)) = { (x, t) | log(\sum_i^n exp(x_i)) <= t, x \in dom f } \equiv
   * { (x, t) | \sum_i^n exp(x_i) <= exp(t), x \in dom f } \equiv
   * { (x, t) | \sum_i^n exp(x_i - t) <= 1, x \in dom f}
   *
   * Exp cone form : { (x, y, z) | y e^{x/y} <= z, y > 0 } \cup { (x, 0, z) | x
   * <= 0, z >= 0 }
   *
   * So we have { (x_i - t, 1, s_i) | e^{x_i - t} <= s_i, \sum_i s_i <= 1 }
   *
   * @return epigraph
   */
  constexpr auto epigraph_impl() const {
    // (dakre) TODO - we have a cartesian product of K_i intersected with a
    // simplex defined by s_i
    return false;
  }
};

}  // namespace convex
