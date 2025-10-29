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
template <linalg::RnSpace Domain, FnTag Tag = FnTag::Convex>
class LogSumExpFn
    : public std::conditional_t<Tag == FnTag::Convex,
                                ConvexFn<Domain, LogSumExpFn<Domain, Tag>>,
                                ConcaveFn<Domain, LogSumExpFn<Domain, Tag>>> {
 public:
  /// @brief bring into scope the convex fn alias / helpers
  using ConvexBase =
      std::conditional_t<Tag == FnTag::Convex,
                         ConvexFn<Domain, LogSumExpFn<Domain, Tag>>,
                         ConcaveFn<Domain, LogSumExpFn<Domain, Tag>>>;
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

    const auto res = linalg::constexpr_ln(sum);
    if constexpr (Tag == FnTag::Convex)
      return res;
    else {
      if (!res.has_value()) return std::nullopt;
      return T(-1) * res.value();
    }
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
  constexpr auto epigraph_impl() const
    requires(Tag == FnTag::Convex)
  {
    return LogSumExpCone<Domain, ViewTag::EpigraphCone>().cone_view();
  }

  /**
   * @brief hypograph impl
   *
   * C = { (x_i + t, 1, s_i) | e^{x_i + 1} <= s_i, \sum_i s_i <= 1 }
   *
   * @return hypograph
   */
  constexpr auto hypograph_impl() const
    requires(Tag == FnTag::Concave)
  {
    return LogSumExpCone<Domain, ViewTag::HypographCone>().cone_view();
  }

  /**
   * @brief negate func
   *
   * @return negate f
   */
  constexpr auto negate_impl() const {
    if constexpr (Tag == FnTag::Convex)
      return LogSumExpFn<Domain, FnTag::Concave>();
    else
      return LogSumExpFn<Domain, FnTag::Convex>();
  }
};

}  // namespace convex
