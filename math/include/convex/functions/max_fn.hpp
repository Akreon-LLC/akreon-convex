/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "convex/sets/norm_cone.hpp"
#include "convex_fn.hpp"

namespace convex {
/**
 * @brief max convex function CRTP
 *
 * F(x) = max { x_1, x_2, ... x_n }, x \in R^n
 */
template <linalg::RnSpace Domain, FnTag Tag = FnTag::Convex>
class MaxFn : public std::conditional_t<Tag == FnTag::Convex,
                                        ConvexFn<Domain, MaxFn<Domain, Tag>>,
                                        ConcaveFn<Domain, MaxFn<Domain, Tag>>> {
 public:
  /// @brief bring into scope convex fn alias / helpers
  using ConvexBase = std::conditional_t<Tag == FnTag::Convex,
                                        ConvexFn<Domain, MaxFn<Domain, Tag>>,
                                        ConcaveFn<Domain, MaxFn<Domain, Tag>>>;
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
    if constexpr (Tag == FnTag::Convex)
      return mx;
    else
      return T(-1) * mx;
  }

  /**
   * @brief epigraph impl
   *
   * F(x) = max { x_1, x_2, ... x_n }, x \in R^n
   * graph(F(x)) = { (x, max(x)) | x \in dom f }
   * epi(F(x)) = { (x, t) | max(x) <= t }
   *
   * @return Linf cone
   */
  constexpr auto epigraph_impl() const
    requires(Tag == FnTag::Convex)
  {
    return NormCone<Domain, NormConeType::LinfCC>().cone_view();
  }

  /**
   * @brief hypograph impl
   *
   * C = { (x, t) | max(x) <= -t }
   *
   * @return C
   */
  constexpr auto hypograph_impl() const
    requires(Tag == FnTag::Concave)
  {
    return NormCone<Domain, NormConeType::LinfCC, ViewTag::HypographCone>()
        .cone_view();
  }

  /**
   * @brief negate impl
   *
   * if concave -> convex
   * else convex -> concave
   *
   * @return negated f
   */
  constexpr auto negate_impl() const {
    if constexpr (Tag == FnTag::Concave)
      return MaxFn<Domain, FnTag::Convex>();
    else
      return MaxFn<Domain, FnTag::Concave>();
  }
};

}  // namespace convex
