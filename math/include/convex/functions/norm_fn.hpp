/*
 * Copyright © 2025 Akreon LLC. All rights reserved.
 *
 * Proprietary and Confidential.
 * This source code is the property of Akreon LLC and constitutes a trade secret
 * protected by U.S. and international law.
 *
 * Unauthorized copying, modification, distribution, or disclosure of this file,
 * via any medium, is strictly prohibited without express written permission
 * from Akreon LLC.
 */


#pragma once

#include "convex/sets/norm_cone.hpp"
#include "convex_fn.hpp"

namespace convex {
/**
 * @brief norm convex function wrapper
 */
template <linalg::RnSpace Domain, linalg::NormLike Norm,
          FnTag Tag = FnTag::Convex>
  requires(linalg::space_dim_v<Domain> == linalg::norm_size_v<Norm>)
class NormFn
    : public std::conditional_t<Tag == FnTag::Convex,
                                ConvexFn<Domain, NormFn<Domain, Norm, Tag>>,
                                ConcaveFn<Domain, NormFn<Domain, Norm, Tag>>> {
 public:
  /// @brief bring into scope the convex fn alias / helpers
  using ConvexBase =
      std::conditional_t<Tag == FnTag::Convex,
                         ConvexFn<Domain, NormFn<Domain, Norm, Tag>>,
                         ConcaveFn<Domain, NormFn<Domain, Norm, Tag>>>;
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
    if constexpr (Tag == FnTag::Convex)
      return Norm()(x);
    else
      return T(-1) * Norm()(x);
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
  constexpr auto epigraph_impl() const
    requires(Tag == FnTag::Convex)
  {
    if constexpr (std::same_as<Norm, linalg::L2<Point>>)
      return NormCone<Domain, NormConeType::L2CC>().cone_view();
    else if constexpr (std::same_as<Norm, linalg::L1<Point>>)
      return NormCone<Domain, NormConeType::L1CC>().cone_view();
    else
      return NormCone<Domain, NormConeType::LinfCC>().cone_view();
  }

  /**
   * @brief hypograph impl (for negated form)
   *
   * C = { (x,t) | - ||x|| >= t }
   *
   * @return hypograph
   */
  constexpr auto hypograph_impl() const
    requires(Tag == FnTag::Concave)
  {
    if constexpr (std::same_as<Norm, linalg::L2<Point>>)
      return NormCone<Domain, NormConeType::L2CC, ViewTag::HypographCone>()
          .cone_view();
    else if constexpr (std::same_as<Norm, linalg::L1<Point>>)
      return NormCone<Domain, NormConeType::L1CC, ViewTag::HypographCone>()
          .cone_view();
    else
      return NormCone<Domain, NormConeType::LinfCC, ViewTag::HypographCone>()
          .cone_view();
  }

  /**
   * @brief negate impl
   *
   * if concave tag -> convex
   * else convex tag -> concave
   *
   * return negated f
   */
  constexpr auto negate_impl() const {
    if constexpr (Tag == FnTag::Concave)
      return NormFn<Domain, Norm, FnTag::Convex>();
    else
      return NormFn<Domain, Norm, FnTag::Concave>();
  }
};
}  // namespace convex
