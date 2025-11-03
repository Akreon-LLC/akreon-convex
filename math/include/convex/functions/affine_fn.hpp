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

#include "affine/affine_map.hpp"
#include "concave_fn.hpp"
#include "convex_fn.hpp"

namespace convex {
/**
 * @brief affine function (map wrapped in convex fn context)
 */
template <linalg::RnSpace Domain, FnTag Tag = FnTag::Convex>
class AffineFn
    : public std::conditional_t<Tag == FnTag::Convex,
                                ConvexFn<Domain, AffineFn<Domain, Tag>>,
                                ConcaveFn<Domain, AffineFn<Domain, Tag>>> {
 public:
  /// @brief bring into scope the convex fn alias / helpers
  using ConvexBase =
      std::conditional_t<Tag == FnTag::Convex,
                         ConvexFn<Domain, AffineFn<Domain, Tag>>,
                         ConcaveFn<Domain, AffineFn<Domain, Tag>>>;
  using ConvexBase::Dim;
  using ConvexBase::N;
  using typename ConvexBase::Point;
  using typename ConvexBase::T;

  /// @brief alias for affine map (codomain 1-dim)
  using Mat = linalg::Matrix<T, 1, N>;

  /**
   * @brief CTOR for affine func
   *
   * @param AffineMap object
   */
  constexpr explicit AffineFn(const affine::AffineMap<Mat>& map_in)
      : map(map_in) {}

  /**
   * @brief eval impl
   *
   * @param point x \in domain
   *
   * @return y \in R
   */
  constexpr std::optional<T> eval_impl(const Point& x) const {
    const linalg::Vector<T, 1> res = map.linear_part() * x + map.translation();
    return res(0);
  }

  /**
   * @brief epigraph impl
   *
   * F(x) = a^T x + b
   * graph(F(x)) = { (x, a^T x + b) | x \in dom f }
   * epi(F(x)) = epi(a^T x + b) = { (x, t) | a^T x + b <= t, x \in dom f } =
   * halfspace
   *
   * @return epigraph
   */
  constexpr auto epigraph_impl() const {
    const auto A = map.linear_part();
    const auto b = map.translation();
    return Halfspace<Domain, HalfspaceType::CLH, ViewTag::EpigraphCone>(
               linalg::as_vector(A), b(0))
        .cone_view();
  }

  /**
   * @brief hypograph impl
   *
   * opposite of epigraph for concave function
   * { (x, t) | -a^T x - b <= -t }, negation handled in cone view
   *
   * @return hypograph
   */
  constexpr auto hypograph_impl() const {
    const auto A = map.linear_part();
    const auto b = map.translation();
    return Halfspace<Domain, HalfspaceType::CUH, ViewTag::HypographCone>(
               linalg::as_vector(A), b(0))
        .cone_view();
  }

  /**
   * @brief negate impl
   *
   * @return
   */
  constexpr auto negate_impl() const {
    if constexpr (Tag == FnTag::Concave)
      return AffineFn<Domain, FnTag::Convex>(map);
    else
      return AffineFn<Domain, FnTag::Concave>(map);
  }

 private:
  /** @brief member vars */
  affine::AffineMap<Mat> map;
};
}  // namespace convex
