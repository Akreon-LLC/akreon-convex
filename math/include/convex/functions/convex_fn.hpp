/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "../cone_view.hpp"
#include "../convex_traits.hpp"

namespace convex {
/**
 * @brief base convex function CRTP
 *
 * Similar space constraints on R^n with convex set domain
 */
template <ConvexDomainLike Domain, typename Derived>
class ConvexFn {
 public:
  /// @brief alias to extract type traits from Domain convex set
  using Space = std::conditional_t<linalg::VectorSpace<Domain>, Domain,
                                   set_space_t<Domain>>;
  using T =
      std::conditional_t<linalg::VectorSpace<Domain>,
                         linalg::space_type_t<Domain>, set_type_t<Domain>>;
  using Point = std::conditional_t<linalg::VectorSpace<Domain>, Domain,
                                   set_point_t<Domain>>;

  /// @brief helper to get dimension size of space & size of point
  static constexpr size_t Dim = set_dim_v<Domain>;
  static constexpr size_t N = linalg::vec_size_v<Point>;

  /**
   * @brief eval
   *
   * F(x) -> y \in R
   *
   * @param point x \in domain
   *
   * @return y (scalar in R)
   */
  constexpr T eval(const Point& x) const { return self().eval_impl(x); }

  /**
   * @brief epigraph of f = above the graph of f (convex set)
   *
   * f : R^n -> R
   * graph(f) = { (x, f(x)) | x \in dom f } \in R^n
   * epi(F) = { (x, t) | x \in dom f, f(x) <= t } \in R^{n+1}
   *
   * @return epigraph of func
   */
  constexpr auto epigraph() const { return self().epigraph_impl(); }

  /**
   * @brief self
   *
   * @return derived class instance
   */
  constexpr auto self() const { return static_cast<const Derived&>(*this); }
};
}  // namespace convex
