/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "../affine_map.hpp"
#include "convex_fn.hpp"

namespace convex {
/**
 * @brief affine function (map wrapped in convex fn context)
 */
template <linalg::RnSpace Domain, FnTag Tag = FnTag::Convex>
class AffineFn : public ConvexFn<Domain, AffineFn<Domain, Tag>> {
 public:
  /// @brief bring into scope the convex fn alias / helpers
  using ConvexBase = ConvexFn<Domain, AffineFn<Domain, Tag>>;
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
  constexpr explicit AffineFn(const AffineMap<Mat>& map_in) : map(map_in) {}

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
   * F(x) = U x + b
   * graph(F(x)) = { (x, U x + b) | x \in dom f }
   * epi(F(x)) = epi(U x + b) = { (x, t) | U x + b <= t, x \in dom f } =
   * halfspace
   *
   * @return epigraph
   */
  constexpr auto epigraph_impl() const
    requires(Tag == FnTag::Convex)
  {
    const auto A = map.linear_part();
    const auto b = map.translation();
    return Halfspace<Domain, HalfspaceType::CLH, ViewTag::ConvexSet>(
               linalg::as_vector(A), b(0))
        .cone_view();
  }

 private:
  /** @brief member vars */
  AffineMap<Mat> map;
};
}  // namespace convex
