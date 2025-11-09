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

#include "convex_fn.hpp"

namespace convex {

/**
 * @brief default coordinate map for rsoc
 *
 * coords (x, t)
 *
 * [ 1 0 ] (x, t) -> (x, t)
 * [ 0 1 ]
 *
 * @return affine map -> rsoc coords
 */
template <linalg::SquareMatrixLike CoordMat>
constexpr auto square_to_rsoc_map(const linalg::mat_type_t<CoordMat>& t) {
  using T = linalg::mat_type_t<CoordMat>;
  static constexpr size_t N = linalg::mat_rows_v<CoordMat>;

  auto I = linalg::make_identity<T, N>();
  I[N - 1][N - 1] = t;
  return affine::AffineMap<CoordMat>(I, linalg::Vector<T, N>{});
}

template <linalg::RnSpace Domain, FnTag Tag = FnTag::Convex>
  requires(linalg::space_dim_v<Domain> == 1)
class SquareFn
    : public std::conditional_t<Tag == FnTag::Convex,
                                ConvexFn<Domain, SquareFn<Domain, Tag>>,
                                ConcaveFn<Domain, SquareFn<Domain, Tag>>> {
 public:
  /// @brief bring into scope convex fn alias / helpers
  using ConvexBase =
      std::conditional_t<Tag == FnTag::Convex,
                         ConvexFn<Domain, SquareFn<Domain, Tag>>,
                         ConcaveFn<Domain, SquareFn<Domain, Tag>>>;
  using ConvexBase::Dim;
  using ConvexBase::N;
  using typename ConvexBase::CoordMat;
  using typename ConvexBase::Point;
  using typename ConvexBase::T;

  /**
   * @brief eval impl
   *
   * @param x \in R
   *
   * @return x^2 \in R
   */
  constexpr std::optional<T> eval_impl(const Point& x) const {
    if constexpr (Tag == FnTag::Convex)
      return x(0) * x(0);
    else
      return T(-1) * x(0) * x(0);
  }

  /**
   * @brief epigraph impl
   *
   * C = { (x, t) | x * x <= t }
   *
   * @return cone
   */
  constexpr auto epigraph_impl() const
    requires(Tag == FnTag::Convex)
  {
    return RotatedSOC<Domain, NormConeType::L2CC, ViewTag::LiftedCone>(
        square_to_rsoc_map<CoordMat>(T(1)));
  }

  /**
   * @brief hypograph impl
   *
   * C = { (x, t) | -(x * x) >= t } \equiv
   * C = { (x, t) | x * x <= -t }
   *
   * @return cone
   */
  constexpr auto hypograph_impl() const
    requires(Tag == FnTag::Concave)
  {
    return RotatedSOC<Domain, NormConeType::L2CC, ViewTag::LiftedCone>(
        square_to_rsoc_map<CoordMat>(T(-1)));
  }

  /**
   * @brief negate impl
   *
   * @return negated f
   */
  constexpr auto negate_impl() const {
    if constexpr (Tag == FnTag::Concave)
      return SquareFn<Domain, FnTag::Convex>();
    else
      return SquareFn<Domain, FnTag::Concave>();
  }
};

}  // namespace convex
