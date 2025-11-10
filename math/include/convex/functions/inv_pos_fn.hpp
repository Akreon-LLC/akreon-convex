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
 * @brief coord map from 1/x to rsoc
 *
 * coords (x, t) -> (x=1, y=x, z=+-t)
 * x2 = x, x1 = dont care
 * [0 0   0] [x1; x2; t] + [1; 0; 0]
 * [0 1   0]
 * [0 0 +-1]
 *
 * @return coord map
 */
template <linalg::SquareMatrixLike CoordMat>
  requires(linalg::mat_rows_v<CoordMat> == 3)
constexpr auto inv_pos_to_rsoc(const linalg::mat_type_t<CoordMat>& t) {
  using T = linalg::mat_type_t<CoordMat>;
  static constexpr size_t N = linalg::mat_rows_v<CoordMat>;
  CoordMat U;
  U[1][1] = T(1);
  U[2][2] = t;
  linalg::Vector<T, N> b;
  b[0] = T(1);
  return affine::AffineMap<CoordMat>(U, b);
}

/**
 * @brief log function
 *
 * f(x) = log(x), x \in R_++
 */
template <linalg::RnSpace Domain, FnTag Tag = FnTag::Convex>
  requires(linalg::space_dim_v<Domain> == 1)
class InvPosFn
    : public std::conditional_t<Tag == FnTag::Convex,
                                ConvexFn<Domain, InvPosFn<Domain, Tag>>,
                                ConcaveFn<Domain, InvPosFn<Domain, Tag>>> {
 public:
  /// @brief bring into scope convex fn alias / helpers
  using ConvexBase =
      std::conditional_t<Tag == FnTag::Convex,
                         ConvexFn<Domain, InvPosFn<Domain, Tag>>,
                         ConcaveFn<Domain, InvPosFn<Domain, Tag>>>;
  using ConvexBase::Dim;
  using ConvexBase::N;
  using typename ConvexBase::Point;
  using typename ConvexBase::T;

  /// @brief alias extended coord mat
  using CoordMatExt = linalg::Matrix<T, N + 2, N + 2>;

  /**
   * @brief eval impl
   *
   * @param x \in R_++
   *
   * @return res \in R_++ or R_-- (depending on convex or concave)
   */
  constexpr std::optional<T> eval_impl(const Point& x) const {
    if (x(0) <= T(0)) return std::nullopt;
    if constexpr (Tag == FnTag::Convex) {
      return T(1) / x(0);
    } else {
      return T(-1) / x(0);
    }
  }

  /**
   * @brief epigraph impl
   *
   * C = { (x, t) | 1 / x <= t } \equiv
   * C = { (x, t) | 1 <= x t }
   *
   * @return rsoc
   */
  constexpr auto epigraph_impl() const
    requires(Tag == FnTag::Convex)
  {
    using LiftedSpace = linalg::Matrix<T, 1, N + 1>;
    return RotatedSOC<LiftedSpace, NormConeType::L2CC, ViewTag::LiftedCone>(
        inv_pos_to_rsoc<CoordMatExt>(T(1)));
  }

  /**
   * @brief hypograph impl
   *
   * C = { (x, t) | 1 / x >= t } \equiv
   * C = { (x, t) | 1 / x <= -t } \equiv
   * C = { (x, t) | 1 <= -x t }
   *
   * @return rsoc
   */
  constexpr auto hypograph_impl() const
    requires(Tag == FnTag::Concave)
  {
    using LiftedSpace = linalg::Matrix<T, 1, N + 1>;
    return RotatedSOC<LiftedSpace, NormConeType::L2CC, ViewTag::LiftedCone>(
        inv_pos_to_rsoc<CoordMatExt>(T(-1)));
  }

  /**
   * @brief negate impl
   *
   * @return negated f
   */
  constexpr auto negate_impl() const {
    if constexpr (Tag == FnTag::Convex)
      return InvPosFn<Domain, FnTag::Concave>();
    else
      return InvPosFn<Domain, FnTag::Convex>();
  }
};

}  // namespace convex
