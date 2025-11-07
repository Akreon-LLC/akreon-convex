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

#include "convex_set.hpp"
#include "exp_cone.hpp"

namespace convex {

/**
 * @brief default coordinate map for log sum exp cone
 *
 * lsec coords: (x, t) -> (x +/- t, 1, s_i)
 *
 * [ I b ] [ x; t ]
 * [ 0 1 ]
 *
 * b = +/- 1 to encode first coord relationship
 *
 * @return affine map -> log sum exp coords
 */
template <linalg::SquareMatrixLike CoordMat, linalg::VectorLike Vec>
  requires(linalg::mat_rows_v<CoordMat> == linalg::vec_size_v<Vec> + 1)
constexpr auto default_lsec_map(const Vec& b) {
  using T = linalg::mat_type_t<CoordMat>;
  static constexpr size_t N = linalg::mat_rows_v<CoordMat>;
  auto U = linalg::make_identity<T, N>();
  U = set_block<0, N - 1, N - 1, N>(U, linalg::as_col_vector(b));
  return affine::AffineMap<CoordMat>(U, linalg::Vector<T, N>{});
}

/**
 * @brief Log Sum Exp cone
 *
 * C = { (x, t) | \sum_i e^{x_i - t} <= 1 }
 * C = k_1 x k_2 ... is the epigraph for log sum exp and is a cartesian product
 * of multiple exp cones
 */
template <linalg::RnSpace Space, ViewTag Tag = ViewTag::LiftedCone>
  requires(linalg::space_dim_v<Space> ==
           linalg::vec_size_v<linalg::space_point_t<Space>>)
class LogSumExpCone : public ConvexSet<Space, LogSumExpCone<Space, Tag>> {
 public:
  /// @brief bring into scope the type traits from the base class
  using ConvexBase = ConvexSet<Space, LogSumExpCone<Space, Tag>>;
  using typename ConvexBase::CoordMat;
  using typename ConvexBase::Point;
  using typename ConvexBase::T;

  /// @brief bring into scope the const data members
  using ConvexBase::Dim;
  using ConvexBase::N;

  /**
   * @brief CTOR
   */
  constexpr explicit LogSumExpCone()
      : ConvexBase(default_lsec_map<CoordMat>(
            linalg::vec_traits<Point>().fill(T(-1)))) {}

  /**
   * @brief CTOR
   *
   * @param affine map input
   */
  constexpr explicit LogSumExpCone(const affine::AffineMap<CoordMat>& map_in)
      : ConvexBase(map_in) {}

  /**
   * @brief is_well_formed impl
   *
   * @return T/F
   */
  constexpr bool is_well_formed_impl() const { return true; }

  /**
   * @brief check if point is in the set
   *
   * @param point in vector space R^n
   * @param scalar tol
   *
   * @return T/F
   */
  constexpr bool contains_impl(const Point& x, const T& /*tol*/) const {
    return contains_impl(std::make_tuple(x, T(1)));
  }

  /**
   * @brief contains with (x,t) in cone (standard as epigraph)
   *
   * Standard Form
   * C' = { (x,t) \in R^n x R | \sum_i e^{x_i - t} <= 1 }
   * {(x_i - t, 1, s_i) | e^{x_i - t} <= s_i, \sum_i s_i <= 1, \forall s_i >= 0}
   *
   * @param (x,t) \in R^n x R_++ or R^n x R depending on lift
   * @param scalar tol
   *
   * @return T/F
   */
  constexpr bool contains_impl(const std::tuple<Point, T>& x_in,
                               const T& /*tol*/) const {
    const auto x_proj = this->project(x_in);
    const auto x = std::get<0>(x_proj);

    T sum(0);
    std::for_each(x.begin(), x.end(), [&sum](const auto& pt) {
      sum += Exp<Space>()(pt, T(1)).value();
    });

    return sum <= T(1);
  }

  /**
   * @brief equality comparison
   *
   * @param convex set
   *
   * @return T/F
   */
  constexpr bool is_equal_impl(const ConvexBase& /* other */) const {
    return true;
  }

  /**
   * @brief transform impl
   *
   * @return self
   */
  template <linalg::MatrixLike Mat>
  constexpr auto transform_impl(const affine::AffineMap<Mat>& /* map */) const {
    return *this;
  }

  /**
   * @brief cone view impl
   *
   * @return cone
   */
  constexpr auto cone_view_impl() const {
    if constexpr (Tag == ViewTag::ConvexSet)
      return ConeView(
          LogSumExpCone<Space, ViewTag::LiftedCone>(this->coord_map));
    else
      return ConeView(LogSumExpCone<Space, Tag>(this->coord_map));
  }
};

}  // namespace convex
