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

#include <iostream>

#include "convex_set.hpp"

namespace convex {

/**
 * @brief default coordinate map for rotated soc
 *
 * C = { (x, y, z) \in R^{N-1} x R_++ x R | ||x||_2^2 <= y z }
 *
 * apply non-linear y z mul later, null out y in x projection
 * [ I 0 ] [ x; z ]
 * [ 0 0 ]
 * [ 0 1 ]
 *
 * rsoc coord map: (x, y z)
 *
 * @return affine map -> rsoc coords
 */
template <linalg::SquareMatrixLike CoordMat>
constexpr auto default_rsoc_map(const linalg::mat_type_t<CoordMat>& t) {
  using T = linalg::mat_type_t<CoordMat>;
  static constexpr size_t N = linalg::mat_rows_v<CoordMat>;

  CoordMat U;
  const auto I = linalg::make_identity<T, N - 2>();
  U = linalg::set_block<0, N - 2, 0, N - 2>(U, I);
  U[N - 1][N - 1] = t;
  return affine::AffineMap<CoordMat>(U, linalg::Vector<T, N>{});
}

/**
 * @brief Rotated second order cone
 *
 * { (x, y, z) | ||x||_2^2 <= y z, y > 0, z >= 0 }
 */
template <linalg::RnSpace Space, NormConeType Type,
          ViewTag Tag = ViewTag::ConvexSet>
  requires(linalg::space_dim_v<Space> ==
               linalg::vec_size_v<linalg::space_point_t<Space>> &&
           linalg::space_dim_v<Space> >= 1 &&
           (Type == NormConeType::L2OC || Type == NormConeType::L2CC))
class RotatedSOC : public ConvexSet<Space, RotatedSOC<Space, Type, Tag>> {
 public:
  /// @brief bring into scope the type traits from the base class
  using ConvexBase = ConvexSet<Space, RotatedSOC<Space, Type, Tag>>;
  using typename ConvexBase::CoordMat;
  using typename ConvexBase::Point;
  using typename ConvexBase::T;

  /// @brief bring into scope the const data members
  using ConvexBase::Dim;

  /// @brief helper to get n elements in point
  static constexpr size_t N = linalg::vec_size_v<Point>;

  /**
   * @brief CTOR for rsoc
   */
  constexpr explicit RotatedSOC()
      : ConvexBase(default_rsoc_map<CoordMat>(T(1))) {}

  /**
   * @brief CTOR for rsoc
   *
   * @param affine map input
   */
  constexpr explicit RotatedSOC(const affine::AffineMap<CoordMat>& map_in)
      : ConvexBase(map_in) {}

  /**
   * @brief contains to see if x is in rotated soc
   *
   * @param point x_in = (x \in R^{n-2}, y \in R_++, z \in R_+)
   *
   * @return T/F
   */
  constexpr bool contains_impl(const Point& x_in, const T& tol) const {
    return contains_impl(std::tuple(x_in, T(1)), tol);
  }

  /**
   * @brief contains to see if x is in rotated soc
   *
   * Standard cone
   * C = { (x, y, z) \in R^n x R_++ x R | ||x||_2^2 <= y z }
   *
   * @param x \in R^{n-2}, y \in R_++
   *
   * @return T/F
   */
  constexpr bool contains_impl(const std::tuple<Point, T>& x_in,
                               const T& tol) const {
    T y(1);
    if constexpr (N > 1) {
      y = std::get<0>(x_in)(N - 1);
      if (y <= T(0)) return false;
    }

    const auto x_proj = this->project(x_in);
    const auto x = std::get<0>(x_proj);

    // (dakre) apply non-linear mul here
    const auto s = y * std::get<1>(x_proj);

    const auto norm_x = linalg::L2<decltype(x)>()(x);
    const auto zeros = linalg::all_zero(x) && s == T(0);

    switch (Type) {
      case NormConeType::L2OC:
        return norm_x * norm_x < s;
      case NormConeType::L2CC:
      default:
        return zeros || norm_x * norm_x <= s + tol;
    }

    return false;
  }

  /**
   * @brief is well formed checks to see if rot soc is valid
   *
   * @return T/F
   */
  constexpr bool is_well_formed_impl() const { return true; }

  /**
   * @brief is equal impl
   */
  constexpr bool is_equal_impl(const ConvexBase& other) const {
    return other.coord_map == this->coord_map;
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
          RotatedSOC<Space, Type, ViewTag::LiftedCone>(this->coord_map));
    else
      return ConeView(RotatedSOC<Space, Type, Tag>(this->coord_map));
  }
};

}  // namespace convex
