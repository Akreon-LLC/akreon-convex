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
 * @brief Rotated second order cone
 *
 * { (x, y, z) | ||x||_2^2 <= y z, y > 0, z >= 0 }
 */
template <linalg::RnSpace Space, NormConeType Type,
          ViewTag Tag = ViewTag::ConvexSet>
  requires(linalg::space_dim_v<Space> ==
               linalg::vec_size_v<linalg::space_point_t<Space>> &&
           linalg::space_dim_v<Space> > 1 &&
           (Type == NormConeType::L2OC || Type == NormConeType::L2CC))
class RotatedSOC : public ConvexSet<Space, RotatedSOC<Space, Type, Tag>> {
 public:
  /// @brief bring into scope the type traits from the base class
  using ConvexBase = ConvexSet<Space, RotatedSOC<Space, Type, Tag>>;
  using typename ConvexBase::Point;
  using typename ConvexBase::T;

  /// @brief bring into scope the const data members
  using ConvexBase::Dim;

  /// @brief helper to get n elements in point
  static constexpr size_t N = linalg::vec_size_v<Point>;

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
   * @param x \in R^{n-2}, y \in R_++, z \in R_+
   *
   * @return T/F
   */
  constexpr bool contains_impl(const std::tuple<Point, T>& x_in, T tol) const {
    const auto x_tmp = std::get<0>(x_in);
    T y = x_tmp(N - 1);
    if (y <= T(0)) return false;
    T z = std::get<1>(x_in);
    if (z < T(0)) return false;
    const auto x = std::get<0>(linalg::splice<N - 1>(x_tmp));

    const auto norm_x = linalg::L2<decltype(x)>()(x);
    const bool zeros = linalg::all_zero(x) && z == T(0);

    y = y == T(0) ? tol : y;

    if constexpr (Tag == ViewTag::HypographCone) {
      z *= T(-1);
      tol *= T(-1);
    }

    switch (Type) {
      case NormConeType::L2OC:
        return norm_x * norm_x < y * z;
      case NormConeType::L2CC:
      default:
        return zeros || norm_x * norm_x <= y * z + tol;
    }
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
      return ConeView(RotatedSOC<Space, Type, ViewTag::LiftedCone>());
    else
      return ConeView(RotatedSOC<Space, Type, Tag>());
  }

  /**
   * @brief project impl
   *
   * @param project map
   * @param in set
   *
   * @return projected set
   */
  template <linalg::MatrixLike MatA, linalg::MatrixLike MatC>
  constexpr std::optional<linalg::Vector<T, linalg::mat_cols_v<MatA> + 1>>
  projective_map_impl(const ProjectiveMap<MatA, MatC>& /* map */,
                      const Point& /* x */) const {
    return std::nullopt;
  }
};
}  // namespace convex
