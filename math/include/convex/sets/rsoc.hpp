/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include <iostream>

#include "convex_set.hpp"

namespace convex {
/**
 * @brief Rotated second order cone
 *
 * { (x, y, z) | ||x||_2^2 <= y z }
 */
template <linalg::RnSpace Space, NormConeType Type,
          ViewTag Tag = ViewTag::ConvexSet>
  requires(linalg::space_dim_v<Space> ==
               linalg::vec_size_v<linalg::space_point_t<Space>> &&
           linalg::space_dim_v<Space> > 2 &&
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

  /// @brief lifted point (based on lifted space or not)
  static constexpr size_t AugSize = N - 2;
  using AugPoint = linalg::Vector<T, AugSize>;

  /**
   * @brief contains to see if x is in rotated soc
   *
   * @param point x_in = (x \in R^{n-2}, y \in R_++, t \in R_+)
   *
   * @return T/F
   */
  constexpr bool contains_impl(const Point& x_in, const T& tol) const {
    T y = x_in(N - 2);
    if (y < T(0)) return false;
    T t = x_in(N - 1);
    if (t < T(0)) return false;
    AugPoint x = std::get<0>(linalg::splice<N - 2>(x_in));

    const auto norm_x = linalg::L2<AugPoint>()(x);
    const bool zeros = linalg::all_zero(x_in);

    y = y == T(0) ? tol : y;

    switch (Type) {
      case NormConeType::L2OC:
        return norm_x * norm_x < y * t;
      case NormConeType::L2CC:
      default:
        return zeros || norm_x * norm_x <= y * t + tol;
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
    requires(Tag == ViewTag::ConvexSet)
  constexpr auto transform_impl(const AffineMap<Mat>& /* map */) const {
    return *this;
  }

  /**
   * @brief cone view impl
   *
   * @return cone
   */
  constexpr auto cone_view_impl() const
    requires(Tag == ViewTag::ConvexSet)
  {
    return ConeView(RotatedSOC<Space, Type, ViewTag::ConvexCone>());
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
    requires(Tag == ViewTag::ConvexSet)
  constexpr std::optional<linalg::Vector<T, linalg::mat_cols_v<MatA> + 1>>
  project_impl(const ProjectiveMap<MatA, MatC>& /* map */,
               const Point& /* x */) const {
    return std::nullopt;
  }
};
}  // namespace convex
