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

namespace convex {

/**
 * @brief default coordinate map for norm cone (with input args)
 *
 * norm cone coords: (x, c, r, t) -> || x - c || <= r * t
 *
 * [ I 0 ] [ x; t ] - [ c; 0 ] = [ x - c; r t]
 * [ 0 r ]
 *
 * @return affine map -> norm cone coords
 */
template <linalg::SquareMatrixLike CoordMat, linalg::VectorLike Vec>
  requires(linalg::mat_rows_v<CoordMat> == linalg::vec_size_v<Vec> + 1)
constexpr auto default_cone_map(const Vec& c,
                                const linalg::mat_type_t<CoordMat>& r) {
  using T = linalg::mat_type_t<CoordMat>;
  static constexpr size_t N = linalg::mat_rows_v<CoordMat>;
  auto I = linalg::make_identity<T, N>();
  I[N - 1][N - 1] *= r;
  linalg::Vector<T, N> b;
  for (size_t i = 0; i < N - 1; ++i) b[i] = T(-1) * c(i);
  return affine::AffineMap<CoordMat>(I, b);
}

/**
 * @brief Norm cone definition valid on a vector space dim(R^n + 1)
 *
 * { (x, t) | ||x|| <= t }... closed norm cone defined on L1, L2, Linf
 * { (x, t) | ||x|| < t }... open cone defined on L1, L2, Linf
 */
template <linalg::RnSpace Space, NormConeType Type,
          ViewTag Tag = ViewTag::LiftedCone>
  requires(linalg::space_dim_v<Space> ==
           linalg::vec_size_v<linalg::space_point_t<Space>>)
class NormCone : public ConvexSet<Space, NormCone<Space, Type, Tag>> {
 public:
  /// @brief bring into scope the type traits from the base class
  using ConvexBase = ConvexSet<Space, NormCone<Space, Type, Tag>>;
  using typename ConvexBase::CoordMat;
  using typename ConvexBase::Point;
  using typename ConvexBase::T;

  /// @brief bring into scope the const data members
  using ConvexBase::Dim;

  /// @brief helper to get n elements in point
  static constexpr size_t N = linalg::vec_size_v<Point>;

  /**
   * @brief CTOR for norm cone
   */
  constexpr explicit NormCone()
      : ConvexBase(default_cone_map<CoordMat>(Point{}, T(1))) {}

  /**
   * @brief CTOR for norm cone
   *
   * @param coord map in
   */
  constexpr explicit NormCone(const affine::AffineMap<CoordMat>& map_in)
      : ConvexBase(map_in) {}

  /**
   * @brief CTOR for norm cone
   *
   * Transformed from norm ball or rotated SOC
   *
   * @param center point from norm ball
   * @param r (radius from norm ball)
   */
  constexpr explicit NormCone(const Point& c_in, const T& r_in)
      : ConvexBase(default_cone_map<CoordMat>(c_in, r_in)) {}

  /**
   * @brief contains function to see if x is in the norm cone (y, t) \in K
   *
   * K = {(x, t) \in R^n x R | ||x|| <= 1}
   *
   * @param point (x,t) - lifted by default
   * @param tolerance
   *
   * @return T/F
   */
  constexpr bool contains_impl(const Point& x, const T& tol) const {
    return contains_impl(std::make_tuple(x, T(1)), tol);
  }

  /**
   * @brief contains function to see if x is in the norm cone (y, t) \in K
   *
   * Standard Cone: K = {(x, t) \in R^n x R_+ | ||x|| <= t}
   *
   * @param point (x,t) - lifted by default
   * @param tolerance
   *
   * @return T/F
   */
  constexpr bool contains_impl(const std::tuple<Point, T>& x_in,
                               const T& /* tol */) const {
    const auto x_proj = this->project(x_in);
    const auto x = std::get<0>(x_proj);
    const auto s = std::get<1>(x_proj);

    const bool zeros = linalg::all_zero(x) && s == T(0);

    switch (Type) {
      case NormConeType::L2OC:
        return linalg::L2<Point>()(x) < s;
      case NormConeType::L2CC:
        return zeros || linalg::L2<Point>()(x) <= s;
      case NormConeType::L1OC:
        return linalg::L1<Point>()(x) < s;
      case NormConeType::L1CC:
        return zeros || linalg::L1<Point>()(x) <= s;
      case NormConeType::LinfOC:
        return linalg::Linf<Point>()(x) < s;
      case NormConeType::LinfCC:
      default:
        return zeros || linalg::Linf<Point>()(x) <= s;
    }
  }

  /**
   * @brief is_well_formed checks to see if norm cone is valid
   *
   * @return T/F
   */
  constexpr bool is_well_formed_impl() const { return true; }

  /**
   * @brief equality comparison
   *
   * @param convex set
   *
   * @return T/F
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
          NormCone<Space, Type, ViewTag::LiftedCone>(this->coord_map));
    else
      return ConeView(NormCone<Space, Type, Tag>(this->coord_map));
  }
};

}  // namespace convex
