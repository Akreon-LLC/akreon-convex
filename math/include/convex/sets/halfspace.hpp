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
 * @brief default coordinate map for halfspace
 *
 * halfspace coords: (x, b) -> a^T x <= b
 *
 * @return affine map -> halfspace coords
 */
template <linalg::SquareMatrixLike CoordMat>
constexpr auto default_halfspace_map(const linalg::mat_type_t<CoordMat>& b) {
  using T = linalg::mat_type_t<CoordMat>;
  static constexpr size_t N = linalg::mat_rows_v<CoordMat>;
  auto I = linalg::make_identity<T, N>();
  I[N - 1][N - 1] = b;
  return affine::AffineMap<CoordMat>(I, linalg::Vector<T, N>{});
}

/**
 * @brief Halfspace defined on a valid vector space and one of the following
 * type traits:
 *
 * {x | a^T x <= b}... closed lower halfspace
 * {x | a^T x < b}... open lower halfspace
 * {x | a^T x >= b}... closed upper halfspace
 * {x | a^T x > b}... open upper halfspace
 */
template <linalg::RnSpace Space, HalfspaceType Type,
          ViewTag Tag = ViewTag::ConvexSet>
  requires(linalg::space_dim_v<Space> ==
           linalg::vec_size_v<linalg::space_point_t<Space>>)
class Halfspace : public ConvexSet<Space, Halfspace<Space, Type, Tag>> {
 public:
  /// @brief bring into scope the type traits from the base class
  using ConvexBase = ConvexSet<Space, Halfspace<Space, Type, Tag>>;
  using typename ConvexBase::Point;
  using typename ConvexBase::T;

  /// @brief bring into scope the const data members
  using ConvexBase::Dim;

  /// @brief helper to get n elements in point
  static constexpr size_t N = linalg::vec_size_v<Point>;

  /// @brief coord map matrix type
  using CoordMat = linalg::Matrix<T, N + 1, N + 1>;

  /**
   * @brief CTOR for halfspace set
   *
   * @param Point (i.e., n-dim vector of type T)
   * @param b bias scalar
   */
  constexpr explicit Halfspace(const Point& a_in, const T& b_in)
      : coord_map(default_halfspace_map<CoordMat>(b_in)), a(a_in), b(b_in) {}

  /**
   * @brief CTOR for halfspace cone
   *
   * @param Point
   * @param b bias scalar
   * @param coordinate map
   */
  constexpr explicit Halfspace(const Point& a_in, const T& b_in,
                               const affine::AffineMap<CoordMat>& map_in)
      : coord_map(map_in), a(a_in), b(b_in) {}

  /**
   * @brief contains function to see if x is in the halfspace a^T x <= b
   *
   * Standard set: { x | a^T x <= b }... { x | a^T x - b <= 0 }
   *
   * @param point x of type T with n elements
   * @param tolerance of type T (ignored)
   *
   * @return T/F
   */
  constexpr bool contains_impl(const Point& x, const T& tol) const {
    return contains_impl(std::tuple(x, T(1)), tol);
  }

  /**
   * @brief contains function to see if x is in the lifted halfspace
   *
   * @param (x, t) \in R^n x R_++ or \in R^n x R depending on lift
   * @param tolerance of type T (ignored)
   *
   * @return T/F
   */
  constexpr bool contains_impl(const std::tuple<Point, T>& x_in,
                               const T& /*tol*/) const {
    const auto x_proj = this->project(x_in);
    const auto x = std::get<0>(x_proj);
    const auto s = std::get<1>(x_proj);
    switch (Type) {
      case HalfspaceType::OUH:
        return linalg::dot(a, x) > s;
      case HalfspaceType::CUH:
        return linalg::dot(a, x) >= s;
      case HalfspaceType::OLH:
        return linalg::dot(a, x) < s;
      case HalfspaceType::CLH:
      default:
        return linalg::dot(a, x) <= s;
    }
  }

  /**
   * @brief is_well_formed checks to see if underlying set is a valid convex
   * set
   *
   * @return T/F
   */
  constexpr bool is_well_formed_impl() const {
    if (linalg::all_zero(a)) {
      if constexpr (Type == HalfspaceType::CLH || Type == HalfspaceType::OLH) {
        if (b < T(0)) return false;
      } else {
        if (b > T(0)) return false;
      }
    }
    return true;
  }

  /**
   * @brief transform halfspace by affine map
   *
   * H = { x | a^T x <= b }
   * f(H) = { y = f(x) | x \in H}... image of halfspace under f mapping
   * where affine map is: y = U x + t -> x = U^{-1} (y - t)
   * so, a^T U^{-1} (y - t) <= b \to a^T U^{-1} y <= b + a^T U^{-1} t
   * we arrive at: f(H) = { y | a^T U^{-1} y <= b + a^T U^{-1} t }
   *
   * @param affine map
   *
   * @return transformed halfspace
   */
  template <linalg::SquareMatrixLike Mat>
    requires(std::same_as<T, linalg::mat_type_t<Mat>> &&
             linalg::mat_cols_v<Mat> == N)
  constexpr auto transform_impl(const affine::AffineMap<Mat>& map) const {
    const auto aT_U =
        transpose(as_col_vector(a)) * linalg::inverse(map.linear_part());
    return Halfspace<Space, Type, Tag>(as_vector(aT_U),
                                       b + (aT_U * map.translation())(0));
  }

  /**
   * @brief is_equal implementation
   *
   * @return T/F
   */
  constexpr bool is_equal_impl(const ConvexBase& other) const {
    return other.self().a == a && other.self().b == b;
  }

  /**
   * @brief inverse perspective func from convex set to cone
   *
   * Standard inverse perspective function:
   * C' = { (x,t) \in R^n x R_++ | a^T x <= b t }
   *
   * @return cone
   */
  constexpr auto cone_view_impl() const {
    if constexpr (Tag == ViewTag::ConvexSet) {
      auto I = linalg::make_identity<T, N + 1>();
      I[N][N] *= b;
      const auto lifted_map =
          affine::AffineMap<CoordMat>(I, linalg::Vector<T, N + 1>{});
      return ConeView(
          Halfspace<Space, Type, ViewTag::LiftedCone>(a, b, lifted_map));
    } else {
      return ConeView(Haflspace<Space, Type, Tag>(a, b, coord_map));
    }
  }

 protected:
  /** @brief member vars in the halfspace */
  friend ConvexBase;
  affine::AffineMap<CoordMat> coord_map;  //< map (x, t) \in R^n x R -> cone

 private:
  /** @brief member variables in the halfspace */
  Point a;  //< n-dim vector of type T
  T b;      //< bias scalar of type T
};

}  // namespace convex
