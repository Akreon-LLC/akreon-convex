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
#include "linalg/symmetric_matrix.hpp"

namespace convex {

/**
 * @brief default coordinate map for ellipsoid
 *
 * E = { x | (x - c)^T P (x - c) <= 1, P \in S^n_+ }
 *
 * ellipsoid map coords: (x - c, 1)
 *
 * @return affine map -> halfspace coords
 */
template <linalg::SquareMatrixLike CoordMat>
constexpr auto default_ellipsoid_map(
    const linalg::Vector<linalg::mat_type_t<CoordMat>,
                         linalg::mat_rows_v<CoordMat> - 1>& c) {
  using T = linalg::mat_type_t<CoordMat>;
  static constexpr size_t N = linalg::mat_rows_v<CoordMat>;

  const auto I = linalg::make_identity<T, N>();

  linalg::Vector<T, N> c_tilde;
  for (size_t i = 0; i < N - 1; ++i) c_tilde[i] = T(-1) * c(i);

  return affine::AffineMap<CoordMat>(I, c_tilde);
}

/**
 * @brief Ellipsoid defined as the following
 *
 * E = { x | (x - c)^T P (x - c) <= 1 } s.t. P \in S^n_+
 */
template <linalg::RnSpace Space, EllipsoidType Type,
          ViewTag Tag = ViewTag::ConvexSet>
  requires(linalg::space_dim_v<Space> ==
           linalg::vec_size_v<linalg::space_point_t<Space>>)
class Ellipsoid : public ConvexSet<Space, Ellipsoid<Space, Type, Tag>> {
 public:
  /// @brief bring into scope the type traits from the base class
  using ConvexBase = ConvexSet<Space, Ellipsoid<Space, Type, Tag>>;
  using typename ConvexBase::CoordMat;
  using typename ConvexBase::Point;
  using typename ConvexBase::T;

  /// @brief bring into scope the const data members
  using ConvexBase::Dim;
  using ConvexBase::N;

  /// @brief common mat type for P
  using Mat = linalg::Matrix<T, N, N>;

  /**
   * @brief CTOR for ellipsoidal set
   *
   * @param center point (i.e., n-dim vector of type T)
   * @param P \in S^n_+
   */
  constexpr explicit Ellipsoid(const Point& c_in, const Mat& P_in)
      : ConvexBase(default_ellipsoid_map<CoordMat>(c_in)), c(c_in), P(P_in) {}

  /**
   * @brief CTOR for ellipsoid set
   *
   * @param center \in R^n
   * @param P \in S^n_+
   * @param affine map (x \in R^n -> cone coords)
   */
  constexpr explicit Ellipsoid(const Point& c_in, const Mat& P_in,
                               const affine::AffineMap<CoordMat>& map_in)
      : ConvexBase(map_in), c(c_in), P(P_in) {}

  /**
   * @brief contains function to see if x is in the ellipsoid (x - c)^T P^{-1}
   * (x - c) <= 1
   *
   * @param point x of type T with n elements
   * @param tolerance of tyep T (ignored)
   *
   * @return T/F
   */
  constexpr bool contains_impl(const Point& x, const T& tol) const {
    return contains_impl(std::tuple(x, T(1)), tol);
  }

  /**
   * @brief contains function to see if x is in the ellipsoid
   *
   * @param (x, t) \in R^n x R_++
   * @param tolerance (scalar)
   *
   * @return T/F
   */
  constexpr bool contains_impl(const std::tuple<Point, T>& x_in,
                               const T& /*tol*/) const {
    const auto x_proj = this->project(x_in);
    const auto x = std::get<0>(x_proj);
    const auto s = std::get<1>(x_proj);

    const auto x_T = linalg::transpose(as_col_vector(x));
    const auto res = (x_T * (linalg::inverse(P) * x));
    static_assert(linalg::vec_size_v<decltype(res)> == 1);

    switch (Type) {
      case EllipsoidType::OE:
        return res(0) < s * s;
      case EllipsoidType::CE:
      default:
        return res(0) <= s * s;
    }
    return false;
  }

  /**
   * @brief is_well_formed implementation checks to see if set is valid
   *
   * @return T/F
   */
  constexpr bool is_well_formed_impl() const {
    linalg::SymmetricMatrix S(P);
    return S.type == linalg::SymmetricMatrixType::PD ||
           S.type == linalg::SymmetricMatrixType::PSD;
  }

  /**
   * @brief transform set from an affine map
   *
   * E = {x | (x - c)^T P (x - c) <= 1}
   * f(x) = U x + t -> { y = f(x) | x \in E }
   * x = U^{-1}(y - t), sub back in we get...
   * E' = {y | (U^{-1}(y - t) - c)^T P (U^{-1}(y - t) - c) <= 1}
   * ... U^{-1}(y - t) - c \to U^{-1}(y - (U c + t)
   * E' = { y | (y - (U c + t))^T U^{-T} P U^{-1} (y - (U c + t)) <= 1
   *
   * c' = U c + t
   * P' = U^{-T} P u
   *
   * @param Affine map
   *
   * @return transformed ellipsoid
   */
  template <linalg::SquareMatrixLike Mat>
    requires(linalg::mat_cols_v<Mat> == N &&
             std::same_as<T, linalg::mat_type_t<Mat>>)
  constexpr auto transform_impl(const affine::AffineMap<Mat>& map) const {
    const auto U_inv = linalg::inverse(map.linear_part());
    return Ellipsoid<Space, Type, Tag>(
        map.linear_part() * c + map.translation(),
        transpose(U_inv) * P * U_inv);
  }

  /**
   * @brief equality comparison
   *
   * @param convex set
   *
   * @return T/F
   */
  constexpr bool is_equal_impl(const ConvexBase& other) const {
    return other.self().c == c && other.self().P == P;
  }

  /**
   * @brief inverse perspective func from convex set to cone
   *
   * Cone lift: { (x, t) \in R^n x R_+ | (x - c t)^T P^{-1} (x - c t) <= t^2 }
   *
   * U = [ I ; -c^T ]
   *     [ 0 ; 1    ]
   *
   * @return cone
   */
  constexpr auto cone_view_impl() const {
    if constexpr (Tag == ViewTag::ConvexSet) {
      auto U = linalg::make_identity<T, N + 1>();

      for (size_t i = 0; i < N; ++i) U[i][N] = T(-1) * c(i);

      affine::AffineMap<CoordMat> map(U, linalg::Vector<T, N + 1>{});
      return ConeView(Ellipsoid<Space, Type, ViewTag::LiftedCone>(c, P, map));
    } else {
      return ConeView(Ellipsoid<Space, Type, Tag>(c, P, this->coord_map));
    }
  }

 private:
  /** @brief member variables in the ellipsoid */
  Point c;  //< n-dim vector of type T representing center of ellipsoid
  Mat P;    //< square matrix representing directions for ellipsoid (valid \in
            // S^n_+)
};

}  // namespace convex
