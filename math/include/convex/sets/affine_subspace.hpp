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
 * @brief default coordinate map for affine subspace
 *
 * affine coords: (x, b) -> A x = b
 * [ A -b ] [ x; t ] = 0
 * [ 0  1 ]
 *
 * [A x - b t = 0; t]
 *
 * @return affine map -> affine coords
 */
template <linalg::MatrixLike CoordMat, linalg::MatrixLike Mat,
          linalg::VectorLike Vec>
  requires(linalg::mat_rows_v<CoordMat> == linalg::vec_size_v<Vec> + 1 &&
           linalg::mat_rows_v<CoordMat> == linalg::mat_rows_v<Mat> + 1 &&
           linalg::mat_cols_v<CoordMat> == linalg::mat_cols_v<Mat> + 1)
constexpr auto default_affine_map(const Mat& A, const Vec& b) {
  using T = linalg::mat_type_t<CoordMat>;
  static constexpr size_t M = linalg::mat_rows_v<Mat>;
  static constexpr size_t N = linalg::mat_cols_v<Mat>;
  CoordMat U;
  U = linalg::set_block<0, M, 0, N>(U, A);
  U = linalg::set_block<0, M, N, N + 1>(U, linalg::as_col_vector(T(-1) * b));
  U[M][N] = T(1);
  return affine::AffineMap<CoordMat>(U, linalg::Vector<T, M + 1>{});
}

/**
 * @brief Affine Set in the perspective of an affine equality set
 *
 * C = { x | A x = b }
 * C is an affine subspace of R^n where A \in R^{m x n} and b \in R^m
 */
template <linalg::RnSpace Space, linalg::MatrixLike Mat,
          ViewTag Tag = ViewTag::ConvexSet>
  requires(linalg::space_dim_v<Space> ==
               linalg::vec_size_v<linalg::space_point_t<Space>> &&
           linalg::space_dim_v<Space> == linalg::mat_cols_v<Mat>)
class AffineSubspace
    : public ConvexSet<Space, AffineSubspace<Space, Mat, Tag>> {
 public:
  /// @brief bring into scope the type traits from the base class
  using ConvexBase = ConvexSet<Space, AffineSubspace<Space, Mat, Tag>>;
  using typename ConvexBase::CoordMat;
  using typename ConvexBase::Point;
  using typename ConvexBase::T;

  /// @brief bring into scope the const data members
  using ConvexBase::Dim;
  using ConvexBase::N;

  /// @brief helper to get number of rows in basis
  static constexpr size_t M = linalg::mat_rows_v<Mat>;

  /// @brief alias for ambient vector
  using AmbientVec = linalg::Vector<T, M>;

  /**
   * @brief CTOR for affine subspace
   *
   * @param A \in R^{m x n}
   * @param b \in R^m
   */
  constexpr explicit AffineSubspace(const Mat& A_in, const AmbientVec& b_in)
      : ConvexBase(default_affine_map<CoordMat>(A_in, b_in)),
        A(A_in),
        b(b_in) {}

  /**
   * @brief CTOR for affine subspace
   *
   * @param A \in R^{m x n}
   * @param b \in R^m
   * @param coord map
   */
  constexpr explicit AffineSubspace(const Mat& A_in, const AmbientVec& b_in,
                                    const affine::AffineMap<CoordMat>& map_in)
      : ConvexBase(map_in), A(A_in), b(b_in) {}

  /**
   * @brief contains function
   *
   * @param point x \in R^n
   * @param tolerance
   *
   * @return T/F
   */
  constexpr bool contains_impl(const Point& x, const T& tol) const {
    return contains_impl(std::tuple(x, T(1)), tol);
  }

  /**
   * @brief contains function for lifted subspace
   *
   * Standard Cone : C' = { (x,t) \in R^n x R_+ | A x = b t }
   * -> zero cone : C' = { (x,t) \in R^n x R_+ | Ax - bt = 0 }
   *
   * @param (x,t) \in R^n x R
   * @param tolerance
   *
   * @return T/F
   */
  constexpr bool contains_impl(const std::tuple<Point, T>& x_in,
                               const T& tol) const {
    // (dakre) x_proj = A x - b t
    const auto x_proj = this->project(x_in);
    const auto y = std::get<0>(x_proj);
    return linalg::L2<AmbientVec>()(y) < tol;
  }

  /**
   * @brief is well formed checks to see if basis fully spans subspace
   *
   * @return T/F
   */
  constexpr bool is_well_formed_impl() const {
    const auto rank = linalg::QR(A).get_rank();
    return rank == N;
  }

  /**
   * @brief equality comparison
   *
   * @param affine set
   *
   * @return T/F
   */
  constexpr bool is_equal_impl(const ConvexBase& other) const {
    return A == other.self().A && b == other.self().b &&
           other.coord_map == this->coord_map;
  }

  /**
   * @brief transform
   *
   * C = { x | A x = b }
   * f(C) = { y = f(x) | x \in C }
   * y = U x + t \to x = U^{-1} (y - t)
   * A U^{-1} (y - t) = b \to A U^{-1} y = b + A U^{-1} t
   *
   * @param affine map
   *
   * @return transformed affine subspace
   */
  template <linalg::SquareMatrixLike TMat>
    requires(linalg::mat_cols_v<TMat> == N &&
             std::same_as<linalg::mat_type_t<TMat>, T>)
  constexpr auto transform_impl(const affine::AffineMap<TMat>& map) const {
    const auto A_tilde = A * linalg::inverse(map.linear_part());
    const auto b_tilde = b + A_tilde * map.translation();
    return AffineSubspace<Space, Mat, Tag>(A_tilde, b_tilde);
  }

  /**
   * @brief to cone view
   *
   * @return cone
   */
  constexpr auto cone_view_impl() const {
    if constexpr (Tag == ViewTag::ConvexSet) {
      return ConeView(AffineSubspace<Space, Mat, ViewTag::LiftedCone>(
          A, b, this->coord_map));
    } else {
      return ConeView(AffineSubspace<Space, Mat, Tag>(A, b, this->coord_map));
    }
  }

 private:
  /** @brief private data members for affine subspace */
  Mat A;         //< basis that spans the space of R^n
  AmbientVec b;  //< offset vector which translates the basis to a new origin
};

}  // namespace convex
