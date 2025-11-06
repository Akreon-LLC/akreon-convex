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
      : A(A_in), b(b_in) {}

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
   * (1) Standard Cone : C' = { (x,t) \in R^n x R_+ | A x = b t }
   * -> zero cone : C' = { (x,t) \in R^n x R_+ | Ax - bt = 0 }
   * (2) Epi/Hypo N/A right now
   *
   * @param (x,t) \in R^n x R
   * @param tolerance
   *
   * @return T/F
   */
  constexpr bool contains_impl(const std::tuple<Point, T>& x_in,
                               const T& tol) const {
    const auto x = std::get<0>(x_in);
    const auto t = std::get<1>(x_in);
    if (t < T(0)) return false;
    if (t == T(0) && !linalg::all_zero(x)) return false;
    return linalg::L2<AmbientVec>()(A * x - t * b) < tol;
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
    return A == other.self().A && b == other.self().b;
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
    if constexpr (Tag == ViewTag::ConvexSet)
      return ConeView(AffineSubspace<Space, Mat, ViewTag::LiftedCone>(A, b));
    else
      return ConeView(AffineSubspace<Space, Mat, Tag>(A, b));
  }

 private:
  /** @brief private data members for affine subspace */
  Mat A;         //< basis that spans the space of R^n
  AmbientVec b;  //< offset vector which translates the basis to a new origin
};

}  // namespace convex
