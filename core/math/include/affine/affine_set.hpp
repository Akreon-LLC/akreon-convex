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

#include "affine_traits.hpp"
#include "linalg/linalg.hpp"

namespace affine {

/**
 * @brief ParametricAffineSet class declaration
 *
 * The set takes the form x0 + span(U1, U2,...Un) where Us are valid column
 * vectors that form a basis for the affine set S = {x0 + U t}
 */
template <linalg::MatrixLike Mat>
class ParametricAffineSet {
 public:
  /// @brief alias to extract underlying type of affine set
  using T = linalg::mat_type_t<Mat>;

  /// @brief helpers to extracting M x N matrix
  static constexpr size_t M = linalg::mat_rows_v<Mat>;
  static constexpr size_t N = linalg::mat_cols_v<Mat>;

  /// @brief alias for underlying vector spanning R^n (domain)
  using DomVec = linalg::Vector<T, N>;

  /// @brief alias for underlying vector spanning R^m (codomain)
  using AmbientVec = linalg::Vector<T, M>;

  /// (dakre) TODO refactor to drop this unecessary constraint
  /// @brief static assertion to check that the system is over determined
  static_assert(
      N >= M, "Parametric affine set currently requires overdetermined system");

  /**
   * @brief CTOR for parametric affine set
   *
   * @param matrix of type T with N x M elements
   * @param vector of type T with M elements
   */
  constexpr explicit ParametricAffineSet(const Mat& U_in,
                                         const AmbientVec& x0_in)
      : U(U_in) {
    this->x0 = x0_in;
  }

  /**
   * @brief is_well_formed_impl
   *
   * Checks to see if the basis is full column rank
   *
   * @return T/F
   */
  constexpr bool is_well_formed() const { return linalg::QR(U).rank() == M; }

  /**
   * @brief return reference point in set
   *
   * @return x0
   */
  constexpr auto point() const { return x0; }

  /**
   * @brief simple basis return
   *
   * @return U
   */
  constexpr auto basis() const { return U; }

  /**
   * @brief project x onto the affine set
   *
   * This one is much simpler than S = {x | Ax = b}, we just perform QR
   * factorization on U and back solve for the residual vector (x - x0) which
   * gives us a t vector. The t vector is then projected onto the basis U and
   * added by the reference point x0 to obtain the new projection
   *
   * @param vector of type T M elements
   *
   * @return projected x on affine set
   */
  constexpr auto project(const DomVec& x) const {
    const auto t = linalg::QR(U).solve(x - this->x0);
    return this->x0 + U * t;
  }

  /**
   * @brief transform affine set by map
   *
   * This one is also more simple than implicit case, we just simply perform the
   * change of basis on U, and then apply the translation directly to the
   * reference point.
   *
   * @param reference to a valid affine map
   *
   * @return parametric affine set object
   */
  constexpr auto transform(const AffineMap<Mat>& map) const {
    return ParametricAffineSet(
        map.linear_part() * U,
        map.linear_part() * this->x0 + map.translation());
  }

  /**
   * @brief contains
   *
   * Simple check to see if the vector lies in the affine set as defined by the
   * parametric set
   *
   * @param vector of type T with M elements
   * @param tolerance scalar
   *
   * @return T/F
   */
  constexpr bool contains(const AmbientVec& x, T tol = 1e-8) const {
    const auto r = x - this->x0;
    const auto t = linalg::QR(U).solve(r);
    const auto residual = U * t - r;
    return linalg::L2<linalg::Vector<T, N>>()(residual) < tol;
  }

  /**
   * @brief equality comparison overload
   *
   * @param affine set
   *
   * @return T/F
   */
  constexpr bool operator==(const ParametricAffineSet<Mat>& other) const {
    return is_equal(other);
  }

  /**
   * @brief inequality comparison overload
   *
   * @param affine set
   *
   * @return T/F
   */
  constexpr bool operator!=(const ParametricAffineSet<Mat>& other) const {
    return !is_equal(other);
  }

 private:
  /**
   * @brief equality comparison
   *
   * @param affine set
   *
   * @return T/F
   */
  constexpr bool is_equal(const ParametricAffineSet<Mat>& other) const {
    const auto basis_equal = all_equal(other.basis(), basis());
    const auto x0_equal = all_equal(other.point(), x0);
    return basis_equal && x0_equal;
  }

  /** @brief member variables in the base class */
  Mat U;          //< basis matrix for the affine set
  AmbientVec x0;  ///< reference point that lies in the affine set S \subset R^m
};

}  // namespace affine
