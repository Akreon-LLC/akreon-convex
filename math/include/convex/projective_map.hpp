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

#include "sets/convex_set.hpp"

namespace convex {

/**
 * @brief Projective Map
 *
 * This is a linear fractional mapping: Ax + b / (c^T x + d) which maps R^n to
 * R^{m+1}
 */
template <linalg::MatrixLike MatA, linalg::MatrixLike MatC>
  requires(std::same_as<linalg::mat_type_t<MatA>, linalg::mat_type_t<MatC>> &&
           linalg::mat_cols_v<MatA> == linalg::mat_cols_v<MatC> &&
           linalg::mat_rows_v<MatC> == 1)
class ProjectiveMap {
 public:
  /// @brief alias to extract underlying type of affine set
  using T = linalg::mat_type_t<MatA>;
  using ScalarT = linalg::scalar_type_t<T>;

  /// @brief alias for underlying vector spanning R^n (domain)
  using DomVec = linalg::Vector<T, linalg::mat_cols_v<MatA>>;

  /// @brief alias for underlying vector spanning codomain for A
  using AmbientVec = linalg::Vector<T, linalg::mat_rows_v<MatA>>;

  /**
   * @brief CTOR
   *
   * @param matrix A defining affine change of basis mapping
   * @param vector b defining affine offset
   * @param matrix C defining linear fractional denominator
   * @param scalar d defining offset
   */
  constexpr explicit ProjectiveMap(const MatA& A_in, const AmbientVec& b_in,
                                   const MatC& C_in, const ScalarT& d_in)
      : A(A_in), b(b_in), C(C_in), d(d_in) {}

  /**
   * @brief Get Q
   *
   * Q = [ A;   b ]
   *     [ c^T; d ]
   *
   * @return Q Matrix
   */
  constexpr auto Q() const {
    constexpr size_t M = linalg::mat_rows_v<MatA>;
    constexpr size_t N = linalg::mat_cols_v<MatA>;
    linalg::Matrix<T, M + 1, N + 1> Q;
    Q = set_block<0, M, 0, N>(Q, A);
    for (size_t i = 0; i < M; ++i) Q[i][N] = b(i);
    for (size_t i = 0; i < N; ++i) Q[M][i] = C(0)(i);
    Q[M][N] = d;
    return Q;
  }

  /**
   * @brief apply linear fractional (projection) map to domain vector
   *
   * @param domain vector x
   *
   * @return codomain vector (projection) [z; t]... y = z / t and t > 0
   */
  constexpr auto apply(const DomVec& x) const {
    constexpr size_t N = linalg::mat_cols_v<MatA>;
    linalg::Vector<T, N + 1> x_tilde;
    for (size_t i = 0; i < N; ++i) x_tilde[i] = x(i);
    x_tilde[N] = T(1);
    return Q() * x_tilde;
  }

 private:
  /** @brief member variables for Projective Map */
  MatA A;        //< basis for affine map
  AmbientVec b;  //< bias term for affine map
  MatC C;        //< row vector for linear fractional c^T
  ScalarT d;     //< linear fractional offset
};

}  // namespace convex
