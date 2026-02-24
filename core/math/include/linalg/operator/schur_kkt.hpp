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

#include "hessian.hpp"
#include "linalg/vector/algebra_traits.hpp"

namespace linalg {

/**
 * @brief KKT Schur Matrix
 *
 * Form assumes the following:
 *
 * Establish conditions for optimality of general convex (differentiable)
 * problem:
 * \nabla f_0(x^*) + \sum_i^m \lambda_i \nabla f_i(x^*) + \sum_i^p \nu_i \nabla
 * h_i(x^*) = 0
 *
 * (i)   f_i(x^*) <= 0
 * (ii)  h_i(x^*) = 0
 * (iii) \lambda_i^* >= 0
 * (iv)  \lambda_i^* f_i(x^*) = 0
 * (v)   \nabla f_0(x^*) + \sum_i^m \lambda_i^* \nabla f_i(x^*) + \sum_i^p \nu_i
 * \nabla h_i(x^*) = 0
 *
 * KKT System
 * [ H A^T ] [ x; nu ] = [ -q; b ]
 * [ A 0   ]
 *
 * Augmented KKT System
 * H = [ 0 0   ] \in R^{m+n x m+n}
 *     [ 0 H_x ]
 * A = [ -F I ] \in R^{m x m + n}
 * A^T = [ -F ] \in R^{m + n x m}
 *       [ I  ]
 *
 * S = F^T (H_x F)
 */
template <SquareMatrixLike HessianMat, MatrixLike EqualityMat>
  requires(std::same_as<mat_type_t<HessianMat>, mat_type_t<EqualityMat>>)
struct SchurKKT {
  /// @brief alias for type
  using T = mat_type_t<HessianMat>;

  /// @brief helpers for rows & sizes of vectors / matrices
  static constexpr size_t M = mat_rows_v<EqualityMat>;
  static constexpr size_t N = mat_cols_v<EqualityMat>;
  static constexpr T eps = 1e-10;

  /// @brief alias for mat and schur matrices
  using Schur = linalg::Matrix<T, N, N>;

  static_assert(mat_rows_v<HessianMat> == M,
                "Hessian Mat Rows != Equality Cols");

  /**
   * @brief make augmented schur
   *
   * \hat{S} = F^T H_x F
   *
   * @param H_x (hessian for cone var in lifted dom-- [ z; x ] \in R^n x E
   * @param F (cone block slice for affine equality)
   *
   * @return \hat{S}
   */
  constexpr auto make(
      const Hessian<HessianMat, DifferentialType::PUSHFORWARD>& H_x,
      const EqualityMat& F) const {
    const auto S = transpose(F) * (H_x.apply(F));
    return regularize(S);
  }

 private:
  /**
   * @brief tikhonov regularization
   *
   * @param Schur mat
   *
   * @return regularized
   */
  constexpr auto regularize(Schur S) const {
    const auto d = get_diag(S);
    T s{0};
    for (const auto& x : d)
      if (linalg::constexpr_abs(x) > s) s = linalg::constexpr_abs(x);

    s = std::max(s, T(1));
    const auto delta = eps * s;
    for (size_t i = 0; i < N; ++i) S(i)(i) += delta;
    return S;
  }
};

}  // namespace linalg
