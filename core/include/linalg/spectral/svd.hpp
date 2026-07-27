/* Copyright © 2025 Akreon LLC. All rights reserved.
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

#include "config/assert.hpp"
#include "eig_decomp.hpp"

namespace linalg {

/**
 * @brief Singular value decomposition class definition
 *
 * Decompose any matrix A into U Sigma V^T where U are the left singular
 * vectors, Sigma are the singular values, and V^T are the right singular
 * vectors
 */
template <MatrixContainer Mat>
struct SVD {
  // @brief aliasing complex and real types for eigen decomposition
  using T = matrix_scalar_t<Mat>;
  using Real = real_type_t<T>;
  using Complex = std::conditional_t<linalg::Complex<T>, T, std::complex<Real>>;

  // @brief helpers for Rows & Cols
  static constexpr std::size_t Rows = mat_rows_v<Mat>;
  static constexpr std::size_t Cols = mat_cols_v<Mat>;

  /**
   * @brief CTOR
   *
   * @param matrix A of type T with Rows x Cols elements
   * @param tolerance dictating algorithm convergence
   */
  constexpr SVD(const Mat& A) : converged(false) { svd(A); }

  /** @brief SVD member variables */
  Matrix<Complex, Cols, Cols> VT; ///< right singular vectors
  Matrix<Complex, Rows, Rows> U;  ///< left singular vectors
  Matrix<Real, Rows, Cols> S;     ///< singular values
  bool converged;                 ///< converged

private:
  /**
   * @brief solve A = U Sigma V^T
   *
   * @param matrix A of type T with Rows x Cols elements
   *
   * @return (void)
   */
  constexpr void svd(const Mat& A) {
    // 1. compute A^T A then solve eigen decomposition
    Matrix<T, Cols, Cols> M;
    if constexpr (linalg::Real<T>) M = transpose(A) * A;
    else M = conjugate_transpose(A) * A;

    const auto result = EigenDecomposition(M);
    converged = result.converged;
    if (!converged) return;

    // 2. convert eigenvalues to singular values
    const auto tmp_lambdas = result.lambda;
    Vector<Real, Cols> lambdas;
    for (std::size_t i = 0; i < Cols; ++i) {
      AK_ASSERT(is_real(tmp_lambdas(i)) &&
                "SVD expects real eigenvalues from A^*A");
      if (!is_real(tmp_lambdas(i))) {
        converged = false;
        return;
      }

      lambdas(i) = tmp_lambdas(i).real();
    }

    // (dakre) these eigenvectors are projected onto the row space thus
    // resembling the right singular vectors of the row space of Lambda above
    // U (A^T A) VT -> U Sigma VT
    const auto& V = result.V;

    for (std::size_t i = 0; i < Cols; ++i)
      S(i, i) = constexpr_sqrt_safe(std::max<Real>(0, lambdas(i)));

    // 3. back-solve for U with VT & sigmas
    // Sigma^{-1} A V = U
    U = make_identity<Complex, Rows>();

    for (std::size_t i = 0; i < Cols; ++i) {
      const auto u_i = (1 / S(i, i)) * as_complex_matrix(A) * get_col(V, i);
      for (std::size_t j = 0; j < Rows; ++j) U(j, i) = u_i(j);
    }

    VT = conjugate_transpose(V);

    // 4. sort singular values & vectors largest -> smallest
    sort_svd_desc();
  }

  /**
   * @brief sort SVD result in descending manner
   *
   * @return (void)
   */
  constexpr void sort_svd_desc() {
    // 1. acquire permutation indices
    const auto P = argsort_desc(get_diag(S));

    // 2. pemute singular vectors and values
    Matrix<Real, Rows, Cols> S_tmp;
    Matrix<Complex, Rows, Rows> U_tmp;
    Matrix<Complex, Cols, Cols> VT_tmp;
    for (std::size_t i = 0; i < Cols; ++i) {
      const auto idx = P[i];
      S_tmp(i, i) = S(idx, idx);

      for (std::size_t r = 0; r < Rows; ++r) U_tmp(r, i) = U(r, idx);

      for (std::size_t c = 0; c < Cols; ++c) VT_tmp(i, c) = VT(idx, c);
    }

    // 3. store result
    S = S_tmp;
    U = U_tmp;
    VT = VT_tmp;
  }
};

} // namespace linalg
