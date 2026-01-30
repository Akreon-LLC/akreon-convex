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

#include "linalg_util.hpp"
#include "matrix_traits.hpp"
#include "symmetric_operator.hpp"

namespace linalg {

/**
 * @brief SymmetricMatrix definition
 *
 * This class depends on square like matrix type and tests if it's symmetric (A
 * = A^T or A = A^H) then stores in reduced flattend vector for look up
 */
template <SquareMatrixLike Mat>
struct SymmetricMatrix : SymmetricOperator<Mat, SymmetricMatrix<Mat>> {
  /// @brief alias for symmetric matrix
  using SymmetricBase = SymmetricOperator<Mat, SymmetricMatrix<Mat>>;
  using typename SymmetricBase::ComplexT;
  using typename SymmetricBase::ScalarT;
  using typename SymmetricBase::T;

  /// @brief helpers for symmetric matrix
  using SymmetricBase::DIM;
  using SymmetricBase::N;

  /**
   * @brief CTOR
   *
   * Creates symmetric matrix with matrix A type T N x N elements
   * - Checks if A is symmetric and sets data elements to vector with N * (N +
   * 1) / 2 elements
   * - Checks eigenvalue signs and sets type
   */
  constexpr explicit SymmetricMatrix(const Mat& A, ScalarT tol = 1e-8)
      : SymmetricBase(A, tol) {}

  /**
   * @brief QR factorization
   *
   * @param Mat A
   * @param tol
   *
   * @return void
   */
  constexpr void factor(const Mat& A, const ScalarT& tol) {
    // (0) First see if the incoming matrix is symmetric
    bool valid = false;
    if constexpr (RealLike<T>)
      valid = all_equal(A, transpose(A));
    else
      valid = all_equal(A, conjugate_transpose(A));

    // (1) No need to set data for invalid one and set type
    // likewise if it is symmetric we can store it temporarily
    // as indefinite one
    if (!valid) {
      this->type = SymmetricMatrixType::INV;
      return;
    }

    this->type = SymmetricMatrixType::IND;

    // (dakre) store lower triangle
    this->L = lower_triangle_to_vec(A);

    // (2) Next calculate the eigenvalues of the matrix and check signs of the
    // eigenvalues by running count
    const auto eig_decomp = EigenDecomposition(A);
    if (!eig_decomp.get_converged()) return;

    // (3) Compute definiteness of matrix
    compute_definiteness(eig_decomp.get_eigenvalues(), tol);
  }

  /**
   * @brief compute definiteness
   *
   * @param vec (diagonal of elements)
   *
   * @return void
   */
  template <typename DiagT>
  constexpr void compute_definiteness(const Vector<DiagT, N>& diag,
                                      const ScalarT& tol) {
    size_t npos = 0, nneg = 0, nzero = 0;

    auto classify_scalar = [&](const auto& d_real) {
      if (d_real > tol)
        ++npos;
      else if (d_real < -tol)
        ++nneg;
      else
        ++nzero;
    };

    if constexpr (is_complex_v<DiagT>) {
      for (const auto& d : diag) classify_scalar(d.real());
    } else {
      for (const auto& d : diag) classify_scalar(d);
    }

    if (nneg == 0 && nzero == 0)
      this->type = SymmetricMatrixType::PD;
    else if (nneg == 0 && nzero > 0)
      this->type = SymmetricMatrixType::PSD;
    else if (npos == 0 && nzero == 0)
      this->type = SymmetricMatrixType::ND;
    else if (npos == 0 && nzero > 0)
      this->type = SymmetricMatrixType::NSD;
    else
      this->type = SymmetricMatrixType::IND;
  }
};

}  // namespace linalg
