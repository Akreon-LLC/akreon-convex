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

#include "linear_algebra.hpp"
#include "matrix_traits.hpp"

namespace linalg {

/// @brief simple enum to define symmetric matrix types
/// INV = Invalid
/// IND = Indefinite
/// PSD = Positive Semidefinite
/// PD = Positive Definite
/// NSD = Negative Semidefinite
/// ND = Negative Definite
enum struct SymmetricMatrixType { INV = 0, IND, PSD, PD, NSD, ND };

/**
 * @brief make symmetric matrix
 *
 * @param symmetric vector (flattened)
 *
 * @return matrix
 */
template <size_t N, VectorLike Vec>
  requires(vec_size_v<Vec> == N * (N + 1) / 2)
constexpr auto make_smat(const Vec& svec) {
  using T = vec_type_t<Vec>;
  Matrix<T, N, N> S;
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      const auto idx = i * (i + 1) / 2 + j;
      S[i][j] = svec(idx);
    }
  }
  const auto S_T = transpose(S);
  Matrix<T, N, N> res;
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      if (i >= j)
        res[i][j] = S(i)(j);
      else
        res[i][j] = S_T(i)(j);
    }
  }
  return res;
}

/**
 * @brief to lower triangular to vec
 *
 * @param matrix
 *
 * @return vec
 */
template <SquareMatrixLike Mat>
constexpr auto lower_triangle_to_vec(const Mat& A) {
  using T = mat_type_t<Mat>;
  static constexpr size_t N = mat_rows_v<Mat>;
  static constexpr size_t K = N * (N + 1) / 2;
  Vector<T, K> vec{};
  for (size_t i = 0, k = 0; i < N; ++i)
    for (size_t j = 0; j <= i; ++j, ++k) vec[k] = A(j)(i);
  return vec;
}

/**
 * @brief SymmetricMatrix definition
 *
 * This class depends on square like matrix type and tests if it's symmetric (A
 * = A^T or A = A^H) then stores in reduced flattend vector for look up
 */
template <SquareMatrixLike Mat>
struct SymmetricMatrix {
  using T = mat_type_t<Mat>;
  using ScalarT = scalar_type_t<T>;
  using ComplexT = std::complex<ScalarT>;

  static constexpr size_t N = mat_rows_v<Mat>;
  static constexpr size_t DIM = (N * (N + 1)) / 2;

  /**
   * @brief CTOR
   *
   * Creates symmetric matrix with matrix A type T N x N elements
   * - Checks if A is symmetric and sets data elements to vector with N * (N +
   * 1) / 2 elements
   * - Checks eigenvalue signs and sets type
   */
  constexpr explicit SymmetricMatrix(const Mat& A, ScalarT tol = 1e-8) {
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
      type = SymmetricMatrixType::INV;
      return;
    }

    type = SymmetricMatrixType::IND;

    // (dakre) store lower triangle
    svec = lower_triangle_to_vec(A);

    // (2) Next calculate the eigenvalues of the matrix and check signs of the
    // eigenvalues by running count
    const auto eig_decomp = EigenDecomposition(A);
    if (!eig_decomp.get_converged()) return;

    size_t psd = 0, pd = 0, nsd = 0, nd = 0;
    const auto lambdas = eig_decomp.get_eigenvalues();
    for (const auto& lambda : lambdas) {
      pd += lambda.real() > tol;
      psd += lambda.real() >= -tol;
      nd += lambda.real() < -tol;
      nsd += lambda.real() <= tol;
    }

    // (3) Set the type appropriately determined by the eigs above
    if (pd == N)
      type = SymmetricMatrixType::PD;
    else if (psd == N)
      type = SymmetricMatrixType::PSD;
    else if (nd == N)
      type = SymmetricMatrixType::ND;
    else if (nsd == N)
      type = SymmetricMatrixType::NSD;

    // (4) House keeping to store off determinant of matrix
    det = eig_decomp.get_determinant();
  }

  /**
   * @brief overload for accessor (provide row col pair and get value)
   *
   * @param size_t i = row index
   * @param size_t j = col index
   *
   * @return svec element
   */
  constexpr auto operator()(size_t i, size_t j) const {
    return svec(index(i, j));
  }

  /**
   * @brief index from (row, col)
   *
   * @param size_t i = row index
   * @param size_t j = col index
   *
   * @return index
   */
  constexpr size_t index(size_t i, size_t j) const {
    assert(j < N && i < N);
    if (i >= j) return i * (i + 1) / 2 + j;
    return j * (j + 1) / 2 + i;
  }

  /**
   * @brief is psd check
   *
   * @return T/F
   */
  constexpr bool is_psd() const {
    return type == SymmetricMatrixType::PD || type == SymmetricMatrixType::PSD;
  }

  /**
   * @brief is nsd check
   *
   * @return T/F
   */
  constexpr bool is_nsd() const {
    return type == SymmetricMatrixType::ND || type == SymmetricMatrixType::NSD;
  }

  /** @brief member variables */
  SymmetricMatrixType type;  //< SymmetricMatrixType value
  Vector<T, DIM> svec;  //< Data stored from input square matrix as flattened
                        // n(n+1)/2 dim vector
  std::optional<ComplexT> det;  //< result of determinant
};

}  // namespace linalg
