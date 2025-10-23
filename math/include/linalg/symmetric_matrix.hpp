/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
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
    if constexpr (!is_complex_v<Mat>)
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

    for (size_t i = 0; i < N; ++i)
      for (size_t j = i; j < N; ++j) data[index(i, j)] = A(i)(j);

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
   * @return data element
   */
  constexpr auto operator()(size_t i, size_t j) const {
    return data(index(i, j));
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
    if (j < i) std::swap(i, j);
    const size_t idx = (j * (j + 1)) / 2 + i;
    return idx;
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
  Vector<T, DIM>
      data;  //< Data stored from input square matrix as flattened n-dim vector
  std::optional<ComplexT> det;  //< result of determinant
};

}  // namespace linalg
