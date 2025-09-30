/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "linear_algebra.hpp"

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
  using ScalarT = scalar_type_t<mat_type_t<Mat>>;

  static constexpr size_t N = mat_rows_v<Mat>;
  static constexpr size_t DIM = (N * (N + 1)) / 2;

  constexpr explicit SymmetricMatrix(const Mat& A, ScalarT tol = 1e-10) {
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
      for (size_t j = i; j < N; ++j) data[i] = A(i)(j);

    // (2) Next calculate the eigenvalues of the matrix and check signs of the
    // eigenvalues by running count
    const auto eig_decomp = EigenDecomposition(A);
    if (!eig_decomp.get_converged()) return;

    size_t psd = 0, pd = 0, nsd = 0, nd = 0;
    for (const auto& lambda : eig_decomp.get_eigenvalues()) {
      pd += lambda.real() > ScalarT(0);
      psd += lambda.real() >= -tol;
      nd += lambda.real() < ScalarT(0);
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
  }

  constexpr size_t idx(size_t i, size_t j) const {
    if (j > N || i > N) return size_t(-1);
    if (j < i) std::swap(i, j);
    return (N * i) + j;
  }

  constexpr auto operator()(size_t idx) const {
    if (idx >= DIM) return NULL;
    return data(idx);
  }

  SymmetricMatrixType type;
  Vector<T, DIM> data;
};

}  // namespace linalg
