/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "matrix_inverse.hpp"

namespace linalg {

/**
 * @brief matrix exponential solution for x(t) = e^{A t} x(0)
 *
 * @param matrix A type T with N x N elements
 *
 * @return resuting matrix
 */
template <SquareMatrixLike Mat>
constexpr auto matrix_exp(const Mat& A) {
  using T = mat_type_t<Mat>;
  constexpr auto N = mat_rows_v<Mat>;
  static_assert(N >= 1);

  const auto eig_decomp = EigenDecomposition(A);

  using ST = scalar_type_t<T>;
  Matrix<std::complex<ST>, N, N> exp_lambda;
  if (!eig_decomp.get_converged()) return exp_lambda;

  const auto V = eig_decomp.get_eigenvectors();
  const auto D = eig_decomp.get_eigenvalues();

  for (size_t i = 0; i < N; ++i) exp_lambda[i][i] = constexpr_exp_lambda(D(i));

  return as_complex_matrix(V) * exp_lambda * as_complex_matrix(inverse(V));
}

}  // namespace linalg
