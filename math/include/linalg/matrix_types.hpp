/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "linear_algebra.hpp"

namespace linalg {

/**
 * @brief is_positive_definite checks if a matrix's eigs are all positive (same
 * as symmetric positive definite)
 *
 * @param matrix T that is N x N
 *
 * @return T/F
 */
template <SquareMatrixLike Mat>
constexpr bool is_positive_definite(
    const Mat& A, scalar_type_t<mat_type_t<Mat>> tol = 1e-10) {
  // (0) check if the matrix is symmetric
  if constexpr (!is_complex_v<mat_type_t<Mat>>) {
    if (!all_equal(A, transpose(A))) return false;
  } else {
    if (!all_equal(A, conjugate_transpose(A))) return false;
  }

  // (1) check if eigs are strictly positive
  const auto eig_decomp = EigenDecomposition(A);
  if (!eig_decomp.get_converged()) return false;
  for (const auto& lambda : eig_decomp.get_eigenvalues())
    if (lambda.real() <= -tol) return false;
  return true;
}

/**
 * @brief is_positive_semidefinite checks if a matrics eigs are greater than or
 * equal to 0
 *
 * @param matrix T that is N x N
 *
 * @return T/F
 */
template <SquareMatrixLike Mat>
constexpr bool is_positive_semidefinite(
    const Mat& A, scalar_type_t<mat_type_t<Mat>> tol = 1e-10) {
  // (0) check if the matrix is symmetric
  if constexpr (!is_complex_v<mat_type_t<Mat>>) {
    if (!all_equal(A, transpose(A))) return false;
  } else {
    if (!all_equal(A, conjugate_transpose(A))) return false;
  }

  // (1) check if eigs are greater than or equal to zero
  const auto eig_decomp = EigenDecomposition(A);
  if (!eig_decomp.get_converged()) return false;
  for (const auto& lambda : eig_decomp.get_eigenvalues())
    if (lambda.real() < -tol) return false;
  return true;
}

}  // namespace linalg
