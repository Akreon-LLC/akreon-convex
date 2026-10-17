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

#include "eig_decomp.hpp"
#include "linalg/linalg_util.hpp"

namespace linalg {

/**
 * @brief Group approximately equal eigenvalues into contiguous index runs.
 *
 * @tparam Vec Vector type with complex scalar entries.
 * @param[in] D Complex eigenvalue vector.
 * @param[in] tol Absolute tolerance for comparing eigenvalues.
 * @return Vector<std::size_t, N> mapping grouped positions to original indices.
 */
template <VectorContainer Vec>
  requires(Complex<vector_element_t<Vec>>)
constexpr auto make_group_permutation(
    const Vec& D, real_type_t<vector_element_t<Vec>> tol = 1e-10) {
  constexpr std::size_t N = vector_dim_v<Vec>;

  Vector<std::size_t, N> p;
  Vector<bool, N> taken;
  std::size_t k = 0;

  for (std::size_t i = 0; i < N; ++i) {
    if (taken(i)) continue;

    for (std::size_t j = i; j < N; ++j) {
      if (!taken(j) && is_same(D(j), D(i), tol)) {
        p(k++) = j;
        taken(j) = true;
      }
    }
  }

  return p;
}

/**
 * @brief Invert an index permutation.
 *
 * @tparam N Number of permuted indices.
 * @param[in] p Permutation of indices 0 through N-1.
 * @return Vector<std::size_t, N> satisfying p_inv(p(i)) = i.
 */
template <std::size_t N>
constexpr auto invert_permutation(const Vector<std::size_t, N>& p) {
  Vector<std::size_t, N> p_inv;
  for (std::size_t i = 0; i < N; ++i) p_inv(p(i)) = i;
  return p_inv;
}

/**
 * @brief Reorder matrix rows and columns by the same permutation.
 *
 * @tparam Mat Square matrix type.
 * @param[in] J Square matrix to reorder.
 * @param[in] p Permutation mapping new indices to original indices.
 * @return Matrix of type Mat with entry (i,j) = J(p(i),p(j)).
 */
template <MatrixContainer Mat>
constexpr auto permute_ptjp(const Mat& J,
                            const Vector<std::size_t, mat_rows_v<Mat>>& p) {
  constexpr std::size_t N = mat_rows_v<Mat>;
  Mat J_p;
  for (std::size_t i = 0; i < N; ++i)
    for (std::size_t j = 0; j < N; ++j) J_p(i, j) = J(p(i), p(j));
  return J_p;
}

/**
 * @brief Scatter matrix entries through a row-and-column permutation.
 *
 * @tparam Mat Square matrix type.
 * @param[in] A Square matrix whose entries are scattered.
 * @param[in] p Permutation mapping source indices to destination indices.
 * @return Matrix R of type Mat satisfying R(p(i),p(j)) = A(i,j).
 */
template <MatrixContainer Mat>
constexpr auto unpermute_papt(const Mat& A,
                              const Vector<std::size_t, mat_rows_v<Mat>>& p) {
  constexpr std::size_t N = mat_rows_v<Mat>;
  Mat R;
  for (std::size_t i = 0; i < N; ++i)
    for (std::size_t j = 0; j < N; ++j) R(p(i), p(j)) = A(i, j);
  return R;
}

/**
 * @brief Compute exp(A) using its eigenvector and generalized-eigenvector
 * basis.
 *
 * @tparam Mat Square matrix type.
 * @param[in] A Square input matrix.
 * @return Complex N-by-N matrix exp(A), or a zero matrix if decomposition or
 * inversion fails.
 */
template <SquareMatrix Mat>
constexpr auto matrix_exp(const Mat& A) {
  using T = matrix_scalar_t<Mat>;
  using ScalarT = real_type_t<T>;
  using Complex = std::complex<ScalarT>;

  constexpr auto N = mat_rows_v<Mat>;
  static_assert(N >= 1, "Matrix exponential requires at least 1 row");

  // Transform A to the eigenvector/generalized-eigenvector basis.
  const auto eig_decomp = EigenDecomposition(A);
  if (!eig_decomp.converged) return Matrix<Complex, N, N>{};

  const auto V = eig_decomp.V;
  const auto [V_inv, event] = MatrixInverse<decltype(V)>()(V);
  if (event != linalg::NumericalEvent::NONE) return Matrix<Complex, N, N>{};
  const auto D = eig_decomp.lambda;
  const auto J = V_inv * (as_complex_matrix(A) * V);

  // Group repeated eigenvalues into contiguous blocks of J.
  const auto p = make_group_permutation(D);
  const auto p_inv = invert_permutation(p);
  const auto Jp = permute_ptjp(J, p); // P^T J P

  Matrix<Complex, N, N> exp_Jp;
  std::size_t i = 0;
  while (i < N) {
    // Identify the block sharing the eigenvalue at index i.
    std::size_t j = i + 1;
    while (j < N && is_same(D(p(j)), D(p(i)))) ++j;
    std::size_t chain_len = j - i;
    const auto eig = D(p(i));
    const auto exp_lambda = constexpr_exp_lambda(eig);

    // A scalar block contributes exp(lambda).
    if (chain_len == 1) {
      exp_Jp(i, i) = exp_lambda;
      i = j;
      continue;
    }

    // For J_block = lambda*I + N, evaluate exp(lambda)*exp(N).
    Matrix<Complex, N, N> N_blk, E_blk;
    // Initialize exp(N) with its identity term.
    for (std::size_t a = 0; a < chain_len; ++a)
      E_blk(i + a, i + a) = Complex(1);
    // Extract the upper-triangular nilpotent part N = J_block - lambda*I.
    for (std::size_t a = 0; a < chain_len; ++a)
      for (std::size_t b = a; b < chain_len; ++b)
        N_blk(i + a, i + b) = Jp(i + a, i + b) - (a == b ? eig : Complex(0));

    // Accumulate I + N + N^2/2! + ... through the block's nilpotency bound.
    Matrix<Complex, N, N> P = N_blk;
    Complex inv_factorial(1);
    for (std::size_t m = 1; m < chain_len; ++m) {
      inv_factorial /= Complex(ScalarT(m));
      for (std::size_t a = 0; a < chain_len; ++a)
        for (std::size_t b = a; b < chain_len; ++b)
          E_blk(i + a, i + b) += (P(i + a, i + b) * inv_factorial);
      const auto upper = i + chain_len - 1;

      Matrix<Complex, N, N> N_sub_blk, P_sub_blk;
      for (std::size_t a = i; a < upper; ++a)
        for (std::size_t b = i; b < upper; ++b)
          N_sub_blk(a, b) = N_blk(a, b), P_sub_blk(a, b) = P(a, b);

      P = P_sub_blk * N_sub_blk;
    }

    for (std::size_t a = 0; a < chain_len; ++a)
      for (std::size_t b = a; b < chain_len; ++b)
        exp_Jp(i + a, i + b) = E_blk(i + a, i + b) * exp_lambda;

    i = j;
  }

  // Undo the grouping and map exp(J) back as V*exp(J)*V^-1.
  const auto exp_J = unpermute_papt(exp_Jp, p_inv);
  return V * (exp_J * V_inv);
}

} // namespace linalg
