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
 * @brief group permutation
 *
 * @param vector of eigenvalues
 * @param tolerance
 *
 * @return vector of indices
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
 * @brief invert permutation
 *
 * @param vector representing group permuation
 *
 * @return inverted permutation
 */
template <std::size_t N>
constexpr auto invert_permutation(const Vector<std::size_t, N>& p) {
  Vector<std::size_t, N> p_inv;
  for (std::size_t i = 0; i < N; ++i) p_inv(p(i)) = i;
  return p_inv;
}

/**
 * @brief permute P^T J P
 *
 * @param Matrix J (jordan form)
 * @param vector p (permutation)
 *
 * @return permuted J
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
 * @brief unpermute P^T A P equivalent to P A P^T
 *
 * @param Matrix A (orig)
 * @param vector p (permutation)
 *
 * @return unpermuted matrix R
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
 * @brief matrix exponential solution for x(t) = e^{A t} x(0)
 *
 * @param matrix A type T with N x N elements
 *
 * @return resuting matrix
 */
template <SquareMatrix Mat>
constexpr auto matrix_exp(const Mat& A) {
  using T = matrix_scalar_t<Mat>;
  using ScalarT = real_type_t<T>;
  using Complex = std::complex<ScalarT>;

  constexpr auto N = mat_rows_v<Mat>;
  static_assert(N >= 1, "Matrix exponential requires at least 1 row");

  const auto eig_decomp = EigenDecomposition(A);
  if (!eig_decomp.converged) return Matrix<Complex, N, N>{};

  const auto V = eig_decomp.V;
  const auto [V_inv, event] = MatrixInverse<decltype(V)>()(V);
  if (event != linalg::NumericalEvent::NONE) return Matrix<Complex, N, N>{};
  const auto D = eig_decomp.lambda;
  const auto J = V_inv * (as_complex_matrix(A) * V);

  const auto p = make_group_permutation(D);
  const auto p_inv = invert_permutation(p);
  const auto Jp = permute_ptjp(J, p); // P^T J P

  Matrix<Complex, N, N> exp_Jp;
  std::size_t i = 0;
  while (i < N) {
    // (0) form block i to j-1
    std::size_t j = i + 1;
    while (j < N && is_same(D(p(j)), D(p(i)))) ++j;
    std::size_t chain_len = j - i;
    const auto eig = D(p(i));
    const auto exp_lambda = constexpr_exp_lambda(eig);

    // (1) set exp(Jp) to exp(lambda) if this is a unique eigenvalue
    if (chain_len == 1) {
      exp_Jp(i, i) = exp_lambda;
      i = j;
      continue;
    }

    // (2) process the jordan block N = (J_block - lambda I)
    Matrix<Complex, N, N> N_blk, E_blk;
    // 2.a set super diag to 1s on E block (E = I + N + N^2 / 2! + ...) up to
    // chain_len representing N^0 = I
    for (std::size_t a = 0; a < chain_len; ++a)
      E_blk(i + a, i + a) = Complex(1);
    // 2.b set N block (N = J_block - lambda I) where C = lambda I + N (7.6.5
    // Matrix Computations)
    for (std::size_t a = 0; a < chain_len; ++a)
      for (std::size_t b = a; b < chain_len; ++b)
        N_blk(i + a, i + b) = Jp(i + a, i + b) - (a == b ? eig : Complex(0));

    // 2.c E represents N block to the chain_len power
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

  const auto exp_J = unpermute_papt(exp_Jp, p_inv);
  return V * (exp_J * V_inv);
}

} // namespace linalg
