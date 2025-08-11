/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include <tuple>

#include "matrix_norms.hpp"
#include "vector_norms.hpp"

/* (dakre) Global TODO
 * 1. when field is complex it is mostly unhandled below (constexpr_sqrt not
 * ready for C case)
 * 2. punting on cholesky and LDL right now, with current approach and checking
 * positive definiteness im not seeing the performance boost needed for MVP,
 * perhaps later on integrating it would make sense
 */

namespace algebra {

template <typename T, size_t N, typename Norm>
  requires(FieldLike<T>)
constexpr auto normalize(const Vector<T, N>& X, const Norm& norm) {
  const auto norm_val = norm(X);
  assert(norm_val > 0);
  return (1 / norm_val) * X;
}

template <typename T, size_t N>
  requires(FieldLike<T>)
constexpr auto euclidean_distance(const Vector<T, N>& X,
                                  const Vector<T, N>& Y) {
  return L2(X - Y);
}

template <typename T, size_t N>
  requires(FieldLike<T>)
constexpr auto manhattan_distance(const Vector<T, N>& X,
                                  const Vector<T, N>& Y) {
  return L1(X - Y);
}

template <typename T, size_t Rows, size_t Cols>
  requires(FieldLike<T>)
constexpr bool all_equal(const algebra::Matrix<T, Rows, Cols>& A,
                         const algebra::Matrix<T, Rows, Cols>& B) {
  bool rv = true;
  for (size_t i = 0; i < Rows; ++i)
    rv &= euclidean_distance(A(i), B(i)) <= 1e-10;
  return rv;
}

template <typename T, size_t Rows, size_t Cols>
  requires(FieldLike<T>)
constexpr bool rows_equal_cols(const Matrix<T, Rows, Cols>& A,
                               const Matrix<T, Cols, Rows>& B) {
  bool rv = true;
  for (size_t i = 0; i < Rows; ++i)
    rv &= euclidean_distance(A(i), get_col(B, i)) <= 1e-10;
  return rv;
}

template <typename T, size_t N>
  requires(FieldLike<T> && !is_complex_v<T>)
constexpr bool is_symmetric(const Matrix<T, N, N>& A) {
  return all_equal(A, transpose(A));
}

// (dakre) TODO below is gram-schmidt in the future update to be
// householder (improved accuracy)
template <typename T, size_t Rows, size_t Cols>
  requires(Rows > 0 && FieldLike<T>)
constexpr auto qr(const Matrix<T, Rows, Cols>& A) {
  // 1. edge case if A is all zeros current GS algo does not handle this well
  // so return I and zeros mat
  Matrix<T, Rows, Cols> Q;
  Matrix<T, Cols, Cols> R;
  if (all_equal(A, Matrix<T, Rows, Cols>{})) {
    for (size_t i = 0; i < Cols; ++i) Q[i][i] = T(1);
    return std::tuple(Q, R);
  }

  // 2. gram schmidt algorithm below
  for (size_t i = 0; i < Cols; ++i) {
    auto v = get_col(A, i);
    for (size_t j = 0; j < i; ++j) {
      R[j][i] = dot(get_col(Q, j), v);
      v = v - R[j][i] * get_col(Q, j);
    }
    const auto q_i = normalize(v, L2<T, Rows>);
    for (size_t j = 0; j < Rows; ++j) Q[j][i] = q_i(j);
    R[i][i] = dot(q_i, get_col(A, i));
  }
  return std::tuple(Q, R);
}

// A x = b => Q R x = b => R x = Q b^T
template <typename T, size_t Rows, size_t Cols>
  requires(FieldLike<T>)
constexpr auto lsq(const Matrix<T, Rows, Cols>& Q,
                   const Matrix<T, Cols, Cols>& R, const Vector<T, Rows>& b) {
  Vector<T, Cols> x, c;

  if constexpr (is_complex_v<T>)
    c = hermitian(Q) * b;
  else
    c = transpose(Q) * b;

  for (int i = static_cast<int>(Cols) - 1; i >= 0; --i) {
    T sum = 0;
    for (size_t j = i + 1; j < Cols; ++j) sum += (R(i)(j) * x(j));
    x[i] = (c(i) - sum) / R(i)(i);
  }

  return x;
}

template <typename T, size_t Rows, size_t Cols>
  requires(Rows == Cols && FieldLike<T>)
constexpr auto eigen_decomposition(const Matrix<T, Rows, Cols>& A,
                                   size_t max_iter = 100, T tol = 1e-10) {
  Matrix<T, Rows, Cols> Lambda = A;
  auto X = make_identity<T, Cols>();

  // Eigendecomposition works with diagonalizing the matrix to solve A X =
  // X \Lambda (\Lambda == eigenvalue matrix which is diagonal), the resulting X
  // is our eigenvector associated with the eigenvalues
  // 1. A_0 = A
  // 2. A_k = Q_k R_k
  // 3. A_{k+1} = R_k Q_k \equiv Q_k^T A_k Q_k similarity transform
  for (size_t i = 0; i < max_iter; ++i) {
    const auto QR = qr(Lambda);
    const auto Q = std::get<0>(QR);
    const auto R = std::get<1>(QR);
    Lambda = R * Q;
    X = X * Q;
    const auto off_diag = Lambda - make_diag(get_diag(Lambda));
    if (frobenius_norm(off_diag) < tol) break;
  }

  return std::tuple(Lambda, X);
}

template <typename T, size_t N>
  requires(FieldLike<T>)
constexpr Vector<T, N> get_eigenvalues(
    const std::tuple<Matrix<T, N, N>, Matrix<T, N, N>>& eig_decomp) {
  const auto Lambda = std::get<0>(eig_decomp);
  return get_diag(Lambda);
}

// A = U \Sigma V^T
template <typename T, size_t Rows, size_t Cols>
  requires(FieldLike<T> && !is_complex_v<T>)
constexpr auto svd(const Matrix<T, Rows, Cols>& A, size_t max_iter = 100,
                   T tol = 1e-10) {
  auto M = transpose(A) * A;
  const auto result = eigen_decomposition(M, max_iter, tol);

  const auto lambdas = get_diag(std::get<0>(result));
  // right singular vectors space row space of Lambda above U (A^T A) VT -> U
  // \Sigma VT
  auto VT = std::get<1>(result);

  Vector<T, Cols> sigmas;
  for (size_t i = 0; i < Cols; ++i)
    sigmas[i] = constexpr_sqrt(std::max<T>(0, lambdas(i)));

  // \Sigma^{-1} A V = U
  auto U = make_identity<T, Rows>();

  const auto V = transpose(VT);

  constexpr auto mn = std::min(Rows, Cols);
  for (size_t i = 0; i < mn; ++i) U[i] = (1 / sigmas(i)) * A * get_col(V, i);

  U = transpose(U);

  U = std::get<0>(qr(U));
  VT = std::get<0>(qr(VT));

  return std::tuple(U, sigmas, VT);
}

}  // namespace algebra
