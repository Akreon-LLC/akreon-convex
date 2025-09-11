/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "linear_algebra.hpp"

namespace linalg {

/**
 * @brief matrix inverse
 *
 * A A^{-1} = I
 *
 * @param matrix A of type T N x N elements
 *
 * @return matrix
 */
template <typename T, size_t N>
  requires(FieldLike<T> && N >= 1)
constexpr auto inverse(const Matrix<T, N, N>& A) {
  const auto I = make_identity<T, N>();
  Matrix<T, N, N> res;
  const auto qr_A = QR(A);
  for (size_t i = 0; i < N; ++i) {
    const auto x = qr_A.solve(get_col(I, i));
    for (size_t j = 0; j < N; ++j) res[j][i] = x(j);
  }
  return res;
}

/**
 * @brief operator overload for matrix inverse
 *
 * @param matrix A of type T N x N elements
 *
 * @return matrix
 */
template <typename T, size_t N>
  requires(FieldLike<T> && N >= 1)
constexpr auto operator/(const Matrix<T, N, N>& A, const Matrix<T, N, N>& B) {
  return A * inverse(B);
}

}  // namespace linalg
