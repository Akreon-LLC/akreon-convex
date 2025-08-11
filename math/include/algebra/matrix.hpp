/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "vector.hpp"

namespace algebra {

template <typename T, size_t N>
  requires(FieldLike<T>)
constexpr Matrix<T, N, N> make_identity() {
  Matrix<T, N, N> I;
  for (size_t i = 0; i < N; ++i) I[i][i] = T(1);
  return I;
}

template <typename T, size_t Rows, size_t Cols>
  requires(FieldLike<T>)
constexpr Matrix<T, Rows, Cols> make_zeros() {
  Matrix<T, Rows, Cols> Z;
  for (size_t i = 0; i < Rows; ++i)
    for (size_t j = 0; j < Cols; ++j) Z[i][j] = T(0);
  return Z;
}

template <typename T, size_t N>
  requires(FieldLike<T> && !is_iterable_v<T>)
constexpr auto as_col_vector(const Vector<T, N>& X) {
  Matrix<T, N, 1> A;
  for (size_t i = 0; i < N; ++i) A[i] = X(i);
  return A;
}

template <typename T, size_t N>
  requires(FieldLike<T> && !is_iterable_v<T>)
constexpr auto as_row_vector(const Vector<T, N>& X) {
  Matrix<T, 1, N> A;
  for (size_t i = 0; i < N; ++i) A[0][i] = X(i);
  return A;
}

template <typename T, size_t Rows, size_t Cols>
  requires(FieldLike<T>)
constexpr auto get_col(const Matrix<T, Rows, Cols>& A, const size_t& idx) {
  assert(idx < Cols);
  Vector<T, Rows> X;
  for (size_t i = 0; i < Rows; ++i) X[i] = A(i)(idx);
  return X;
}

template <typename T, size_t Rows, size_t Cols>
  requires(FieldLike<T>)
constexpr auto get_row(const Matrix<T, Rows, Cols>& A, const size_t& idx) {
  assert(idx < Rows);
  return A(idx);
}

template <typename T, size_t Rows, size_t Cols>
  requires(FieldLike<T>)
constexpr auto get_diag(const Matrix<T, Rows, Cols>& A) {
  constexpr size_t mn = std::min(Rows, Cols);
  Vector<T, mn> diag;
  for (size_t i = 0; i < mn; ++i) diag[i] = A(i)(i);
  return diag;
}

template <typename T, size_t N>
  requires(FieldLike<T>)
constexpr auto make_diag(const Vector<T, N>& X) {
  Matrix<T, N, N> diag;
  for (size_t i = 0; i < N; ++i) diag[i][i] = X(i);
  return diag;
}

}  // namespace algebra
