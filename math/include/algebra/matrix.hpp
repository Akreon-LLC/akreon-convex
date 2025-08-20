/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "vector.hpp"
#include "vector_ops.hpp"

namespace algebra {

/**
 * @brief create an identity matrix of size N type T
 *
 * @return I
 */
template <typename T, size_t N>
  requires(FieldLike<T>)
constexpr Matrix<T, N, N> make_identity() {
  Matrix<T, N, N> I;
  for (size_t i = 0; i < N; ++i) I[i][i] = T(1);
  return I;
}

/**
 * @brief create a matrix filled with zeros of size Rows x Cols of type T
 *
 * @return Zeros
 */
template <typename T, size_t Rows, size_t Cols>
  requires(FieldLike<T>)
constexpr Matrix<T, Rows, Cols> make_zeros() {
  Matrix<T, Rows, Cols> Z;
  for (size_t i = 0; i < Rows; ++i)
    for (size_t j = 0; j < Cols; ++j) Z[i][j] = T(0);
  return Z;
}

/**
 * @brief express the vector X as a column vector (N x 1)
 *
 * @param vector X of type T and N elements
 *
 * @return (N x 1) matrix
 */
template <typename T, size_t N>
  requires(FieldLike<T> && !is_iterable_v<T>)
constexpr auto as_col_vector(const Vector<T, N>& X) {
  Matrix<T, N, 1> A;
  for (size_t i = 0; i < N; ++i) A[i] = X(i);
  return A;
}

/**
 * @brief express the vector X as a row vector (1 x N)
 *
 * @param vector X of type T and N elements
 *
 * @return (1 x N) matrix
 */
template <typename T, size_t N>
  requires(FieldLike<T> && !is_iterable_v<T>)
constexpr auto as_row_vector(const Vector<T, N>& X) {
  Matrix<T, 1, N> A;
  for (size_t i = 0; i < N; ++i) A[0][i] = X(i);
  return A;
}

/**
 * @brief express matrix as complex one
 *
 * @param Matrix A type T with Rows x Cols elements
 *
 * @return A \in C
 */
template <typename T, size_t Rows, size_t Cols>
  requires(FieldLike<T>)
constexpr auto as_complex_matrix(const Matrix<T, Rows, Cols>& A) {
  Matrix<std::complex<scalar_type_t<T>>, Rows, Cols> result;
  if constexpr (!is_complex_v<T>) {
    for (size_t i = 0; i < Rows; ++i)
      for (size_t j = 0; j < Cols; ++j) result[i][j] = std::complex<T>(A(i)(j));
  } else {
    result = A;
  }
  return result;
}

/**
 * @brief get the column of matrix A based on given index
 *
 * @param matrix A of size Rows x Cols of type T
 * @param index
 *
 * @return vector X
 */
template <typename T, size_t Rows, size_t Cols>
  requires(FieldLike<T>)
constexpr auto get_col(const Matrix<T, Rows, Cols>& A, const size_t& idx) {
  assert(idx < Cols);
  Vector<T, Rows> X;
  for (size_t i = 0; i < Rows; ++i) X[i] = A(i)(idx);
  return X;
}

/**
 * @brief get the row of matrix A based on given index
 *
 * @param matrix A of size Rows x Cols of type T
 * @param index
 *
 * @return vector X
 */
template <typename T, size_t Rows, size_t Cols>
  requires(FieldLike<T>)
constexpr auto get_row(const Matrix<T, Rows, Cols>& A, const size_t& idx) {
  assert(idx < Rows);
  return A(idx);
}

/**
 * @brief get the diagonal of a matrix A
 *
 * @param matrix A of size Rows x Cols of type T
 *
 * @return diagonal vector
 */
template <typename T, size_t Rows, size_t Cols>
  requires(FieldLike<T>)
constexpr auto get_diag(const Matrix<T, Rows, Cols>& A) {
  constexpr size_t mn = std::min(Rows, Cols);
  Vector<T, mn> diag;
  for (size_t i = 0; i < mn; ++i) diag[i] = A(i)(i);
  return diag;
}

/**
 * @brief make a diagonal matrix from a vector
 *
 * @param vector X of size N and type T
 *
 * @return diagonal matrix
 */
template <typename T, size_t N>
  requires(FieldLike<T>)
constexpr auto make_diag(const Vector<T, N>& X) {
  Matrix<T, N, N> diag;
  for (size_t i = 0; i < N; ++i) diag[i][i] = X(i);
  return diag;
}

/**
 * @brief get n rows from a matrix based on start & end indices provided as
 * template args
 *
 * @param matrix A of type T Rows x Cols elements
 *
 * @return matrix
 */
template <size_t Start, size_t End, typename T, size_t Rows, size_t Cols>
  requires(End - Start <= Rows && Start < Rows)
constexpr auto get_n_rows(const Matrix<T, Rows, Cols>& A) {
  Matrix<T, End - Start, Cols> result;
  for (size_t i = Start, idx = 0; i < End; ++i, ++idx)
    result[idx] = get_row(A, i);
  return result;
}

/**
 * @brief get n cols from a matrix based on start & end indices provided as
 * template args
 *
 * @param matrix A of type T Rows x Cols elements
 *
 * @return matrix
 */
template <size_t Start, size_t End, typename T, size_t Rows, size_t Cols>
  requires(End - Start <= Cols && Start < Cols)
constexpr auto get_n_cols(const Matrix<T, Rows, Cols>& A) {
  Matrix<T, End - Start, Rows> result;
  for (size_t i = Start, idx = 0; i < End; ++i, ++idx)
    result[idx] = get_col(A, i);
  return transpose(result);
}

/**
 * @brief get upper right triangular matrix from A
 *
 * @param matrix A of type T Rows x Cols elements
 *
 * @return matrix
 */
template <typename T, size_t Rows, size_t Cols>
  requires(FieldLike<T>)
constexpr auto get_upper_right_triangular_matrix(
    const Matrix<T, Rows, Cols>& A) {
  Matrix<T, Rows, Cols> R;
  for (size_t i = 0; i < Rows; ++i)
    for (size_t j = i; j < Cols; ++j) R[i][j] = A(i)(j);
  return R;
}

/**
 * @brief get lower left triangular matrix from A
 *
 * @param matrix A of type T Rows x Cols elements
 *
 * @return matrix
 */
template <typename T, size_t Rows, size_t Cols>
  requires(FieldLike<T>)
constexpr auto get_lower_left_triangular_matrix(
    const Matrix<T, Rows, Cols>& A) {
  Matrix<T, Rows, Cols> L;
  for (size_t i = 0; i < Cols; ++i)
    for (size_t j = i; j < Rows; ++j) L[j][i] = A(j)(i);
  return L;
}

}  // namespace algebra
