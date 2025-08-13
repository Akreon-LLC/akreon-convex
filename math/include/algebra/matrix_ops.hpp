/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "matrix.hpp"
#include "vector_ops.hpp"

namespace algebra {

/**
 * @brief outer product between two matrices A & B
 *
 * A * B = C where C is a new matrix of Rows_of_A x Cols_of_B
 *
 * @param matrix A of type T Rows1 x Cols1 elements
 * @param matrix B of type T Cols1 x Cols2 elements
 *
 * @return matrix
 */
template <typename T, size_t Rows1, size_t Cols1, size_t Cols2>
  requires(FieldLike<T>)
constexpr auto outer_product(const Matrix<T, Rows1, Cols1>& A,
                             const Matrix<T, Cols1, Cols2>& B) {
  using ST = std::conditional_t<is_complex_v<T>, std::complex<scalar_type_t<T>>,
                                scalar_type_t<T>>;
  Matrix<ST, Rows1, Cols2> result;
  for (size_t i = 0; i < Rows1; ++i)
    for (size_t j = 0; j < Cols2; ++j) result[i][j] = A(i)(0) * B(0)(j);
  return result;
}

/**
 * @brief transpose of a matrix
 *
 * A^T (rows and cols flipped from A)
 *
 * @param matrix A of type T Rows x Cols elements
 *
 * @return A^T
 */
template <typename T, size_t Rows, size_t Cols>
  requires(FieldLike<T>)
constexpr auto transpose(const Matrix<T, Rows, Cols>& A) {
  Matrix<T, Cols, Rows> result;
  for (size_t i = 0; i < Rows; ++i)
    for (size_t j = 0; j < Cols; ++j) result[j][i] = A(i)(j);

  return result;
}

/**
 * @brief trace of a matrix
 *
 * Sum of the diagonal elements of a matrix
 *
 * @param matrix A of type T Rows x Cols elements
 *
 * @return sum
 */
template <typename T, size_t Rows, size_t Cols>
  requires(FieldLike<T> && Rows == Cols && Rows > 0)
constexpr auto trace(const Matrix<T, Rows, Cols>& A) {
  T sum = 0;
  for (size_t i = 0; i < Rows; ++i) sum += A(i)(i);
  return sum;
}

/**
 * @brief conjugate transpose of a matrix
 *
 * Like A^T, however the field is complex and the elements get conjugated
 *
 * @param matrix A of type T Rows x Cols elements
 *
 * @return A^*
 */
template <typename T, size_t Rows, size_t Cols>
constexpr auto conjugate_transpose(const Matrix<T, Rows, Cols>& A)
  requires(is_complex_v<T>)
{
  auto AT = transpose(A);
  for (size_t i = 0; i < Cols; ++i)
    for (size_t j = 0; j < Rows; ++j) AT[i][j] = std::conj(AT[i][j]);
  return AT;
}

/**
 * @brief operator overload for matrix multiplication
 *
 * @param matrix A of type T Rows x Cols elements
 * @param matrix B of type T Cols x Rows elements
 *
 * @return A*B
 */
template <typename T, size_t Rows, size_t Cols>
  requires(FieldLike<T>)
constexpr auto operator*(const Matrix<T, Rows, Cols>& A,
                         const Matrix<T, Cols, Rows>& B) {
  Matrix<T, Rows, Rows> result;
  const auto BT = transpose(B);
  for (size_t i = 0; i < Rows; ++i)
    for (size_t j = 0; j < Rows; ++j) result[i][j] = dot(A(i), BT(j));
  return result;
}

/**
 * @brief operator overload for matrix multiplication between matrix and vector
 *
 * @param matrix A of type T Rows x Cols elements
 * @param vector X of type T Cols elements
 *
 * @return A*X
 */
template <typename T, size_t Rows, size_t Cols>
  requires(FieldLike<T>)
constexpr auto operator*(const Matrix<T, Rows, Cols>& A,
                         const Vector<T, Cols>& X) {
  Vector<T, Rows> result;
  for (size_t i = 0; i < Rows; ++i) result[i] = dot(A(i), X);
  return result;
}

}  // namespace algebra
