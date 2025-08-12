/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "matrix.hpp"
#include "vector_ops.hpp"

namespace algebra {

template <typename T, size_t Rows1, size_t Cols1, size_t Cols2>
  requires(is_complex_v<T> || std::is_arithmetic_v<T>)
constexpr auto outer_product(const Matrix<T, Rows1, Cols1>& A,
                             const Matrix<T, Cols1, Cols2>& B) {
  using ST = std::conditional_t<is_complex_v<T>, std::complex<scalar_type_t<T>>,
                                scalar_type_t<T>>;
  Matrix<ST, Rows1, Cols2> result;
  for (size_t i = 0; i < Rows1; ++i)
    for (size_t j = 0; j < Cols2; ++j) result[i][j] = A(i)(0) * B(0)(j);
  return result;
}

template <typename T, size_t Rows, size_t Cols>
  requires(FieldLike<T>)
constexpr auto transpose(const Matrix<T, Rows, Cols>& A) {
  Matrix<T, Cols, Rows> result;
  for (size_t i = 0; i < Rows; ++i)
    for (size_t j = 0; j < Cols; ++j) result[j][i] = A(i)(j);

  return result;
}

template <typename T, size_t Rows, size_t Cols>
  requires(FieldLike<T> && Rows == Cols && Rows > 0)
constexpr auto trace(const Matrix<T, Rows, Cols>& A) {
  T sum = 0;
  for (size_t i = 0; i < Rows; ++i) sum += A(i)(i);
  return sum;
}

template <typename T, size_t Rows, size_t Cols>
constexpr auto conjugate_transpose(const Matrix<T, Rows, Cols>& A)
  requires(is_complex_v<T>)
{
  auto AT = transpose(A);
  for (size_t i = 0; i < Rows; ++i)
    for (size_t j = 0; j < Cols; ++j) AT[i][j] = std::conj(AT[i][j]);
  return AT;
}

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

template <typename T, size_t Rows, size_t Cols>
  requires(FieldLike<T>)
constexpr auto operator*(const Matrix<T, Rows, Cols>& A,
                         const Vector<T, Cols>& X) {
  Vector<T, Rows> result;
  for (size_t i = 0; i < Rows; ++i) result[i] = dot(A(i), X);
  return result;
}

}  // namespace algebra
