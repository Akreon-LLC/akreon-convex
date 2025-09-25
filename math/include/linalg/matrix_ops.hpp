/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "matrix.hpp"
#include "vector_ops.hpp"

namespace linalg {

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
template <MatrixLike MatA, MatrixLike MatB>
  requires(MatMulCompatible<MatA, MatB>)
constexpr auto outer_product(const MatA& A, const MatB& B) {
  using T = mat_type_t<MatA>;
  constexpr auto Rows = mat_rows_v<MatA>;
  constexpr auto Cols = mat_cols_v<MatB>;
  Matrix<T, Rows, Cols> result;
  for (size_t i = 0; i < Rows; ++i)
    for (size_t j = 0; j < Cols; ++j) result[i][j] = A(i)(0) * B(0)(j);
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
template <SquareMatrixLike Mat>
  requires(mat_rows_v<Mat> > 0)
constexpr auto trace(const Mat& A) {
  using T = mat_type_t<Mat>;
  constexpr auto Rows = mat_rows_v<Mat>;
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
template <MatrixLike Mat>
constexpr auto conjugate_transpose(const Mat& A)
  requires(is_complex_v<mat_type_t<Mat>>)
{
  constexpr auto Rows = mat_rows_v<Mat>;
  constexpr auto Cols = mat_cols_v<Mat>;
  auto A_t = transpose(A);
  for (size_t i = 0; i < Cols; ++i)
    for (size_t j = 0; j < Rows; ++j) A_t[i][j] = std::conj(A_t[i][j]);
  return A_t;
}

/**
 * @brief operator overload for matrix multiplication
 *
 * @param matrix A of type T Rows x Cols elements
 * @param matrix B of type T Cols x Rows elements
 *
 * @return A*B
 */
template <MatrixLike MatA, MatrixLike MatB>
  requires(MatMulCompatible<MatA, MatB>)
constexpr auto operator*(const MatA& A, const MatB& B) {
  using T = mat_type_t<MatA>;
  constexpr auto Rows = mat_rows_v<MatA>;
  constexpr auto Cols = mat_cols_v<MatB>;
  Matrix<T, Rows, Cols> result;
  const auto B_t = transpose(B);
  for (size_t i = 0; i < Rows; ++i)
    for (size_t j = 0; j < Cols; ++j) result[i][j] = dot(A(i), B_t(j));
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
template <MatrixLike Mat, VectorLike Vec>
  requires(MatVecCompatible<Mat, Vec>)
constexpr auto operator*(const Mat& A, const Vec& X) {
  using T = mat_type_t<Mat>;
  constexpr auto Rows = mat_rows_v<Mat>;
  Vector<T, Rows> result;
  for (size_t i = 0; i < Rows; ++i) result[i] = dot(A(i), X);
  return result;
}

}  // namespace linalg
