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

#include "operator/operator_traits.hpp"
#include "vector/vector.hpp"

namespace linalg {

/**
 * @brief add two vectors together to get a third
 *
 * A + B = C
 *
 * @param vector A of type T with N elements
 * @param vector B of type T with N elements
 *
 * @return vector
 */
template <VectorLike Vec>
constexpr auto add(const Vec& A, const Vec& B) {
  // (dakre) below is safeguard against non-square matrix acting as vector with
  // duck type T as Vector<Vector<...>>
  using T = vec_type_t<Vec>;
  constexpr auto N = vec_size_v<Vec>;
  Vector<T, N> C;
  for (size_t i = 0; i < N; ++i) C(i) = A(i) + B(i);
  return C;
}

/**
 * @brief operator overload for add
 *
 * @see add
 */
template <VectorLike Vec>
constexpr auto operator+(const Vec& A, const Vec& B) {
  return add(A, B);
}

/**
 * @brief multiply a vector A with a scalar alpha
 *
 * alpha * A = alpha A
 *
 * @param alpha of scalar type T
 * @param vector x of type T with N elements
 *
 * @return vector
 */
template <VectorLike Vec, FieldLike Alpha>
constexpr auto scalar_multiplication(const Alpha& alpha, const Vec& A) {
  using T = vec_type_t<Vec>;
  constexpr auto N = vec_size_v<Vec>;
  Vector<T, N> C;
  for (size_t i = 0; i < N; ++i) C(i) = alpha * A(i);
  return C;
}

/**
 * @brief operator overload for scalar multiplication
 *
 * @see scalar_multiplication
 */
template <VectorLike Vec, FieldLike Alpha>
constexpr auto operator*(const Alpha& alpha, const Vec& A) {
  return scalar_multiplication(alpha, A);
}

/**
 * @brief operator overload for subtraction
 *
 * @see add
 * @see scalar_multiplication
 */
template <VectorLike Vec>
constexpr auto operator-(const Vec& A, const Vec& B) {
  using T = vec_type_t<Vec>;
  return A + scalar_type_t<T>(-1) * B;
}

/**
 * @brief dot product or inner product between two vectors A and B
 *
 * A^T A = scalar or dot(A, B) = scalar
 *
 * @param vector of type T with N elements
 * @param vector of type T with N elements
 *
 * @return scalar
 */
template <VectorLike Vec>
constexpr auto dot(const Vec& A, const Vec& B) {
  using T = vec_type_t<Vec>;
  constexpr auto N = vec_size_v<Vec>;
  T value = T(0);
  for (size_t i = 0; i < N; ++i) value += A(i) * B(i);
  return value;
}

/**
 * @brief tranpose of a vector
 *
 * Vector is naturally row vector in how it's stored so the transpose flips it
 * from a row to column based matrix
 *
 * @param vector of type T with N elements
 *
 * @return (N x 1) matrix
 */
template <VectorLike Vec>
constexpr auto transpose(const Vec& x) {
  using T = vec_type_t<Vec>;
  constexpr auto N = vec_size_v<Vec>;
  Matrix<T, N, 1> x_t;
  for (size_t i = 0; i < N; ++i) x_t(i)(0) = x(i);
  return x_t;
}

/**
 * @brief conjugate a vector
 *
 * @param vector of type T with N elements
 *
 * @return conjugated vector
 */
template <VectorLike Vec>
  requires(ComplexLike<vec_type_t<Vec>>)
constexpr auto conjugate(const Vec& x) {
  using T = vec_type_t<Vec>;
  constexpr auto N = vec_size_v<Vec>;
  Vector<T, N> result;
  for (size_t i = 0; i < N; ++i) result(i) = std::conj(x(i));
  return result;
}

/**
 * @brief cross product between x x y = Z
 *
 * @param vector of type T with 3 elements
 * @param vector of type T with 3 elements
 *
 * @return (3x1) vector
 */
template <VectorLike Vec>
  requires(vec_size_v<Vec> == 3)
constexpr auto cross(const Vec& x, const Vec& y) {
  return Vec{x(1) * y(2) - x(2) * y(1), -(x(0) * y(2) - x(2) * y(0)),
             x(0) * y(1) - x(1) * y(0)};
}

/**
 * @brief simple function that checks if a vector contains only zeros (hence
 * spans nullspace)
 *
 * @param vector of type T with N elements
 *
 * @return T/F
 */
template <VectorLike Vec>
  requires(vec_size_v<Vec> > 0)
constexpr bool all_zero(const Vec& x) {
  using T = vec_type_t<Vec>;
  constexpr auto N = vec_size_v<Vec>;
  int cnt = 0;
  for (size_t i = 0; i < N; ++i) cnt += x(i) == T(0);
  return cnt == N;
}

/**
 * @brief splice operator
 *
 * @param vector type T with N elements
 *
 * @return tuple
 */
template <size_t idx, VectorLike Vec>
  requires(vec_size_v<Vec> > 0 && idx < vec_size_v<Vec>)
constexpr auto splice(const Vec& x) {
  using T = vec_type_t<Vec>;
  constexpr auto N = vec_size_v<Vec>;
  Vector<T, idx> a;
  size_t i = 0;
  for (; i < idx; ++i) a(i) = x(i);
  Vector<T, N - idx> b;
  for (size_t j = 0; i < N; ++i, ++j) b(j) = x(i);
  return std::tuple(a, b);
}

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
    for (size_t j = 0; j < Cols; ++j) result(i)(j) = A(i)(0) * B(0)(j);
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
    for (size_t j = 0; j < Cols; ++j) result(j)(i) = A(i)(j);
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
  requires(ComplexLike<mat_type_t<Mat>>)
{
  constexpr auto Rows = mat_rows_v<Mat>;
  constexpr auto Cols = mat_cols_v<Mat>;
  auto A_t = transpose(A);
  for (size_t i = 0; i < Cols; ++i)
    for (size_t j = 0; j < Rows; ++j) A_t(i)(j) = std::conj(A_t(i)(j));
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
    for (size_t j = 0; j < Cols; ++j) result(i)(j) = dot(A(i), B_t(j));
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
  for (size_t i = 0; i < Rows; ++i) result(i) = dot(A(i), X);
  return result;
}

/**
 * @brief min element
 *
 * @param vector x
 *
 * @return min element
 */
template <VectorLike Vec>
  requires(vec_size_v<Vec> > 0)
constexpr auto min(const Vec& x) {
  using T = vec_type_t<Vec>;
  constexpr size_t N = vec_size_v<Vec>;
  T mn = x(0);
  for (size_t i = 1; i < N; ++i) mn = x(i) < mn ? x(i) : mn;
  return mn;
}

/**
 * @brief max element
 *
 * @param vector x
 *
 * @return max element
 */
template <VectorLike Vec>
  requires(vec_size_v<Vec> > 0)
constexpr auto max(const Vec& x) {
  using T = vec_type_t<Vec>;
  constexpr size_t N = vec_size_v<Vec>;
  T mx = x(0);
  for (size_t i = 1; i < N; ++i) mx = x(i) > mx ? x(i) : mx;
  return mx;
}

/**
 * @brief min element
 *
 * @param vector x
 *
 * @return min element
 */
template <VectorLike Vec>
  requires(vec_size_v<Vec> > 0)
constexpr size_t arg_min(const Vec& x) {
  using T = vec_type_t<Vec>;
  constexpr size_t N = vec_size_v<Vec>;
  T mn = x(0);
  size_t idx{0};
  for (size_t i = 1; i < N; ++i)
    if (x(i) < mn) mn = x(i), idx = i;
  return idx;
}

/**
 * @brief max element
 *
 * @param vector x
 *
 * @return max element
 */
template <VectorLike Vec>
  requires(vec_size_v<Vec> > 0)
constexpr size_t arg_max(const Vec& x) {
  using T = vec_type_t<Vec>;
  constexpr size_t N = vec_size_v<Vec>;
  T mx = x(0);
  size_t idx{0};
  for (size_t i = 1; i < N; ++i)
    if (x(i) > mx) mx = x(i), idx = i;
  return idx;
}

} // namespace linalg
