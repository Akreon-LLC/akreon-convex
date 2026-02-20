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

#include "config/solver_traits.hpp"
#include "norms/matrix_norms.hpp"
#include "norms/vector_norms.hpp"
#include "operator/operator_traits.hpp"

namespace linalg {

/**
 * @brief get n elements based on start to end indices
 *
 * @param vector of type T with N elements
 *
 * @return vector of type T with End - Start elements
 */
template <size_t Start, size_t End, VectorLike Vec>
  requires(End - Start < vec_size_v<Vec> && Start < vec_size_v<Vec>)
constexpr auto get_elements(const Vec& x) {
  using T = vec_type_t<Vec>;
  Vector<T, End - Start> result;
  for (size_t i = Start, idx = 0; i < End; ++i, ++idx) result(idx) = x(i);
  return result;
}

/**
 * @brief create an identity matrix of size N type T
 *
 * @return I
 */
template <typename T, size_t N>
constexpr Matrix<T, N, N> make_identity() {
  Matrix<T, N, N> I;
  for (size_t i = 0; i < N; ++i) I(i)(i) = T(1);
  return I;
}

/**
 * @brief create a matrix filled with zeros of size Rows x Cols of type T
 *
 * @return Zeros
 */
template <typename T, size_t Rows, size_t Cols>
constexpr Matrix<T, Rows, Cols> make_zeros() {
  Matrix<T, Rows, Cols> Z;
  for (size_t i = 0; i < Rows; ++i)
    for (size_t j = 0; j < Cols; ++j) Z(i)(j) = T(0);
  return Z;
}

/**
 * @brief express the vector X as a column vector (N x 1)
 *
 * @param vector X of type T and N elements
 *
 * @return (N x 1) matrix
 */
template <VectorLike Vec>
constexpr auto as_col_vector(const Vec& X) {
  using T = vec_type_t<Vec>;
  constexpr auto N = vec_size_v<Vec>;
  Matrix<T, N, 1> A;
  for (size_t i = 0; i < N; ++i) A(i) = X(i);
  return A;
}

/**
 * @brief express the vector X as a row vector (1 x N)
 *
 * @param vector X of type T and N elements
 *
 * @return (1 x N) matrix
 */
template <VectorLike Vec>
constexpr auto as_row_vector(const Vec& X) {
  using T = vec_type_t<Vec>;
  constexpr auto N = vec_size_v<Vec>;
  Matrix<T, 1, N> A;
  for (size_t i = 0; i < N; ++i) A(0)(i) = X(i);
  return A;
}

/**
 * @brief express matrix as complex one
 *
 * @param Matrix A type T with Rows x Cols elements
 *
 * @return A \in C
 */
template <MatrixLike Mat>
constexpr auto as_complex_matrix(const Mat& A) {
  using T = mat_type_t<Mat>;
  constexpr auto Rows = mat_rows_v<Mat>;
  constexpr auto Cols = mat_cols_v<Mat>;
  Matrix<std::complex<scalar_type_t<T>>, Rows, Cols> result;
  if constexpr (RealLike<T>) {
    for (size_t i = 0; i < Rows; ++i)
      for (size_t j = 0; j < Cols; ++j) result(i)(j) = std::complex<T>(A(i)(j));
  } else {
    result = A;
  }
  return result;
}

/**
 * @brief express col vector as vector
 *
 * @param Matrix A type T with N x 1 elements
 *
 * @return vector
 */
template <typename T, size_t N>
  requires(FieldLike<T>)
constexpr auto as_vector(const Matrix<T, N, 1>& A) {
  Vector<T, N> res;
  for (size_t i = 0; i < N; ++i) res(i) = A(i)(0);
  return res;
}

/**
 * @brief express row vector as vector
 *
 * @param Matrix A type T with 1 x N elements
 *
 * @return vector
 */
template <typename T, size_t N>
  requires(FieldLike<T>)
constexpr auto as_vector(const Matrix<T, 1, N>& A) {
  Vector<T, N> res;
  for (size_t i = 0; i < N; ++i) res(i) = A(0)(i);
  return res;
}

/**
 * @brief get the column of matrix A based on given index
 *
 * @param matrix A of size Rows x Cols of type T
 * @param index
 *
 * @return vector X
 */
template <MatrixLike Mat>
constexpr auto get_col(const Mat& A, const size_t& idx) {
  using T = mat_type_t<Mat>;
  constexpr auto Rows = mat_rows_v<Mat>;
  assert(idx < mat_cols_v<Mat> && "get_col index out of bounds");
  Vector<T, Rows> X;
  for (size_t i = 0; i < Rows; ++i) X(i) = A(i)(idx);
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
template <MatrixLike Mat>
constexpr auto get_row(const Mat& A, const size_t& idx) {
  assert(idx < mat_rows_v<Mat> && "get_row index out of bounds");
  return A(idx);
}

/**
 * @brief get the diagonal of a matrix A
 *
 * @param matrix A of size Rows x Cols of type T
 *
 * @return diagonal vector
 */
template <MatrixLike Mat>
constexpr auto get_diag(const Mat& A) {
  using T = mat_type_t<Mat>;
  constexpr auto Rows = mat_rows_v<Mat>;
  constexpr auto Cols = mat_cols_v<Mat>;
  constexpr size_t mn = std::min(Rows, Cols);
  Vector<T, mn> diag;
  for (size_t i = 0; i < mn; ++i) diag(i) = A(i)(i);
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
  for (size_t i = 0; i < N; ++i) diag(i)(i) = X(i);
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
template <size_t Start, size_t End, MatrixLike Mat>
  requires(End - Start <= mat_rows_v<Mat> && Start < mat_rows_v<Mat>)
constexpr auto get_n_rows(const Mat& A) {
  using T = mat_type_t<Mat>;
  constexpr auto Cols = mat_cols_v<Mat>;
  Matrix<T, End - Start, Cols> result;
  for (size_t i = Start, idx = 0; i < End; ++i, ++idx)
    result(idx) = get_row(A, i);
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
template <size_t Start, size_t End, MatrixLike Mat>
  requires(End - Start <= mat_cols_v<Mat> && Start < mat_cols_v<Mat>)
constexpr auto get_n_cols(const Mat& A) {
  using T = mat_type_t<Mat>;
  constexpr auto Rows = mat_rows_v<Mat>;
  Matrix<T, End - Start, Rows> result;
  for (size_t i = Start, idx = 0; i < End; ++i, ++idx)
    result(idx) = get_col(A, i);
  return transpose(result);
}

/**
 * @brief get block from matrix
 *
 * @param matrix A of type T Rows x Cols elements
 *
 * @return matrix
 */
template <size_t StartRow, size_t EndRow, size_t StartCol, size_t EndCol,
          MatrixLike Mat>
  requires(EndRow - StartRow <= mat_rows_v<Mat> && StartRow < mat_rows_v<Mat> &&
           EndCol - StartCol <= mat_cols_v<Mat> && StartCol < mat_cols_v<Mat>)
constexpr auto get_block(const Mat& A) {
  using T = mat_type_t<Mat>;
  Matrix<T, EndRow - StartRow, EndCol - StartCol> result;
  for (size_t i = StartRow, a = 0; i < EndRow; ++i, ++a)
    for (size_t j = StartCol, b = 0; j < EndCol; ++j, ++b)
      result(a)(b) = A(i)(j);
  return result;
}

/**
 * @brief set block of matrix from matrix
 *
 * @param matrix A of type T Rows x Cols elements
 * @param matrix B of type T (EndRow - StartRow) x (EndCol - StartCol)
 *
 * @return matrix A
 */
template <size_t StartRow, size_t EndRow, size_t StartCol, size_t EndCol,
          MatrixLike Mat>
  requires(EndRow - StartRow <= mat_rows_v<Mat> && StartRow < mat_rows_v<Mat> &&
           EndCol - StartCol <= mat_cols_v<Mat> && StartCol < mat_cols_v<Mat>)
constexpr auto set_block(
    Mat A,
    const Matrix<mat_type_t<Mat>, EndRow - StartRow, EndCol - StartCol>& B) {
  for (size_t i = StartRow, a = 0; i < EndRow; ++i, ++a)
    for (size_t j = StartCol, b = 0; j < EndCol; ++j, ++b) A(i)(j) = B(a)(b);
  return A;
}

/**
 * @brief get upper right triangular matrix from A
 *
 * @param matrix A of type T Rows x Cols elements
 *
 * @return matrix
 */
template <MatrixLike Mat>
constexpr auto get_upper_right_triangular_matrix(const Mat& A) {
  constexpr auto Rows = mat_rows_v<Mat>;
  constexpr auto Cols = mat_cols_v<Mat>;
  Mat R;
  for (size_t i = 0; i < Rows; ++i)
    for (size_t j = i; j < Cols; ++j) R(i)(j) = A(i)(j);
  return R;
}

/**
 * @brief get lower left triangular matrix from A
 *
 * @param matrix A of type T Rows x Cols elements
 *
 * @return matrix
 */
template <MatrixLike Mat>
constexpr auto get_lower_left_triangular_matrix(const Mat& A) {
  constexpr auto Rows = mat_rows_v<Mat>;
  constexpr auto Cols = mat_cols_v<Mat>;
  Mat L;
  for (size_t i = 0; i < Cols; ++i)
    for (size_t j = i; j < Rows; ++j) L(j)(i) = A(j)(i);
  return L;
}

/**
 * @brief Normalize a vector given a norm function
 *
 * @param vector of type T with N elements
 * @param reference to a Norm object (L1, L2, max_norm, etc.)
 *
 * @return normalized vector
 */
template <VectorLike Vec, typename Norm>
constexpr auto normalize(const Vec& X, const Norm& norm) {
  const auto norm_val = norm(X);
  assert(norm_val > 0 && "normalize requires strictly positive norm");
  return (1 / norm_val) * X;
}

/**
 * @brief Compute the euclidean distance between two vectors (L2 norm of delta)
 *
 * @param vector X of type T with N elements
 * @param vector Y of type T with N elements
 *
 * @return distance between two vectors as a scalar
 */
template <VectorLike Vec>
constexpr auto euclidean_distance(const Vec& X, const Vec& Y) {
  return L2<Vec>()(X - Y);
}

/**
 * @brief Compute the manhattan distance between two vectors (L1 norm of delta)
 *
 * @param vector X of type T with N elements
 * @param vector Y of type T with N elements
 *
 * @return distance between two vectors as a scalar
 */
template <VectorLike Vec>
constexpr auto manhattan_distance(const Vec& X, const Vec& Y) {
  return L1<Vec>()(X - Y);
}

/**
 * @brief checks if two vectors are equivalent to a given tolerance (different
 * than == operator on vector which must be exactly equal
 *
 * @param vector X of type T N elements
 * @param vector Y of type T N elements
 *
 * @return T/F
 */
template <typename T, size_t N>
  requires(FieldLike<T>)
constexpr bool all_equal(const Vector<T, N>& X, const Vector<T, N>& Y,
                         scalar_type_t<T> tol = 1e-10) {
  return euclidean_distance(X, Y) <= tol;
}

/**
 * @brief checks if two given matrix are equivalent
 *
 * @param matrix A of type T with Rows x Cols elements
 * @param vector B of type T with Rows x Cols elements
 * @param tolerance set to 1e-10 by default
 *
 * @return T/F
 */
template <typename T, size_t Rows, size_t Cols>
  requires(FieldLike<T>)
constexpr bool all_equal(const Matrix<T, Rows, Cols>& A,
                         const Matrix<T, Rows, Cols>& B,
                         scalar_type_t<T> tol = 1e-10) {
  bool rv = true;
  for (size_t i = 0; i < Rows; ++i) rv &= all_equal(A(i), B(i), tol);
  return rv;
}

/**
 * @brief checks if the row space of one matrix is equivalent to the column
 * space of another matrix
 *
 * @param matrix A of type T with Rows x Cols elements
 * @param vector B of type T with Cols x Rows elements
 *
 * @return T/F
 */
template <typename T, size_t Rows, size_t Cols>
  requires(FieldLike<T>)
constexpr bool rows_equal_cols(const Matrix<T, Rows, Cols>& A,
                               const Matrix<T, Cols, Rows>& B) {
  bool rv = true;
  for (size_t i = 0; i < Rows; ++i)
    rv &= euclidean_distance(A(i), get_col(B, i)) <= 1e-10;
  return rv;
}

/**
 * @brief make symmetric matrix
 *
 * @param symmetric vector (flattened)
 *
 * @return matrix
 */
template <size_t N, VectorLike Vec>
  requires(vec_size_v<Vec> == N * (N + 1) / 2)
constexpr auto make_smat(const Vec& L) {
  using T = vec_type_t<Vec>;
  Matrix<T, N, N> S;
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      const auto idx = i * (i + 1) / 2 + j;
      S(i)(j) = L(idx);
    }
  }
  const auto S_T = transpose(S);
  Matrix<T, N, N> res;
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      if (i >= j)
        res(i)(j) = S(i)(j);
      else
        res(i)(j) = S_T(i)(j);
    }
  }
  return res;
}

/**
 * @brief get lower triangle with diag
 *
 * [ * 0 0 0 ]
 * [ * * 0 0 ]
 * [ * * * 0 ]
 * [ * * * * ]
 *
 * @param matrix A
 *
 * @return vec
 */
template <SquareMatrixLike Mat>
constexpr auto get_lower_triangle_with_diag(const Mat& A) {
  using T = mat_type_t<Mat>;
  static constexpr size_t N = mat_rows_v<Mat>;
  static constexpr size_t K = N * (N + 1) / 2;
  Vector<T, K> vec{};
  for (size_t i = 0, k = 0; i < N; ++i)
    for (size_t j = 0; j <= i; ++j, ++k) vec(k) = A(i)(j);
  return vec;
}

/**
 * @brief get lower triangle
 *
 * [ 0 0 0 0 ]
 * [ * 0 0 0 ]
 * [ * * 0 0 ]
 * [ * * * 0 ]
 *
 * @param matrix A
 *
 * @return vec
 */
template <SquareMatrixLike Mat>
constexpr auto get_lower_triangle(const Mat& A) {
  using T = mat_type_t<Mat>;
  static constexpr size_t N = mat_rows_v<Mat>;
  static constexpr size_t K = N * (N + 1) / 2 - N;
  Vector<T, K> vec{};
  for (size_t i = 0, k = 0; i < N; ++i)
    for (size_t j = 0; j < i; ++j, ++k) vec(k) = A(i)(j);
  return vec;
}

/**
 * @brief make lower triangle
 *
 * @param vector x
 *
 * @return matrix L
 */
template <SquareMatrixLike Mat, VectorLike Vec>
  requires(std::same_as<mat_type_t<Mat>, vec_type_t<Vec>>)
constexpr auto make_unit_lower_triangle(const Vec& x) {
  using T = mat_type_t<Mat>;
  static constexpr size_t N = mat_rows_v<Mat>;
  static_assert(N * (N + 1) / 2 - N == vec_size_v<Vec>,
                "Invalid vector size to make lower triangle of square matrix");
  Mat L{};
  for (size_t i = 0, k = 0; i < N; ++i) {
    size_t j = 0;
    for (; j < i; ++j, ++k) L(i)(j) = x(k);
    if (i == j) L(i)(j) = T(1);
  }

  return L;
}

}  // namespace linalg
