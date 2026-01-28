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

#include "algebra/solver_traits.hpp"
#include "norms/matrix_norms.hpp"
#include "norms/vector_norms.hpp"

namespace linalg {

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
  assert(norm_val > 0);
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
 * @brief back solve linear systems of equations
 *
 * R x = rhs
 *
 * @param upper right triangular R matrix
 * @param rhs vector to be solved
 *
 * @return solved x vector
 */
template <MatrixLike Mat, VectorLike Vec>
  requires(std::same_as<mat_type_t<Mat>, vec_type_t<Vec>> &&
           mat_rows_v<Mat> == vec_size_v<Vec>)
constexpr auto back_solve(const Mat& R, const Vec& rhs) {
  using T = mat_type_t<Mat>;
  constexpr size_t Rows = mat_rows_v<Mat>;
  constexpr size_t Cols = mat_cols_v<Mat>;

  Vector<T, Cols> x;
  constexpr auto mn = std::min(Rows, Cols);
  for (int i = static_cast<int>(mn) - 1; i >= 0; --i) {
    T sum = 0;
    for (size_t j = i + 1; j < Cols; ++j) sum += (R(i)(j) * x(j));

    assert(R(i)(i) != T(0));
    x[i] = (rhs(i) - sum) / R(i)(i);
  }
  return x;
}

/**
 * @brief forward solve linear system of equations
 *
 * R^T x = rhs ... R transposed is lower left instead of upper right
 *
 * @param lower left triangular matrix
 * @param rhs vector to be solved
 *
 * @return solved x vector
 */
template <MatrixLike Mat, VectorLike Vec>
  requires(std::same_as<mat_type_t<Mat>, vec_type_t<Vec>> &&
           mat_rows_v<Mat> == vec_size_v<Vec>)
constexpr auto forward_solve(const Mat& L, const Vec& rhs) {
  using T = mat_type_t<Mat>;
  constexpr size_t Rows = mat_rows_v<Mat>;
  constexpr size_t Cols = mat_cols_v<Mat>;

  Vector<T, Cols> x;
  constexpr auto mn = std::min(Rows, Cols);
  for (int i = 0; i < static_cast<int>(mn); ++i) {
    T sum = 0;
    for (int j = 0; j < i; ++j) sum += (L(i)(j) * x(j));

    assert(L(i)(i) != T(0));
    x[i] = (rhs(i) - sum) / L(i)(i);
  }
  return x;
}

/**
 * @brief minimum norm solution for underdetermined system m < n for A
 *
 * min x s.t. ||A x - y||
 *
 * @param Q \in R^{n x n} (orthogonal basis from QR)
 * @param R \in R^{n x m} (upper right triangular from QR)
 *
 * @return x_min
 */
template <MatrixLike MatQ, MatrixLike MatR, VectorLike Vec>
  requires(MatMulCompatible<MatQ, MatR> && MatVecCompatible<MatR, Vec>)
constexpr auto min_norm(const MatQ& Q, const MatR& R, const Vec& b) {
  using T = mat_type_t<MatQ>;
  if constexpr (RealLike<T>) {
    const auto y = forward_solve(transpose(R), b);
    return Q * y;
  } else {
    const auto y = forward_solve(conjugate_transpose(R), b);
    return Q * y;
  }
}

}  // namespace linalg
