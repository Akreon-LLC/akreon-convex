/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "vector.hpp"

namespace algebra {

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
template <typename T, size_t N>
  requires(FieldLike<T>)
constexpr auto add(const Vector<T, N>& A, const Vector<T, N>& B) {
  // (dakre) below is safeguard against non-square matrix acting as vector with
  // duck type T as Vector<Vector<...>>
  static_assert(number_of_rows_v<decltype(A)> == number_of_rows_v<decltype(B)>);
  Vector<T, N> C;
  for (size_t i = 0; i < N; ++i) C[i] = A(i) + B(i);
  return C;
}

/**
 * @brief operator overload for add
 *
 * @see add
 */
template <typename T, size_t N1, size_t N2>
  requires(FieldLike<T>)
constexpr auto operator+(const Vector<T, N1>& A, const Vector<T, N2>& B) {
  return add(A, B);
}

/**
 * @brief multiply a vector A with a scalar alpha
 *
 * \alpha * A = \alpha A
 *
 * @param alpha of scalar type T
 * @param vector X of type T with N elements
 *
 * @return vector
 */
template <typename T, size_t N>
  requires(FieldLike<T>)
constexpr auto scalar_multiplication(const auto& alpha, const Vector<T, N>& A) {
  Vector<T, N> C;
  for (size_t i = 0; i < N; ++i) C[i] = alpha * A(i);
  return C;
}

/**
 * @brief operator overload for scalar multiplication
 *
 * @see scalar_multiplication
 */
template <typename T, size_t N>
  requires(FieldLike<T>)
constexpr auto operator*(const auto& alpha, const Vector<T, N>& A) {
  return scalar_multiplication(alpha, A);
}

/**
 * @brief operator overload for subtraction
 *
 * @see add
 * @see scalar_multiplication
 */
template <typename T, size_t N>
  requires(FieldLike<T>)
constexpr auto operator-(const Vector<T, N>& A, const Vector<T, N>& B) {
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
template <typename T, size_t N>
  requires(FieldLike<T>)
constexpr auto dot(const Vector<T, N>& A, const Vector<T, N>& B) {
  static_assert(number_of_cols_v<decltype(A)> == number_of_rows_v<decltype(B)>);
  using ST = std::conditional_t<is_complex_v<T>, std::complex<scalar_type_t<T>>,
                                scalar_type_t<T>>;
  ST value = ST(0);
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
template <typename T, size_t N>
  requires(FieldLike<T>)
constexpr auto transpose(const Vector<T, N>& X) {
  Matrix<T, N, 1> XT;
  for (size_t i = 0; i < N; ++i) XT[i][0] = X(i);
  return XT;
}

/**
 * @brief conjugate a vector
 *
 * @param vector of type T with N elements
 *
 * @return conjugated vector
 */
template <typename T, size_t N>
  requires(FieldLike<T> && is_complex_v<T>)
constexpr auto conjugate(const Vector<T, N>& X) {
  Vector<T, N> result;
  for (size_t i = 0; i < N; ++i) result[i] = std::conj(X(i));
  return result;
}

/**
 * @brief cross product between X x Y = Z
 *
 * @param vector of type T with 3 elements
 * @param vector of type T with 3 elements
 *
 * @return (3x1) vector
 */
template <typename T>
  requires(FieldLike<T>)
constexpr Vector<T, 3> cross(const Vector<T, 3>& X, const Vector<T, 3>& Y) {
  Vector<T, 3> result{X(1) * Y(2) - X(2) * Y(1), -(X(0) * Y(2) - X(2) * Y(0)),
                      X(0) * Y(1) - X(1) * Y(0)};
  return result;
}

}  // namespace algebra
