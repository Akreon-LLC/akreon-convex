/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "vector.hpp"

namespace algebra {

template <typename T, size_t N1, size_t N2>
  requires(FieldLike<T>)
constexpr auto add(const Vector<T, N1>& A, const Vector<T, N2>& B) {
  static_assert(N1 == N2);
  static_assert(number_of_rows_v<decltype(A)> == number_of_rows_v<decltype(B)>);
  Vector<T, N1> C;
  for (size_t i = 0; i < N1; ++i) C[i] = A(i) + B(i);
  return C;
}

template <typename T, size_t N1, size_t N2>
  requires(FieldLike<T>)
constexpr auto operator+(const Vector<T, N1>& A, const Vector<T, N2>& B) {
  return add(A, B);
}

template <typename T, size_t N>
  requires(FieldLike<T>)
constexpr auto scalar_multiplication(const auto& alpha, const Vector<T, N>& A) {
  Vector<T, N> C;
  for (size_t i = 0; i < N; ++i) C[i] = alpha * A(i);
  return C;
}

template <typename T, size_t N>
  requires(FieldLike<T>)
constexpr auto operator*(const auto& alpha, const Vector<T, N>& A) {
  return scalar_multiplication(alpha, A);
}

template <typename T, size_t N>
  requires(FieldLike<T>)
constexpr auto operator-(const Vector<T, N>& A, const Vector<T, N>& B) {
  return A + scalar_type_t<T>(-1) * B;
}

template <typename T, size_t N1, size_t N2>
  requires(FieldLike<T>)
constexpr auto dot(const Vector<T, N1>& A, const Vector<T, N2>& B) {
  static_assert(N1 == N2);
  static_assert(number_of_cols_v<decltype(A)> == number_of_rows_v<decltype(B)>);
  using ST = std::conditional_t<is_complex_v<T>, std::complex<scalar_type_t<T>>,
                                scalar_type_t<T>>;
  ST value = 0;
  for (size_t i = 0; i < N1; ++i) value += A(i) * B(i);
  return value;
}

template <typename T, size_t N>
  requires(FieldLike<T>)
constexpr auto transpose(const Vector<T, N>& X) {
  Matrix<T, N, 1> XT;
  for (size_t i = 0; i < N; ++i) XT[i][0] = X(i);
  return XT;
}

template <typename T, size_t N>
  requires(FieldLike<T>)
constexpr auto conjugate(const Vector<T, N>& X) {
  Vector<T, N> result;
  for (size_t i = 0; i < N; ++i) result[i] = std::conj(X(i));
  return result;
}

}  // namespace algebra
