/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "field.hpp"
#include "util.hpp"

namespace algebra {

template <typename T, size_t N>
struct Vector {
  using value_type = T;

  static_assert(FieldLike<T>, "Vector<T, N> requires T to satisfy FieldLike");

  static constexpr size_t s_size = N;

  std::array<T, N> coordinates;

  template <typename... Coords>
  constexpr Vector(Coords... coords)
    requires(sizeof...(coords) == N)
      : coordinates{coords...} {}

  constexpr Vector() : coordinates{} {}

  constexpr T operator()(const size_t& idx) const { return coordinates[idx]; }

  constexpr auto view() { return coordinates.view(); }
};

template <size_t N>
using Vectord = Vector<double, N>;

template <size_t N>
using Vectorf = Vector<float, N>;

template <size_t N>
using Vectorcd = Vector<std::complex<double>, N>;

template <size_t N>
using Vectorcf = Vector<std::complex<float>, N>;

template <size_t Rows, size_t Cols>
using Matrixd = Vector<Vectord<Cols>, Rows>;

template <size_t Rows, size_t Cols>
using Matrixf = Vector<Vectorf<Cols>, Rows>;

template <size_t Rows, size_t Cols>
using Matrixcd = Vector<Vectorcd<Cols>, Rows>;

template <size_t Rows, size_t Cols>
using Matrixcf = Vector<Vectorcf<Cols>, Rows>;

template <typename T>
struct number_of_cols {
  static constexpr size_t value = 0;
};

template <typename T, size_t N1, size_t N2>
struct number_of_cols<Vector<Vector<T, N1>, N2>> {
  static constexpr size_t value = N2;
};

template <typename T, size_t N1, size_t N2>
constexpr auto add(const Vector<T, N1>& A, const Vector<T, N2>& B) {
  static_assert(N1 == N2);
  static_assert(number_of_cols<decltype(A)>::value ==
                number_of_cols<decltype(B)>::value);
  Vector<T, N1> C;
  for (size_t i = 0; i < N1; ++i) C.coordinates[i] = A(i) + B(i);
  return C;
}

template <typename T, size_t N1, size_t N2>
constexpr auto operator+(const Vector<T, N1>& A, const Vector<T, N2>& B) {
  return add(A, B);
}

template <typename T, size_t N>
constexpr auto scalar_multiplication(const auto& alpha, const Vector<T, N>& A) {
  Vector<T, N> C;
  for (size_t i = 0; i < N; ++i) C.coordinates[i] = alpha * A(i);
  return C;
}

template <typename T, size_t N>
constexpr auto operator*(const scalar_type_t<T>& alpha, const Vector<T, N>& A) {
  return scalar_multiplication(alpha, A);
}

template <typename T, size_t N>
constexpr auto operator*(const Vector<T, N>& A, const scalar_type_t<T>& alpha) {
  return scalar_multiplication(alpha, A);
}

template <typename T, size_t N1, size_t N2>
constexpr auto dot(const Vector<T, N1>& A, const Vector<T, N2>& B) {
  static_assert(N1 == N2);
  static_assert(number_of_cols<decltype(A)>::value ==
                    number_of_cols<decltype(B)>::value &&
                number_of_cols<decltype(A)>::value == 0);
  scalar_type_t<T> value = 0;
  for (size_t i = 0; i < N1; ++i) value += A(i) * B(i);
  return value;
}

template <typename T>
constexpr auto L2(const T& x, const T& y) {
  static_assert(!is_iterable_v<T>);
  return constexpr_abs(x - y);
}

template <typename T, size_t N>
constexpr auto L2(const Vector<T, N>& X, const Vector<T, N>& Y) {
  return constexpr_sqrt(dot(X, Y));
}

template <typename T, size_t N>
concept InnerProduct = requires(const scalar_type_t<Vector<T, N>>& alpha,
                                const scalar_type_t<Vector<T, N>>& beta,
                                const Vector<T, N>& X, const Vector<T, N>& Y) {
  { alpha * X } -> std::same_as<Vector<T, N>>;
  { alpha * (X + Y) == alpha * X + alpha * Y };
  { (alpha + beta) * X == alpha * X + beta * X };
  { (alpha * beta) * X == alpha * (beta * X) };
  { L2(X, Y) } -> std::same_as<scalar_type_t<T>>;
};

}  // namespace algebra
