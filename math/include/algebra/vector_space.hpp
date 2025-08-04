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

  constexpr auto begin() const { return coordinates.begin(); }

  constexpr auto end() const { return coordinates.end(); }

  constexpr T operator()(const size_t& idx) const { return coordinates[idx]; }

  constexpr bool operator==(const Vector& other) const {
    return is_equal(other);
  }

  constexpr bool operator!=(const Vector& other) const {
    return !is_equal(other);
  }

  constexpr auto view() { return coordinates.view(); }

 private:
  constexpr bool is_equal(const Vector& other) const {
    if (s_size != other.s_size) return false;
    for (size_t i = 0; i < s_size; ++i)
      if (other.coordinates[i] != coordinates[i]) return false;
    return true;
  }
};

template <size_t N>
using Vectord = Vector<double, N>;

template <size_t N>
using Vectorf = Vector<float, N>;

template <size_t N>
using Vectorcd = Vector<std::complex<double>, N>;

template <size_t N>
using Vectorcf = Vector<std::complex<float>, N>;

template <typename T, size_t Rows, size_t Cols>
using Matrix = Vector<Vector<T, Rows>, Cols>;

template <size_t Rows, size_t Cols>
using Matrixd = Vector<Vectord<Rows>, Cols>;

template <size_t Rows, size_t Cols>
using Matrixf = Vector<Vectorf<Rows>, Cols>;

template <size_t Rows, size_t Cols>
using Matrixcd = Vector<Vectorcd<Rows>, Cols>;

template <size_t Rows, size_t Cols>
using Matrixcf = Vector<Vectorcf<Rows>, Cols>;

template <typename T>
struct number_of_rows {
  static constexpr size_t value = 0;
};

template <typename T, size_t N1, size_t N2>
struct number_of_rows<Vector<Vector<T, N1>, N2>> {
  static constexpr size_t value = N1;
};

template <typename T>
constexpr size_t number_of_rows_v =
    number_of_rows<std::remove_cvref_t<T>>::value;

template <typename T>
struct number_of_cols {
  static constexpr size_t value = 0;
};

template <typename T, size_t N1, size_t N2>
struct number_of_cols<Vector<Vector<T, N1>, N2>> {
  static constexpr size_t value = N2;
};

template <typename T>
constexpr size_t number_of_cols_v =
    number_of_cols<std::remove_cvref_t<T>>::value;

template <typename T, size_t N>
struct Identity {
  constexpr Vector<Vector<T, N>, N> operator()() const {
    Vector<Vector<T, N>, N> I;
    for (size_t i = 0; i < N; ++i) I.coordinates[i].coordinates[i] = 1;
    return I;
  }
};

template <typename T, size_t N1, size_t N2>
constexpr auto add(const Vector<T, N1>& A, const Vector<T, N2>& B) {
  static_assert(N1 == N2);
  static_assert(number_of_rows_v<decltype(A)> == number_of_rows_v<decltype(B)>);
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

template <typename T, size_t N>
constexpr auto operator-(const Vector<T, N>& A, const Vector<T, N>& B) {
  return A + (-1) * B;
}

template <typename T, size_t N1, size_t N2>
  requires(is_complex_v<T> || std::is_arithmetic_v<T>)
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
constexpr auto transpose(const Vector<T, N>& A) {
  constexpr size_t cols = number_of_rows_v<decltype(A)>;
  constexpr size_t rows = N;
  static_assert(cols > 0);
  using InnerT =
      std::conditional_t<is_complex_v<T>,
                         Vector<std::complex<scalar_type_t<T>>, rows>,
                         Vector<scalar_type_t<T>, rows>>;
  Vector<InnerT, cols> result;
  for (size_t i = 0; i < rows; ++i)
    for (size_t j = 0; j < cols; ++j)
      result.coordinates[j].coordinates[i] = A(i)(j);

  return result;
}

template <typename T, size_t N>
constexpr auto L2(const Vector<T, N>& X) {
  return constexpr_sqrt(dot(X, X));
}

template <typename T, size_t N>
constexpr auto euclidean_distance(const Vector<T, N>& X,
                                  const Vector<T, N>& Y) {
  return L2(X + -1 * Y);
}

template <typename T, size_t N>
constexpr auto L1(const Vector<T, N>& X) {
  T sum = 0;
  for (const auto& x : X) sum += constexpr_abs(x);
  return sum;
}

template <typename T, size_t N>
constexpr auto manhattan_distance(const Vector<T, N>& X,
                                  const Vector<T, N>& Y) {
  return L1(X + -1 * Y);
}

template <typename T, size_t N>
constexpr auto max_norm(const Vector<T, N>& X) {
  T mx = 0;
  for (const auto& x : X) mx = std::max(mx, constexpr_abs(x));
  return mx;
}

template <typename T, size_t N, typename Norm>
constexpr auto normalize(const Vector<T, N>& X, const Norm& norm) {
  const auto norm_val = norm(X);
  assert(norm_val > 0);
  return (1 / norm_val) * X;
}

template <typename T, size_t N>
constexpr auto trace(const Vector<T, N>& A) {
  static_assert(N > 0);
  static_assert(number_of_cols_v<decltype(A)> == number_of_rows_v<decltype(A)>);
  // (dakre) TODO determine a better way to generalize this
  using InnerT =
      std::conditional_t<is_complex_v<T>, std::complex<scalar_type_t<T>>,
                         scalar_type_t<T>>;
  InnerT sum = 0;
  for (size_t i = 0; i < N; ++i) sum += A(i)(i);
  return sum;
}

template <typename T, size_t N>
constexpr auto hermitian(const Vector<T, N>& A)
  requires(is_complex_v<T>)
{
  static_assert(N > 0);
  const auto n_cols = number_of_cols_v<decltype(A)>;
  const auto n_rows = number_of_rows_v<decltype(A)>;
  static_assert(n_cols == n_rows);
  auto AT = transpose(A);
  for (size_t i = 0; i < n_rows; ++i)
    for (size_t j = 0; j < n_cols; ++j)
      AT.coordinates[i].coordinates[j] =
          std::conj(AT.coordinates[i].coordinates[j]);
  return AT;
}

template <typename T, size_t N, size_t M>
constexpr auto operator*(const Vector<Vector<T, N>, M>& A,
                         const Vector<Vector<T, M>, N>& B) {
  Vector<Vector<T, N>, N> result;
  const auto BT = transpose(B);
  for (size_t i = 0; i < N; ++i)
    for (size_t j = 0; j < N; ++j)
      result.coordinates[i].coordinates[j] = dot(A(i), BT(j));
  return result;
}

template <typename T, size_t N, size_t M>
constexpr auto operator*(const Vector<Vector<T, N>, M>& A,
                         const Vector<T, M>& X) {
  Vector<T, N> result;
  for (size_t i = 0; i < N; ++i) result.coordinates[i] = dot(A(i), X);
  return result;
}

template <typename T, size_t N>
constexpr auto frobenius_norm(const Vector<T, N>& A) {
  static_assert(N > 0);
  static_assert(number_of_cols_v<decltype(A)> == number_of_rows_v<decltype(A)>);
  scalar_type_t<T> result = 0;
  if constexpr (!is_complex_v<T>) {
    result = trace(transpose(A) * A);
  } else {
    const auto tmp = trace(hermitian(A) * A);
    assert(is_real(tmp));
    result = tmp.real();
  }
  return constexpr_sqrt(result);
}

template <typename T, size_t N>
concept InnerProduct = requires(const scalar_type_t<Vector<T, N>>& alpha,
                                const scalar_type_t<Vector<T, N>>& beta,
                                const Vector<T, N>& X, const Vector<T, N>& Y) {
  { alpha * X } -> std::same_as<Vector<T, N>>;
  { alpha * (X + Y) == alpha * X + alpha * Y };
  { (alpha + beta) * X == alpha * X + beta * X };
  { (alpha * beta) * X == alpha * (beta * X) };
  { dot(X, Y) } -> std::same_as<scalar_type_t<T>>;
  { L2(X) >= 0.f };
  { L2(Vector<T, 1>{0.f}) == 0.f };
  { constexpr_abs(alpha) * L2(X) == L2(constexpr_abs(alpha) * X) };
  { constexpr_abs(dot(X, Y)) <= L2(X) + L2(Y) };  // cauchy-schwarz inequality
  { L2(X + Y) <= L2(X) + L2(Y) };                 // triangle inequality
};

}  // namespace algebra
