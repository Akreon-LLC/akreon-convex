/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include <tuple>

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
using Matrix = Vector<Vector<T, Cols>, Rows>;

template <size_t Rows, size_t Cols>
using Matrixd = Vector<Vectord<Cols>, Rows>;

template <size_t Rows, size_t Cols>
using Matrixf = Vector<Vectorf<Cols>, Rows>;

template <size_t Rows, size_t Cols>
using Matrixcd = Vector<Vectorcd<Rows>, Rows>;

template <size_t Rows, size_t Cols>
using Matrixcf = Vector<Vectorcf<Cols>, Rows>;

template <typename T>
struct number_of_rows {
  static constexpr size_t value = 0;
};

template <typename T, size_t Rows, size_t Cols>
struct number_of_rows<Matrix<T, Rows, Cols>> {
  static constexpr size_t value = Rows;
};

template <typename T>
constexpr size_t number_of_rows_v =
    number_of_rows<std::remove_cvref_t<T>>::value;

template <typename T>
struct number_of_cols {
  static constexpr size_t value = 0;
};

template <typename T, size_t Rows, size_t Cols>
struct number_of_cols<Matrix<T, Rows, Cols>> {
  static constexpr size_t value = Cols;
};

template <typename T>
constexpr size_t number_of_cols_v =
    number_of_cols<std::remove_cvref_t<T>>::value;

template <typename T, size_t N>
struct Identity {
  constexpr Matrix<T, N, N> operator()() const {
    Matrix<T, N, N> I;
    for (size_t i = 0; i < N; ++i) I.coordinates[i].coordinates[i] = 1;
    return I;
  }
};

template <typename T, size_t N>
  requires(!is_iterable_v<T>)
constexpr auto as_col_vector(const Vector<T, N>& X) {
  Matrix<T, N, 1> A;
  for (size_t i = 0; i < N; ++i) A.coordinates[i] = X(i);
  return A;
}

template <typename T, size_t N>
  requires(!is_iterable_v<T>)
constexpr auto as_row_vector(const Vector<T, N>& X) {
  Matrix<T, 1, N> A;
  for (size_t i = 0; i < N; ++i) A.coordinates[0].coordinates[i] = X(i);
  return A;
}

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
constexpr auto operator*(const auto& alpha, const Vector<T, N>& A) {
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

template <typename T, size_t Rows1, size_t Cols1, size_t Cols2>
  requires(is_complex_v<T> || std::is_arithmetic_v<T>)
constexpr auto outer_product(const Matrix<T, Rows1, Cols1>& A,
                             const Matrix<T, Cols1, Cols2>& B) {
  using ST = std::conditional_t<is_complex_v<T>, std::complex<scalar_type_t<T>>,
                                scalar_type_t<T>>;
  Matrix<ST, Rows1, Cols2> result;
  for (size_t i = 0; i < Rows1; ++i)
    for (size_t j = 0; j < Cols2; ++j)
      result.coordinates[i].coordinates[j] = A(i)(0) * B(0)(j);
  return result;
}

template <typename T, size_t N>
constexpr auto transpose(const Vector<T, N>& X) {
  Matrix<T, N, 1> XT;
  for (size_t i = 0; i < N; ++i) XT.coordinates[i].coordinates[0] = X(i);
  return XT;
}

template <typename T, size_t Rows, size_t Cols>
constexpr auto transpose(const Matrix<T, Rows, Cols>& A) {
  Matrix<T, Cols, Rows> result;
  for (size_t i = 0; i < Rows; ++i)
    for (size_t j = 0; j < Cols; ++j)
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

template <typename T, size_t Rows, size_t Cols>
  requires(Rows == Cols && Rows > 0)
constexpr auto trace(const Matrix<T, Rows, Cols>& A) {
  T sum = 0;
  for (size_t i = 0; i < Rows; ++i) sum += A(i)(i);
  return sum;
}

template <typename T, size_t Rows, size_t Cols>
constexpr auto hermitian(const Matrix<T, Rows, Cols>& A)
  requires(is_complex_v<T>)
{
  auto AT = transpose(A);
  for (size_t i = 0; i < Rows; ++i)
    for (size_t j = 0; j < Cols; ++j)
      AT.coordinates[i].coordinates[j] =
          std::conj(AT.coordinates[i].coordinates[j]);
  return AT;
}

template <typename T, size_t N>
  requires(is_complex_v<T> && !is_iterable_v<T>)
constexpr auto conjugate(const Vector<T, N>& X) {
  Vector<T, N> result;
  for (size_t i = 0; i < N; ++i) result.coordinates[i] = std::conj(X(i));
  return result;
}

template <typename T, size_t Rows, size_t Cols>
constexpr auto operator*(const Matrix<T, Rows, Cols>& A,
                         const Matrix<T, Cols, Rows>& B) {
  Matrix<T, Rows, Cols> result;
  const auto BT = transpose(B);
  for (size_t i = 0; i < Rows; ++i)
    for (size_t j = 0; j < Rows; ++j)
      result.coordinates[i].coordinates[j] = dot(A(i), BT(j));
  return result;
}

template <typename T, size_t Rows, size_t Cols>
constexpr auto operator*(const Matrix<T, Rows, Cols>& A,
                         const Vector<T, Cols>& X) {
  Vector<T, Rows> result;
  for (size_t i = 0; i < Rows; ++i) result.coordinates[i] = dot(A(i), X);
  return result;
}

template <typename T, size_t Rows, size_t Cols>
  requires(Rows == Cols && Rows > 0)
constexpr auto frobenius_norm(const Matrix<T, Rows, Cols>& A) {
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

template <typename T, size_t Rows, size_t Cols>
  requires(Rows >= Cols && Rows > 0 &&
           (is_complex_v<T> || std::is_arithmetic_v<T>))
constexpr auto qr(const Matrix<T, Rows, Cols>& A) {
  Matrix<T, Rows, Cols> Q;
  Matrix<T, Cols, Cols> R;
  for (size_t i = 0; i < Cols; ++i) {
    auto u = A(i);
    for (size_t j = 0; j < i; ++j) {
      R.coordinates[j].coordinates[i] = dot(Q(j), A(i));
      u = u - R(j)(i) * Q(j);
    }
    Q.coordinates[i] = normalize(u, L2<T, Cols>);
    R.coordinates[i].coordinates[i] = dot(u, Q(i));
  }
  return std::tuple(transpose(Q), R);
}

#if 0
// A x = b => Q R x = b => R x = Q b^T
template <typename T, size_t Rows, size_t Cols>
  requires(is_complex_v<T> || std::is_arithmetic_v<T>)
constexpr auto lsq(const Matrix<T, Rows, Cols>& Q,
                   const Matrix<T, Rows, Rows>& R, const Vector<T, Cols>& b) {
  Vector<T, Cols> x;
  const auto c = transpose(Q) * b;
  for (int i = Cols - 1; i >= 0; --i) {
    T sum = 0;
    for (int j = i + 1; j < Rows; ++j) sum += (R(i)(j) * x(j));
    x.coordinates[i] = (c(i) - sum) / R(i)(i);
  }

  return x;
}
#endif

}  // namespace algebra
