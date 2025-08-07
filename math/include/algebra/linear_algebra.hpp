/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include <assert.h>

#include <ranges>
#include <tuple>

#include "field.hpp"
#include "util.hpp"

namespace algebra {

template <typename T, size_t N>
struct Vector {
  using value_type = T;

  static_assert(FieldLike<T>);

  static constexpr size_t s_size = N;

  template <typename... Coords>
  constexpr Vector(Coords... coords)
    requires(sizeof...(coords) == N)
      : coordinates{coords...} {}

  constexpr Vector() : coordinates{} {}

  constexpr auto begin() const { return coordinates.begin(); }

  constexpr auto end() const { return coordinates.end(); }

  constexpr const T& operator()(const size_t& idx) const {
    assert(idx < N);
    return coordinates[idx];
  }

  constexpr T& operator[](const size_t& idx) {
    assert(idx < N);
    return coordinates[idx];
  }

  constexpr bool operator==(const Vector& other) const {
    return is_equal(other);
  }

  constexpr bool operator!=(const Vector& other) const {
    return !is_equal(other);
  }

  constexpr auto view() { return std::span<T, N>{coordinates}; }

  constexpr auto size() const { return N; }

 private:
  constexpr bool is_equal(const Vector& other) const {
    for (size_t i = 0; i < s_size; ++i)
      if (other.coordinates[i] != coordinates[i]) return false;
    return true;
  }

  std::array<T, N> coordinates;
};

template <typename T, size_t N>
constexpr Matrix<T, N, N> make_identity() {
  Matrix<T, N, N> I;
  for (size_t i = 0; i < N; ++i) I[i][i] = 1;
  return I;
}

template <typename T, size_t Rows, size_t Cols>
constexpr Matrix<T, Rows, Cols> make_zeros() {
  Matrix<T, Rows, Cols> Z;
  for (size_t i = 0; i < Rows; ++i)
    for (size_t j = 0; j < Cols; ++j) Z[i][j] = 0;
  return Z;
}

template <typename T, size_t N>
  requires(!is_iterable_v<T>)
constexpr auto as_col_vector(const Vector<T, N>& X) {
  Matrix<T, N, 1> A;
  for (size_t i = 0; i < N; ++i) A[i] = X(i);
  return A;
}

template <typename T, size_t N>
  requires(!is_iterable_v<T>)
constexpr auto as_row_vector(const Vector<T, N>& X) {
  Matrix<T, 1, N> A;
  for (size_t i = 0; i < N; ++i) A[0][i] = X(i);
  return A;
}

template <typename T, size_t Rows, size_t Cols>
  requires(FieldLike<T>)
constexpr auto get_col(const Matrix<T, Rows, Cols>& A, const size_t& idx) {
  assert(idx < Cols);
  Vector<T, Rows> X;
  for (size_t i = 0; i < Rows; ++i) X[i] = A(i)(idx);
  return X;
}

template <typename T, size_t Rows, size_t Cols>
  requires(FieldLike<T>)
constexpr auto get_row(const Matrix<T, Rows, Cols>& A, const size_t& idx) {
  assert(idx < Rows);
  return A(idx);
}

template <typename T, size_t Rows, size_t Cols>
  requires(FieldLike<T>)
constexpr auto get_diag(const Matrix<T, Rows, Cols>& A) {
  constexpr size_t mn = std::min(Rows, Cols);
  Vector<T, mn> diag;
  for (size_t i = 0; i < mn; ++i) diag[i] = A(i)(i);
  return diag;
}

template <typename T, size_t N>
  requires(FieldLike<T>)
constexpr auto make_diag(const Vector<T, N>& X) {
  Matrix<T, N, N> diag;
  for (size_t i = 0; i < N; ++i) diag[i][i] = X(i);
  return diag;
}

template <typename T, size_t N1, size_t N2>
constexpr auto add(const Vector<T, N1>& A, const Vector<T, N2>& B) {
  static_assert(N1 == N2);
  static_assert(number_of_rows_v<decltype(A)> == number_of_rows_v<decltype(B)>);
  Vector<T, N1> C;
  for (size_t i = 0; i < N1; ++i) C[i] = A(i) + B(i);
  return C;
}

template <typename T, size_t N1, size_t N2>
constexpr auto operator+(const Vector<T, N1>& A, const Vector<T, N2>& B) {
  return add(A, B);
}

template <typename T, size_t N>
constexpr auto scalar_multiplication(const auto& alpha, const Vector<T, N>& A) {
  Vector<T, N> C;
  for (size_t i = 0; i < N; ++i) C[i] = alpha * A(i);
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
    for (size_t j = 0; j < Cols2; ++j) result[i][j] = A(i)(0) * B(0)(j);
  return result;
}

template <typename T, size_t N>
constexpr auto transpose(const Vector<T, N>& X) {
  Matrix<T, N, 1> XT;
  for (size_t i = 0; i < N; ++i) XT[i][0] = X(i);
  return XT;
}

template <typename T, size_t Rows, size_t Cols>
constexpr auto transpose(const Matrix<T, Rows, Cols>& A) {
  Matrix<T, Cols, Rows> result;
  for (size_t i = 0; i < Rows; ++i)
    for (size_t j = 0; j < Cols; ++j) result[j][i] = A(i)(j);

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
    for (size_t j = 0; j < Cols; ++j) AT[i][j] = std::conj(AT[i][j]);
  return AT;
}

template <typename T, size_t N>
  requires(is_complex_v<T> && !is_iterable_v<T>)
constexpr auto conjugate(const Vector<T, N>& X) {
  Vector<T, N> result;
  for (size_t i = 0; i < N; ++i) result[i] = std::conj(X(i));
  return result;
}

template <typename T, size_t Rows, size_t Cols>
constexpr auto operator*(const Matrix<T, Rows, Cols>& A,
                         const Matrix<T, Cols, Rows>& B) {
  Matrix<T, Rows, Rows> result;
  const auto BT = transpose(B);
  for (size_t i = 0; i < Rows; ++i)
    for (size_t j = 0; j < Rows; ++j) result[i][j] = dot(A(i), BT(j));
  return result;
}

template <typename T, size_t Rows, size_t Cols>
constexpr auto operator*(const Matrix<T, Rows, Cols>& A,
                         const Vector<T, Cols>& X) {
  Vector<T, Rows> result;
  for (size_t i = 0; i < Rows; ++i) result[i] = dot(A(i), X);
  return result;
}

template <typename T, size_t Rows, size_t Cols>
  requires(Rows == Cols && Rows > 0 && FieldLike<T>)
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
    auto v = get_col(A, i);
    for (size_t j = 0; j < i; ++j) {
      R[j][i] = dot(get_col(Q, j), v);
      v = v - R[j][i] * get_col(Q, j);
    }
    const auto q_i = normalize(v, L2<T, Rows>);
    for (size_t j = 0; j < Rows; ++j) Q[j][i] = q_i(j);
    R[i][i] = dot(q_i, get_col(A, i));
  }
  return std::tuple(Q, R);
}

// (dakre) TODO below is gram-schmidt in the future update to be
// householder (improved accuracy)
//
// A x = b => Q R x = b => R x = Q b^T
template <typename T, size_t Rows, size_t Cols>
  requires(is_complex_v<T> || std::is_arithmetic_v<T>)
constexpr auto lsq(const Matrix<T, Rows, Cols>& Q,
                   const Matrix<T, Cols, Cols>& R, const Vector<T, Rows>& b) {
  Vector<T, Cols> x;
  const auto c = transpose(Q) * b;

  for (int i = static_cast<int>(Cols) - 1; i >= 0; --i) {
    T sum = 0;
    for (size_t j = i + 1; j < Cols; ++j) sum += (R(i)(j) * x(j));
    x[i] = (c(i) - sum) / R(i)(i);
  }

  return x;
}

// (dakre) TODO 
// 1) constexpr limits the amount of iterations may need to
// monitor performance over time
// 2) handle degenerate zeros case (edge)

// A = U \Sigma V^T
template <typename T, size_t Rows, size_t Cols>
constexpr auto svd(const Matrix<T, Rows, Cols>& A, size_t max_iter = 100,
                   double tol = 1e-10) {
  Matrix<T, Cols, Cols> M = transpose(A) * A;
  auto VT = make_identity<T, Cols>();

  for (size_t i = 0; i < max_iter; ++i) {
    const auto QR = qr(M);
    const auto Q = std::get<0>(QR);
    const auto R = std::get<1>(QR);
    M = R * Q;    // Apply similarity transform Mkp1 = Q^T Mk Q == R Q
    VT = VT * Q;  // Store right singular vectors (span the row space of M)
    const auto off_diag = M - make_diag(get_diag(M));
    if (frobenius_norm(off_diag) < tol) break;
  }

  const auto lambdas = get_diag(M);
  Vector<T, Cols> sigmas;
  for (size_t i = 0; i < Cols; ++i)
    sigmas[i] = constexpr_sqrt(std::max<T>(0, lambdas(i)));

  // \Sigma^{-1} A V = U
  auto U = make_identity<T, Rows>();
  const auto V = transpose(VT);
  constexpr auto mn = std::min(Rows, Cols);
  for (size_t i = 0; i < mn; ++i) U[i] = (1 / sigmas(i)) * A * get_col(V, i);
  U = transpose(U);

  U = std::get<0>(qr(U));
  VT = std::get<0>(qr(VT));

  return std::tuple(U, sigmas, VT);
}

}  // namespace algebra
