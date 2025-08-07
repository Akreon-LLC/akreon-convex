/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

namespace algebra {

template <typename T, size_t N>
struct Vector;

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
using Matrixcd = Vector<Vectorcd<Cols>, Rows>;

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

template <typename T, typename = void>
struct is_iterable : std::false_type {};

template <typename T>
struct is_iterable<T, std::void_t<decltype(std::begin(std::declval<T>())),
                                  decltype(std::end(std::declval<T>()))>>
    : std::true_type {};

template <typename T>
constexpr bool is_iterable_v = is_iterable<T>::value;

template <typename T>
constexpr bool is_real(const std::complex<T>& z) {
  return z.imag() == T(0);
}

template <typename T>
constexpr T constexpr_abs(const T& x) {
  return x < 0 ? -1 * x : x;
}

template <typename T>
constexpr T constexpr_sqrt(const T& x, T guess = 1.0) {
  // (dakre) newton step 0.5 * (guess + x / guess)
  return (guess * guess - x > 1e-10 || x - guess * guess > 1e-10)
             ? constexpr_sqrt(x, 0.5 * (guess + x / guess))
             : guess;
}

}  // namespace algebra
