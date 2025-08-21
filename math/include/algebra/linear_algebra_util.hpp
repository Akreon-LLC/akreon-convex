/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "field.hpp"

/// @brief macro for capping compile step eval (default max is ~512)
#define MAX_STEP_EVAL 512

namespace algebra {

/** \struct Vector
 *  \brief Structure that deals with vector types \in vector spaces
 *  \details See vector struct definition for more details
 *  \ingroup algebra
 */
template <typename T, size_t N>
struct Vector;

/// @brief alias for deducing vector of doubles with N elements
template <size_t N>
using Vectord = Vector<double, N>;

/// @brief alias for deducing vector of floats with N elements
template <size_t N>
using Vectorf = Vector<float, N>;

/// @brief alias for deducing vector of complex doubles with N elements
template <size_t N>
using Vectorcd = Vector<std::complex<double>, N>;

/// @brief alias for deducing vector of complex floats with N elements
template <size_t N>
using Vectorcf = Vector<std::complex<float>, N>;

/// @brief alias for deducing matrix as vector of vectors of type T with Rows x
/// Cols elements
template <typename T, size_t Rows, size_t Cols>
using Matrix = Vector<Vector<T, Cols>, Rows>;

/// @brief alias for deducing matrix as vector of vectors of type double with
/// Rows x Cols elements
template <size_t Rows, size_t Cols>
using Matrixd = Vector<Vectord<Cols>, Rows>;

/// @brief alias for deducing matrix as vector of vectors of type float with
/// Rows x Cols elements
template <size_t Rows, size_t Cols>
using Matrixf = Vector<Vectorf<Cols>, Rows>;

/// @brief alias for deducing matrix as vector of vectors of type complex double
/// with Rows x Cols elements
template <size_t Rows, size_t Cols>
using Matrixcd = Vector<Vectorcd<Cols>, Rows>;

/// @brief alias for deducing matrix as vector of vectors of type complex float
/// with Rows x Cols elements
template <size_t Rows, size_t Cols>
using Matrixcf = Vector<Vectorcf<Cols>, Rows>;

/** \struct number_of_rows
 *  \brief Structure that stores the number of rows of a matrix type (default 0)
 *  \details base case for meta-templating type traits for number_of_rows based
 * on template deduction
 *  \ingroup algebra
 */
template <typename T>
struct number_of_rows {
  static constexpr size_t value = 0;
};

/// @brief value = Rows if template argument equals proper Matrix type
template <typename T, size_t Rows, size_t Cols>
struct number_of_rows<Matrix<T, Rows, Cols>> {
  static constexpr size_t value = Rows;
};

/// @brief number_of_rows_v is a variable holding the number of rows a matrix
/// type has
template <typename T>
constexpr size_t number_of_rows_v =
    number_of_rows<std::remove_cvref_t<T>>::value;

/** \struct number_of_cols
 *  \brief Structure that stores the number of cols of a matrix type (default 0)
 *  \details base case for meta-templating type traits for number_of_cols based
 * on template deduction
 *  \ingroup algebra
 */
template <typename T>
struct number_of_cols {
  static constexpr size_t value = 0;
};

/// @brief value = Cols if template argument equals proper Matrix type
template <typename T, size_t Rows, size_t Cols>
struct number_of_cols<Matrix<T, Rows, Cols>> {
  static constexpr size_t value = Cols;
};

/// @brief number_of_cols_v is a variable holding the number of cols a matrix
/// type has
template <typename T>
constexpr size_t number_of_cols_v =
    number_of_cols<std::remove_cvref_t<T>>::value;

/** \struct is_iterable
 *  \brief Structure that deals in type_trait deduction for checking if type is
 * iterable (false initially)
 *  \details scalar_type is an overloaded type that strips away meta-templates
 * to see if the type is iterable
 *  \ingroup algebra
 */
template <typename T, typename = void>
struct is_iterable : std::false_type {};

/// @brief True if type is iterable satisfying begin / end functions
template <typename T>
struct is_iterable<T, std::void_t<decltype(std::begin(std::declval<T>())),
                                  decltype(std::end(std::declval<T>()))>>
    : std::true_type {};

/// @brief is_iterable_v is a variable holding the value of iterable result
template <typename T>
constexpr bool is_iterable_v = is_iterable<T>::value;

/**
 * @brief Checks if the complex variable is imaginary to a tolerance
 *
 * @param complex variable of type T
 * @param tolerance satisfying the imaginary value check
 *
 * @return T/F
 */
template <typename T>
constexpr bool is_real(const std::complex<T>& z, scalar_type_t<T> tol = 1e-10) {
  return z.imag() <= tol;
}

/**
 * @brief perform absolute value computation
 *
 * @param some type T variable x
 *
 * @return |x|
 */
template <typename T>
  requires(!is_complex_v<T>)
constexpr T constexpr_abs(const T& x) {
  return x < 0 ? -1 * x : x;
}

/**
 * @brief perform square root of input variable x via root finding
 *
 * @param some type T variable x
 * @param initial guess of type T
 * @param depth size indicating recursion stack depth
 *
 * @return sqrt(x)
 */
template <typename T>
  requires(FieldLike<T> && !is_complex_v<T>)
constexpr T constexpr_sqrt(const T& x, T guess = 1e-10, int depth = 0) {
  // (dakre) newton step 0.5 * (guess + x / guess)
  if (depth + 1 < MAX_STEP_EVAL && constexpr_abs(guess * guess - x) > 1e-10)
    return constexpr_sqrt(x, T(0.5) * (guess + x / guess), depth + 1);
  return guess;
}

/**
 * @brief perform sin via taylor series approximation
 *
 * @param x as type T
 *
 * @return cosine approx of x
 */
template <typename T>
  requires(FieldLike<T> && !is_complex_v<T>)
constexpr auto constexpr_sin(const T& x, int terms = 12) {
  T sum = x, term = x;
  for (int k = 1; k < terms; ++k) {
    term *= (-(x * x)) / (T(2 * k - 1) * T(2 * k));
    sum += term;
  }
  return sum;
}

/**
 * @brief perform sin via taylor series approximation
 *
 * @param x as type T
 *
 * @return cosine approx of x
 */
template <typename T>
  requires(FieldLike<T> && !is_complex_v<T>)
constexpr auto constexpr_cos(const T& x, int terms = 12) {
  T sum = T(1), term = T(1);
  for (int k = 1; k < terms; ++k) {
    term *= (-(x * x)) / (T(2 * k - 1) * T(2 * k));
    sum += term;
  }
  return sum;
}

/**
 * @brief perform exp on an imaginary number (unit circle)
 *
 * @param theta \in R (e^{i \theta})
 *
 * @return complex number
 */
template <typename T>
  requires(FieldLike<T> && !is_complex_v<T>)
constexpr auto constexpr_expi(const T& theta, int terms = 12) {
  return std::complex<T>(constexpr_cos(theta, terms),
                         constexpr_sin(theta, terms));
}

/*
 * @brief constexpr for loop which recursively unrolls a dynamic for loop into
 * compile time recursive loop
 *
 * @param functor object that is forwarded
 *
 * @return (void)
 */
template <size_t Start, size_t End, typename Func>
  requires(End - Start < MAX_STEP_EVAL)
constexpr void constexpr_for(Func&& f) {
  if constexpr (Start < End) {
    f(std::integral_constant<size_t, Start>{});
    constexpr_for<Start + 1, End>(std::forward<Func>(f));
  }
}

/**
 * @brief argsort in descending manner
 *
 * @param Vector type T size N x 1
 *
 * @return Vector of indices
 */
template <typename T, size_t N>
  requires(!is_complex_v<T>)
constexpr auto argsort_desc(const Vector<T, N>& x) {
  std::array<size_t, N> p;
  for (size_t i = 0; i < N; ++i) p[i] = i;
  for (size_t i = 0; i + 1 < N; ++i) {
    size_t largest = i;
    for (size_t j = i + 1; j < N; ++j)
      if (x(p[j]) > x(p[largest])) largest = j;
    if (largest != i) std::swap(p[i], p[largest]);
  }
  return p;
}

}  // namespace algebra
