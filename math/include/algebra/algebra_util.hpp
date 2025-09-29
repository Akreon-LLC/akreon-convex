/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "algebra_traits.hpp"

/// @brief macro for capping compile step eval (default max is ~512)
#define MAX_STEP_EVAL 256

namespace linalg {

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
 * @brief perform abs on complex var
 *
 * @param some type T var x
 *
 * @return |x|
 */
template <typename T>
  requires(is_complex_v<T>)
constexpr auto constexpr_absc(const T& x) {
  return constexpr_sqrt(x.real() * x.real() + x.imag() * x.imag());
}

/**
 * @brief perform sin via taylor series approximation
 *
 * \sum_{k=0}^{\infty}\frac{(-1)^{k}x^{2k+1}}{(2k+1)!}
 *
 * @param x as type T
 *
 * @return cosine approx of x
 */
template <typename T>
  requires(FieldLike<T> && !is_complex_v<T>)
constexpr auto constexpr_sin(const T& x, int terms = 16) {
  T sum = x, term = x;
  for (int k = 1; k < terms; ++k) {
    term *= (-(x * x)) / (T(2 * k) * T(2 * k + 1));
    sum += term;
  }
  return sum;
}

/**
 * @brief perform sin via taylor series approximation
 *
 * \sum_{k=0}^{\inf}\frac{(-1)^{k}x^{2k}}{(2k)!} = cos x
 *
 * @param x as type T
 *
 * @return cosine approx of x
 */
template <typename T>
  requires(FieldLike<T> && !is_complex_v<T>)
constexpr auto constexpr_cos(const T& x, int terms = 16) {
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
constexpr auto constexpr_expi(const T& theta, int terms = 16) {
  return std::complex<T>(constexpr_cos(theta, terms),
                         constexpr_sin(theta, terms));
}

/**
 * @brief exp on real value
 *
 * @param x \in R
 *
 * @return real number
 */
template <typename T>
  requires(FieldLike<T> && !is_complex_v<T>)
constexpr auto constexpr_exp(const T& x, int terms = 16) {
  T sum = 1;
  T term = 1;
  for (int k = 1; k < terms; ++k) {
    term *= x / k;
    sum += term;
  }
  return sum;
}

/**
 * @brief perform exp on eigenvalue : \lambda = \alpha + j \omega
 *
 * @param lambda \in C
 *
 * @return complex number
 */
template <typename T>
  requires(FieldLike<T> && is_complex_v<T>)
constexpr auto constexpr_exp_lambda(const T& lambda, int terms = 16) {
  return constexpr_exp(lambda.real()) * constexpr_expi(lambda.imag(), terms);
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

}  // namespace linalg
