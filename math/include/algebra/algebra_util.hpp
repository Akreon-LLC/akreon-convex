/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include <optional>

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
template <FieldLike T>
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
template <FieldLike T>
  requires(!is_complex_v<T>)
constexpr T constexpr_abs(const T& x) {
  return x < 0 ? -1 * x : x;
}

/**
 * @brief perform square root of input variable x via root finding
 *
 * f(g) = g^2 - x \to df/dg = 2 g
 * Newton formula : g_{k+1} = g_k - \frac{g^2 - x} / {2 g_k} \equiv 0.5 * (g_n +
 * x / g_n)
 *
 * @param x \in R
 * @param g \in R (guess)
 *
 * @return sqrt(x)
 */
template <FieldLike T>
  requires(!is_complex_v<T>)
constexpr T constexpr_sqrt(const T& x, T g = 1e-10, T tol = 1e-10) {
  if (x == T(0)) return T(1e-30);
  // (dakre) TODO refactor to optional or something this is a global for
  // removing non-constexpr safe assert calls
  if (x < T(0)) return T(size_t(-1));

  for (size_t k = 0; k < MAX_STEP_EVAL; ++k) {
    if (constexpr_abs(g * g - x) <= tol) return g;
    g = T(0.5) * (g + x / g);
  }
  return g;
}

/**
 * @brief nth root of an input var x \in Reals via newton step
 *
 * f(g) = g^n - x \to df/dg = n g^{n-1}
 * Newton formula : g_{k+1} = g_k - \frac{g_k^n - x}{n g^{n-1}} \equiv
 * g_{k+1} = 1/n ((n - 1) g_k + x / g_k^{n-1})
 *
 * @param x \in R
 * @param root \in N
 * @param g \n R (guess)
 *
 * @return nth root of x
 */
template <RealLike T>
constexpr T constexpr_nrt(T x, size_t root, T g = 1e-10, T tol = 1e-10) {
  if (x == T(0)) return T(1e-30);

  if (root % 2 == 1) {
    if (x < T(0)) x = -x, g = -g;
  } else {
    // (dakre) similar TODO as above
    if (x < T(0)) return T(size_t(-1));
  }

  auto pow_n = [&]() {
    T pow(1);
    for (size_t i = 0; i < root; ++i) pow *= g;
    return pow;
  };

  for (size_t k = 0; k < MAX_STEP_EVAL; ++k) {
    T gkm1(1);
    for (size_t l = 0; l < root - 1; ++l) gkm1 *= g;

    g = (T(root - 1) * g + x / gkm1) / T(root);
    if (constexpr_abs(pow_n() - x) <= tol) return g;
  }
  return g;
}

/**
 * @brief perform abs on complex var
 *
 * @param some type T var x
 *
 * @return |x|
 */
template <ComplexLike T>
constexpr scalar_type_t<T> constexpr_absc(const T& x) {
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
template <RealLike T>
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
template <RealLike T>
constexpr auto constexpr_cos(const T& x, int terms = 16) {
  T sum = T(1), term = T(1);
  for (int k = 1; k < terms; ++k) {
    term *= (-(x * x)) / (T(2 * k - 1) * T(2 * k));
    sum += term;
  }
  return sum;
}

/**
 * @brief atanh (inverse hyperbolic tangent function)
 *
 * atanh(x) = x + x^3 / 3 + x^5 / 5 + ... = \sum_{n=0}^{\infty}
 * \frac{x^{2n+1}}{2n+1}
 *
 * @param x \in (-1, 1)
 *
 * @return y \in R
 */
template <RealLike T>
constexpr std::optional<T> constexpr_atanh(const T& x, int terms = 16) {
  if (constexpr_abs(x) >= T(1)) return std::nullopt;
  const T x2 = x * x;
  T term(x), sum(x);
  for (int k = 1; k < terms; ++k) {
    term *= x2;
    sum += (term / T(2 * k + 1));
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
template <RealLike T>
constexpr auto constexpr_expi(const T& theta, int terms = 16) {
  return std::complex<T>(constexpr_cos(theta, terms),
                         constexpr_sin(theta, terms));
}

/**
 * @brief exp (sum of a power series)
 *
 * exp(x) = 1 + x + x^2/2! + x^3/3!... = \sum_{n=0}^\infty
 * \frac{x^n}{n!}
 *
 * @param x \in R
 *
 * @return real number
 */
template <RealLike T>
constexpr auto constexpr_exp(const T& x, int terms = 16) {
  T sum(1), term(1);
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
template <ComplexLike T>
constexpr auto constexpr_exp_lambda(const T& lambda, int terms = 16) {
  return constexpr_exp(lambda.real()) * constexpr_expi(lambda.imag(), terms);
}

/**
 * @brief natural log
 *
 * [old] less numerically friendly
 * Taylor series expansion for 1 + t, t = 0
 * log(1 + t) = t - t^2 / 2 + t^3 / 3 - ... =
 * \sum_{k=1}^{\infty}(-1)^{k+1}\frac{t^k}{k}
 *
 * Restricting x \in R_++, and let t = x - 1... log(1 + x - 1) = log(x)
 * log(x) \approx (x - 1) - (x - 1)^2 / 2 + (x - 1)^3 - ...
 *
 * [new] use atanh
 * log(x) \approx 2 * atan(y) where y = (x - 1) / (x + 1)
 *
 * @return real number
 */
template <RealLike T>
constexpr std::optional<T> constexpr_ln(const T& x, int terms = 16) {
  if (x <= T(0)) return std::nullopt;

  const T y = (x - T(1)) / (x + T(1));
  const auto res = constexpr_atanh(y, terms);
  if (!res.has_value()) return std::nullopt;
  return T(2) * res.value();
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
 * @brief compare complex vars
 *
 * @param complex var a
 * @param complex var b
 *
 * @return T/F
 */
template <ComplexLike T>
constexpr bool is_same(const T& a, const T& b, scalar_type_t<T> tol = 1e-10) {
  return constexpr_absc(a - b) < tol;
}

/**
 * @brief argsort in descending manner
 *
 * @param Vector type T size N x 1
 *
 * @return Vector of indices
 */
template <VectorLike Vec>
  requires(RealLike<vec_type_t<Vec>>)
constexpr auto argsort_desc(const Vec& x) {
  static constexpr size_t N = vec_size_v<Vec>;

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
