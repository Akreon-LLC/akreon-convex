/*
 * Copyright © 2025 Akreon LLC. All rights reserved.
 *
 * Proprietary and Confidential.
 * This source code is the property of Akreon LLC and constitutes a trade secret
 * protected by U.S. and international law.
 *
 * Unauthorized copying, modification, distribution, or disclosure of this file,
 * via any medium, is strictly prohibited without express written permission
 * from Akreon LLC.
 */

#pragma once

#include <array>
#include <limits>
#include <optional>

#include "algebra_traits.hpp"

/// @brief macro for capping compile step eval (default max is ~512)
#define MAX_STEP_EVAL 256

namespace linalg {

/**
 * @brief array to tuple
 *
 * @param array types
 *
 * @return tuple
 */
template <typename T, size_t N>
constexpr auto array_to_tuple(const std::array<T, N>& x) {
  auto to_tuple = [&]<size_t... Is>(const auto& arr,
                                    std::index_sequence<Is...>) {
    return std::make_tuple(arr[Is]...);
  };
  return to_tuple(x, std::make_index_sequence<N>{});
}

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
 * @brief constexpr-safe division with optional domain check
 *
 * @param num numerator
 * @param den denominator
 *
 * @return num / den when denominator is nonzero, else nullopt
 */
template <RealLike T>
constexpr std::optional<T> constexpr_div(const T& num, const T& den) {
  if (!(den != T(0))) return std::nullopt;
  return num / den;
}

/**
 * @brief constexpr-safe scalar division with fallback
 *
 * @param num numerator
 * @param den denominator
 * @param fallback fallback value when denominator is zero
 *
 * @return scalar division result or fallback
 */
template <RealLike T>
constexpr T constexpr_div_safe(const T& num, const T& den,
                               const T& fallback = T(0)) {
  if (const auto r = constexpr_div(num, den); r.has_value()) return *r;
  return fallback;
}

/**
 * @brief power of a scalar
 *
 * @param x \in R
 * @param power \in N
 *
 * @return pow
 */
template <RealLike T>
constexpr T constexpr_pow(const T& x, const size_t& power) {
  T res(x);
  for (size_t i = 1; i < power; ++i) res *= x;
  return res;
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
template <RealLike T>
constexpr std::optional<T> constexpr_sqrt(const T& x, T g = 1e-10,
                                          T tol = 1e-10) {
  if (!(x >= T(0))) return std::nullopt;
  if (x == T(0)) return std::numeric_limits<T>::min();
  if (!(g > T(0))) g = T(1);

  auto abs = [](const auto& val) { return val < T(0) ? -val : val; };

  for (size_t k = 0; k < MAX_STEP_EVAL; ++k) {
    if (!(g > T(0))) g = T(1);
    if (abs(g * g - x) <= tol) return g;
    g = T(0.5) * (g + constexpr_div_safe(x, g, g));
  }
  return g;
}

/**
 * @brief safe scalar sqrt wrapper around optional constexpr_sqrt
 *
 * @param x input scalar
 * @param floor minimum scalar for fallback
 * @param g initial guess
 * @param tol convergence tolerance
 *
 * @return scalar sqrt with optional fallback handling
 */
template <RealLike T>
constexpr T constexpr_sqrt_safe(const T& x, const T& floor = T(0), T g = 1e-10,
                                T tol = 1e-10) {
  if (const auto r = constexpr_sqrt(x, g, tol); r.has_value()) return *r;

  const auto floor_safe = (floor >= T(0)) ? floor : T(0);
  if (const auto rf = constexpr_sqrt(floor_safe, g, tol); rf.has_value())
    return *rf;

  return T(0);
}

/**
 * @brief perform absolute value computation
 *
 * @param x (FieldLike)
 *
 * @return |x|
 */
template <FieldLike T>
constexpr auto constexpr_abs(const T& x) {
  if constexpr (RealLike<T>)
    return x < 0 ? -1 * x : x;
  else
    return constexpr_sqrt_safe(x.real() * x.real() + x.imag() * x.imag());
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
constexpr std::optional<T> constexpr_nrt(T x, size_t root, T g = 1e-10,
                                         T tol = 1e-10) {
  if (root == 0) return std::nullopt;
  if (x == T(0)) return std::numeric_limits<T>::min();

  T sign = T(1);
  if (x < T(0)) {
    if (root % 2 == 0) return std::nullopt;
    sign = T(-1);
    x = -x;
  }

  if (!(g > T(0))) g = T(1);

  auto pow_n = [&](const T& base, const size_t& n) {
    T pow(1);
    for (size_t i = 0; i < n; ++i) pow *= base;
    return pow;
  };

  for (size_t k = 0; k < MAX_STEP_EVAL; ++k) {
    const auto gkm1 = pow_n(g, root - 1);
    if (!(gkm1 > T(0))) return std::nullopt;

    g = (T(root - 1) * g + constexpr_div_safe(x, gkm1, g)) / T(root);
    if (constexpr_abs(pow_n(g, root) - x) <= tol) return sign * g;
  }
  return sign * g;
}

/**
 * @brief safe scalar nth-root wrapper around optional constexpr_nrt
 *
 * @param x input scalar
 * @param root root degree
 * @param floor fallback scalar
 * @param g initial guess
 * @param tol convergence tolerance
 *
 * @return scalar nth-root with optional fallback handling
 */
template <RealLike T>
constexpr T constexpr_nrt_safe(T x, const size_t& root, const T& floor = T(0),
                               T g = 1e-10, T tol = 1e-10) {
  if (const auto r = constexpr_nrt(x, root, g, tol); r.has_value()) return *r;

  const auto floor_safe = (floor >= T(0)) ? floor : T(0);
  if (const auto rf = constexpr_nrt(floor_safe, root, g, tol); rf.has_value())
    return *rf;

  return T(0);
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
 * @brief constexpr-safe atanh with fallback
 *
 * @param x input scalar
 * @param fallback fallback value when |x| >= 1
 * @param terms series terms
 *
 * @return scalar atanh result or fallback
 */
template <RealLike T>
constexpr T constexpr_atanh_safe(const T& x, const T& fallback = T(0),
                                 int terms = 16) {
  if (const auto r = constexpr_atanh(x, terms); r.has_value()) return *r;
  return fallback;
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
 * [old 1] less numerically friendly
 * Taylor series expansion for 1 + t, t = 0
 * log(1 + t) = t - t^2 / 2 + t^3 / 3 - ... =
 * \sum_{k=1}^{\infty}(-1)^{k+1}\frac{t^k}{k}
 *
 * Restricting x \in R_++, and let t = x - 1... log(1 + x - 1) = log(x)
 * log(x) \approx (x - 1) - (x - 1)^2 / 2 + (x - 1)^3 - ...
 *
 * [old 2] use atanh
 * log(x) \approx 2 * atan(y) where y = (x - 1) / (x + 1)
 *
 * [new] range reduce x = m * 2^e where m in [0.5, 2]
 * \forall x > 0 we have x = m 2^e, m \in [0.5, 1] and e \in Z
 * ln(x) = ln(m) + e ln(2)
 * ** the above makes this much more numerically stable
 *
 * - perform atan(y) as before
 * - sum 2a with e*ln(2)
 *
 * @return real number
 */
template <RealLike T>
constexpr std::optional<T> constexpr_ln(const T& x_in, int terms = 16) {
  if (!(x_in > T(0))) return std::nullopt;

  T x = x_in;
  int e = 0;

  for (int i = 0; i < 64 && x > T(2); ++i) {
    x *= T(0.5);
    ++e;
  }
  for (int i = 0; i < 64 && x < T(0.5); ++i) {
    x *= T(2);
    --e;
  }

  const T y = constexpr_div_safe(x - T(1), x + T(1), T(0));
  const auto a = constexpr_atanh(y, terms);
  if (!a) return std::nullopt;

  constexpr T ln2 = T(0.693147180559945309417232121458176568L);

  return T(2) * (*a) + T(e) * ln2;
}

/**
 * @brief natural log with domain sanitization and fallback
 *
 * Ensures no optional access is required at call sites that need a scalar.
 * Handles NaN/invalid by clamping to a positive floor.
 *
 * @param x input scalar
 * @param floor positive floor value used when x is invalid/non-positive
 * @param terms series terms
 *
 * @return ln(max(x, floor)) with NaN-safe clamping semantics
 */
template <RealLike T>
constexpr T constexpr_ln_safe(const T& x,
                              const T& floor = std::numeric_limits<T>::min(),
                              int terms = 16) {
  const T floor_safe = (floor > T(0)) ? floor : std::numeric_limits<T>::min();
  const T x_safe = (x > floor_safe) ? x : floor_safe;
  const auto l = constexpr_ln(x_safe, terms);
  if (l.has_value()) return *l;

  const auto l_floor = constexpr_ln(floor_safe, terms);
  return l_floor.has_value() ? *l_floor : T(0);
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
  requires(End - Start >= 0)
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
  return constexpr_abs(a - b) < tol;
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

/// @brief useful constant ln(2)
template <FieldLike T>
inline constexpr T ln2_v{linalg::constexpr_ln_safe(T(2))};

}  // namespace linalg
