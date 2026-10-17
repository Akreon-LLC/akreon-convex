/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdio>
#include <random>

#include "math.hpp"

using namespace linalg;
using namespace convex;
using namespace config;

namespace functional_test {

/// @brief global fuzz test params
constexpr std::size_t TRIALS = 100;
constexpr double TOL = 1e-3;

/// @brief aliases for fuzz tests
template <std::size_t N>
using SpaceT = Matrix<double, 1, N>;

template <std::size_t N>
using VecT = Vector<double, N>;

template <std::size_t N>
using MatT = Matrix<double, N, N>;

/**
 * @brief make bounded R^n_+ prob
 *
 * @param x (dec var)
 * @param prob (convex problem)
 * @param cap (scalar)
 *
 * @return new prob
 */
template <std::size_t N, typename Var, typename Prob>
auto make_bounded_nonneg_problem(const Var& x, const Prob& prob,
                                 const double cap) {
  Matrix<double, N + 1, N> A{};
  Vector<double, N + 1> b{};

  for (std::size_t i = 0; i < N; ++i) {
    A(i, i) = -1.0;
    A(N, i) = 1.0;
  }
  b(N) = cap;

  const auto bounded_poly = Polyhedron(A, b, x);
  return prob.in_set(bounded_poly);
}

/**
 * @brief make R^n_+ prob
 *
 * @param x (dec var)
 * @param prob (convex problem)
 *
 * @return new prob
 */
template <std::size_t N, typename Var, typename Prob>
auto make_nonneg_problem(const Var& x, const Prob& prob) {
  Matrix<double, N, N> A{};
  Vector<double, N> b{};

  for (std::size_t i = 0; i < N; ++i) A(i, i) = -1.0;

  const auto nonneg_poly = Polyhedron(A, b, x);
  return prob.in_set(nonneg_poly);
}

/**
 * @brief make vec helper
 *
 * @param rng (random range)
 * @param d (uniform real dist)
 *
 * @return vec
 */
template <std::size_t N>
auto make_vec(std::mt19937& rng, std::uniform_real_distribution<double>& d) {
  VecT<N> x{};
  for (std::size_t i = 0; i < N; ++i) x(i) = d(rng);
  return x;
}

/**
 * @brief make diag psd
 *
 * @param rng (random range)
 * @param d (uniform real dist)
 *
 * @return P
 */
template <std::size_t N>
auto make_diag_psd(std::mt19937& rng,
                   std::uniform_real_distribution<double>& d) {
  MatT<N> P{};
  for (std::size_t i = 0; i < N; ++i) P(i, i) = d(rng);
  return P;
}

/**
 * @brief sum decision
 *
 * @param z (vec)
 *
 * @return sum
 */
template <std::size_t N, linalg::VectorContainer Vec>
double sum_decision(const Vec& z) {
  double s = 0.0;
  for (std::size_t i = 0; i < N; ++i) s += z(i);
  return s;
}

/**
 * @brief all nonneg
 *
 * @param z (vec)
 * @param tol (scalar)
 *
 * @return T/F
 */
template <std::size_t N, linalg::VectorContainer Vec>
bool all_nonnegative(const Vec& z, double tol) {
  for (std::size_t i = 0; i < N; ++i) {
    if (z(i) < -tol) return false;
  }
  return true;
}

/**
 * @brief make fuzz config
 *
 * @return config
 */
inline auto make_fuzz_config() {
  PrimalDualIPMConfig<double> cfg{};
  cfg.steps = 24;
  cfg.primal_feas_tol = TOL;
  cfg.dual_feas_tol = TOL;
  cfg.duality_gap_tol = TOL;
  return cfg;
}

/**
 * @brief run trials
 *
 * @param name (string)
 * @param trials (std::size_t)
 * @param trial func
 *
 * @return T/F
 */
template <typename Fn>
bool run_trials(const char* name, const std::size_t trials, Fn&& fn) {
  std::mt19937 rng(0xC0FFEE);
  for (std::size_t i = 0; i < trials; ++i) {
    if (!fn(rng)) {
      std::fprintf(stderr, "%s failed on trial %zu\n", name, i);
      return false;
    }
  }
  std::printf("%s passed %zu trials\n", name, trials);
  return true;
}

} // namespace functional_test
