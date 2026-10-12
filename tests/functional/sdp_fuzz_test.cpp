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

#include "convex_fuzz_common.hpp"
#include <algorithm>
#include <array>
#include <charconv>
#include <string_view>
#include <cmath>

namespace functional_test {

// Independent Cholesky check of M + tolerance*I. No solver factorization or
// spectral routines are used in the feasibility oracle.
template <std::size_t N>
bool psd_with_tolerance(const MatT<N>& matrix, double tolerance) {
  MatT<N> lower{};
  for (std::size_t i = 0; i < N; ++i) {
    for (std::size_t j = 0; j <= i; ++j) {
      double value = matrix(i, j) + (i == j ? tolerance : 0.0);
      for (std::size_t k = 0; k < j; ++k) value -= lower(i, k) * lower(j, k);
      if (!std::isfinite(value)) return false;
      if (i == j) {
        if (value <= 0.0) return false;
        lower(i, j) = std::sqrt(value);
      } else lower(i, j) = value / lower(j, j);
    }
  }
  return true;
}

// min <C,X>, B <= X <= B+H. Write X=B+D*Z*D, H=D^2,
// C=D^-1*Q*diag(lambda)*Q^T*D^-1. Then 0<=Z<=I and the
// objective is <C,B> + sum(lambda_i * (Q^T Z Q)_ii).
// Without equality, fill precisely the negative eigenvalue directions.
// With tr(Z)=budget, fill directions in increasing eigenvalue order.
// These give independent primal/dual optimality bounds, including a
// fractional eigenvalue direction in the equality case.
template <std::size_t N, bool Equality>
bool check_sdp(std::mt19937& rng, std::size_t trial, std::size_t steps) {
  constexpr std::size_t M = N * (N + 1) / 2;
  std::uniform_real_distribution<double> unit(0.0, 1.0), signed_unit(-1.0, 1.0);
  auto Q = make_identity<double, N>();
  for (std::size_t i = 0; i < N; ++i) {
    for (std::size_t j = i + 1; j < N; ++j) {
      const double angle = 6.0 * signed_unit(rng);
      const double c = std::cos(angle), s = std::sin(angle);
      for (std::size_t k = 0; k < N; ++k) {
        const double a = Q(k, i), b = Q(k, j);
        Q(k, i) = c * a - s * b;
        Q(k, j) = s * a + c * b;
      }
    }
  }
  std::array<double, N> diagonal{}, eigenvalues{}, allocation{};
  MatT<N> B{}, H{}, C{}, reference{};
  for (std::size_t i = 0; i < N; ++i) {
    diagonal[i] = 0.7 + 0.8 * unit(rng);
    eigenvalues[i] = signed_unit(rng);
    for (std::size_t j = 0; j <= i; ++j)
      B(i, j) = B(j, i) = 0.3 * signed_unit(rng);
    H(i, i) = diagonal[i] * diagonal[i];
  }
  // Both signs ensure a nontrivial, boundary optimum in every box trial.
  eigenvalues[0] = -0.2 - unit(rng);
  eigenvalues[N - 1] = 0.2 + unit(rng);
  std::sort(eigenvalues.begin(), eigenvalues.end());
  const double budget = 0.25 + (N - 0.5) * unit(rng);
  double remaining = budget;
  for (std::size_t k = 0; k < N; ++k) {
    allocation[k] = Equality ? std::clamp(remaining, 0.0, 1.0)
                             : (eigenvalues[k] < 0.0 ? 1.0 : 0.0);
    remaining -= allocation[k];
  }
  for (std::size_t i = 0; i < N; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      reference(i, j) = B(i, j);
      for (std::size_t k = 0; k < N; ++k) {
        C(i, j) +=
            Q(i, k) * eigenvalues[k] * Q(j, k) / (diagonal[i] * diagonal[j]);
        reference(i, j) +=
            diagonal[i] * Q(i, k) * allocation[k] * Q(j, k) * diagonal[j];
      }
    }
  }
  // Build packed symmetric coordinates explicitly so the oracle also checks
  // the sqrt(2) off-diagonal convention used by LMI canonicalization.
  Vector<MatT<N>, M> coefficients{}, negative_coefficients{};
  VecT<M> objective{}, trace_coefficients{}, witness{};
  double base_cost = 0.0, base_trace = 0.0;
  std::size_t index = 0;
  for (std::size_t i = 0; i < N; ++i) {
    for (std::size_t j = 0; j <= i; ++j, ++index) {
      const double weight = i == j ? 1.0 : std::sqrt(2.0);
      coefficients(index)(i, j) = coefficients(index)(j, i) = 1.0 / weight;
      negative_coefficients(index) = -1.0 * coefficients(index);
      objective(index) = weight * C(i, j);
      witness(index) = weight * reference(i, j);
      base_cost += (i == j ? 1.0 : 2.0) * C(i, j) * B(i, j);
      if (i == j) {
        trace_coefficients(index) = 1.0 / H(i, i);
        base_trace += B(i, i) / H(i, i);
      }
    }
  }
  const auto x = Variable<VarMeta<SpaceT<M>>>();
  const auto bounded = make_problem(x)
                           .in_set(LMI(coefficients, -1.0 * B, x))
                           .in_set(LMI(negative_coefficients, B + H, x));
  const auto prob = [&] {
    const auto cost = WeightedSum<SpaceT<M>>()(objective, x);
    if constexpr (Equality)
      return bounded
          .in_set(Hyperplane(trace_coefficients, base_trace + budget, x))
          .minimize(cost);
    else return bounded.minimize(cost);
  }();
  double expected_cost = base_cost;
  for (std::size_t i = 0; i < N; ++i)
    expected_cost += eigenvalues[i] * allocation[i];
  bool emission_ok = prob.validate();
  const auto mapped = prob.constraint_map()(witness);
  index = 0;
  for (std::size_t i = 0; i < N; ++i) {
    for (std::size_t j = 0; j <= i; ++j, ++index) {
      const double weight = i == j ? 1.0 : std::sqrt(2.0);
      emission_ok &= std::abs(mapped(index) -
                              weight * (reference(i, j) - B(i, j))) < 1e-10;
      emission_ok &=
          std::abs(mapped(M + index) -
                   weight * (B(i, j) + H(i, j) - reference(i, j))) < 1e-10;
    }
  }
  if constexpr (Equality)
    emission_ok &= std::abs((prob.equality_map().linear_part() * witness)(0) -
                            prob.equality_map().translation()(0)) < 1e-10;
  emission_ok &=
      std::abs(inner_product(objective, witness) - expected_cost) < 1e-10;
  emission_ok &= psd_with_tolerance<N>(reference - B, 1e-9) &&
                 psd_with_tolerance<N>(B + H - reference, 1e-9);
  auto cfg = make_fuzz_config();
  cfg.steps = steps;
  const PrimalDualIPM solver(prob, cfg);
  const auto& res = solver.res;
  MatT<N> actual{};
  index = 0;
  bool finite = true;
  for (std::size_t i = 0; i < N; ++i) {
    for (std::size_t j = 0; j <= i; ++j, ++index) {
      finite &= std::isfinite(res.iterate.z(index));
      actual(i, j) = actual(j, i) =
          res.iterate.z(index) / (i == j ? 1.0 : std::sqrt(2.0));
    }
  }
  const double cost = inner_product(objective, res.iterate.z);
  const double eq_error =
      Equality ? std::abs(inner_product(trace_coefficients, res.iterate.z) -
                          base_trace - budget)
               : 0.0;
  const bool feasible = finite && psd_with_tolerance<N>(actual - B, TOL) &&
                        psd_with_tolerance<N>(B + H - actual, TOL) &&
                        eq_error <= TOL;
  const bool ok =
      emission_ok && feasible && res.tag == SolverResult::OPTIMAL &&
      !res.stalled && std::isfinite(cost) &&
      std::abs(cost - expected_cost) <= 2 * N * cfg.duality_gap_tol + TOL;
  if (!ok) {
    std::printf("seed=0xC0FFEE order=%zu equality=%d trial=%zu emission=%d "
                "feasible=%d tag=%u stalled=%d event=%u iterations=%zu "
                "gap=%.6g cost=%.9g reference=%.9g equality_error=%.6g\n",
                N, Equality, trial, emission_ok, feasible, unsigned(res.tag),
                res.stalled, unsigned(res.event), res.num_iter, res.mu, cost,
                expected_cost, eq_error);
    std::printf("budget=%.17g; B, H, C rows:\n", budget);
    for (std::size_t i = 0; i < N; ++i) {
      for (const auto& matrix : {B, H, C}) {
        for (std::size_t j = 0; j < N; ++j) std::printf(" %.17g", matrix(i, j));
        std::printf(" | ");
      }
      std::printf("\n");
    }
  }
  return ok;
}
} // namespace functional_test

int main(int argc, char** argv) {
  using namespace functional_test;
  // Allow the slow equality-constrained cases to converge; an optional budget
  // supports convergence ablations without changing cases or acceptance
  // tolerances.
  std::size_t steps = 256;
  if (argc > 2) return 2;
  if (argc == 2) {
    const std::string_view argument(argv[1]);
    const auto parsed = std::from_chars(
        argument.data(), argument.data() + argument.size(), steps);
    if (parsed.ec != std::errc{} ||
        parsed.ptr != argument.data() + argument.size() || steps == 0) {
      std::fprintf(stderr,
                   "usage: sdp_fuzz_test [positive iteration budget]\n");
      return 2;
    }
  }
  std::printf("seed=0xC0FFEE steps=%zu tolerance=%.3g\n", steps, TOL);
  std::mt19937 rng(0xC0FFEE);
  std::size_t failures = 0;
  for (std::size_t trial = 0; trial < TRIALS; ++trial) {
    failures += !check_sdp<2, false>(rng, trial, steps);
    failures += !check_sdp<2, true>(rng, trial, steps);
    failures += !check_sdp<3, false>(rng, trial, steps);
    failures += !check_sdp<3, true>(rng, trial, steps);
  }
  std::printf("sdp_fuzz_test: %zu/%zu passed (dense PSD bounds, with/without "
              "equality)\n",
              4 * TRIALS - failures, 4 * TRIALS);
  return failures == 0 ? 0 : 1;
}
