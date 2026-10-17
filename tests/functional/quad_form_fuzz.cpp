/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#include "convex_fuzz_common.hpp"
#include <cmath>

namespace functional_test {

// Independent solution of min 0.5*x^T*P*x+q^T*x+r, x>=0, sum(x)<=cap.
template <std::size_t N>
auto diagonal_reference(const MatT<N>& P, const VecT<N>& q, double cap) {
  auto point = [&](double lambda) {
    VecT<N> value{};
    for (std::size_t i = 0; i < N; ++i)
      value(i) = std::max(0.0, -(q(i) + lambda) / P(i, i));
    return value;
  };
  if (sum_decision<N>(point(0.0)) <= cap) return point(0.0);
  double lo = 0.0, hi = 1.0;
  while (sum_decision<N>(point(hi)) > cap) hi *= 2.0;
  for (unsigned k = 0; k < 80; ++k) {
    const double mid = (lo + hi) / 2.0;
    if (sum_decision<N>(point(mid)) > cap) lo = mid;
    else hi = mid;
  }
  return point(hi);
}

template <std::size_t N>
bool check_quad_form(const MatT<N>& P, const VecT<N>& q, double r, double cap,
                     std::size_t trial, bool regularizer) {
  const auto x = Variable<VarMeta<SpaceT<N>>>();
  Matrix<double, N + 1, N> A{};
  Vector<double, N + 1> b{};
  const double scale = 1.0 / cap;
  for (std::size_t i = 0; i < N; ++i) {
    A(i, i) = -scale;
    A(N, i) = scale;
  }
  b(N) = cap * scale;
  const auto prob = make_problem(x)
                        .in_set(Polyhedron(A, b, x))
                        .minimize(make_quad_form(P, q, r, x));
  constexpr std::size_t Columns =
      mat_cols_v<decltype(prob.constraint_map().linear_part())>;
  if constexpr (Columns != N + 2) {
    if (trial == 0)
      std::printf("QuadForm has %zu auxiliaries; expected coupled norm and "
                  "square (2)\n",
                  Columns - N);
    return false;
  } else {
    const auto reference = diagonal_reference<N>(P, q, cap);
    // Verify an independent feasible witness strictly inside the x constraints.
    // Then place its norm and square auxiliaries on their exact boundaries.
    Vector<double, Columns> witness{};
    for (std::size_t i = 0; i < N; ++i) witness(i) = cap / (2.0 * N);
    double quadratic = 0.0, expected_cost = r;
    for (std::size_t i = 0; i < N; ++i) {
      quadratic += 0.5 * P(i, i) * witness(i) * witness(i);
      expected_cost +=
          0.5 * P(i, i) * reference(i) * reference(i) + q(i) * reference(i);
    }
    witness(N) = std::sqrt(quadratic);
    witness(N + 1) = quadratic;
    const auto mapped = prob.constraint_map()(witness);
    const auto c = prob.linear_objective();
    bool emission_ok =
        all_nonnegative<N>(witness, 0.0) && sum_decision<N>(witness) <= cap &&
        all_nonnegative<N>(reference, 0.0) &&
        sum_decision<N>(reference) <= cap + 1e-8 && prob.validate() &&
        std::abs(c(N)) <= 1e-12 && std::abs(c(N + 1) - 1.0) <= 1e-12 &&
        std::abs(prob.builder.coord_map.translation()(0) - r) <= 1e-12;
    for (std::size_t i = 0; i < N; ++i)
      emission_ok &= std::abs(c(i) - q(i)) <= 1e-12;
    const std::size_t norm_offset = N + 1;
    double norm_sq = 0.0;
    for (std::size_t i = 0; i < N; ++i)
      norm_sq += mapped(norm_offset + i) * mapped(norm_offset + i);
    emission_ok &= mapped(norm_offset + N) >= std::sqrt(norm_sq) - 1e-8;
    const std::size_t square_offset = 2 * N + 2;
    emission_ok &=
        mapped(square_offset + 2) >=
        std::hypot(mapped(square_offset), mapped(square_offset + 1)) - 1e-8;
    // Lowering the bound while keeping the norm coordinate must be rejected.
    auto below = witness;
    below(N + 1) -= 0.1;
    const auto invalid = prob.constraint_map()(below);
    emission_ok &=
        invalid(square_offset + 2) <
        std::hypot(invalid(square_offset), invalid(square_offset + 1));
    if (!emission_ok) {
      std::printf("trial=%zu regularizer=%d: invalid canonical emission\n",
                  trial, regularizer);
      return false;
    }
    const auto cfg = make_fuzz_config();
    const PrimalDualIPM solver(prob, cfg);
    const auto& res = solver.res;
    double actual_cost = r;
    bool feasible = all_nonnegative<N>(res.iterate.z, TOL) &&
                    sum_decision<N>(res.iterate.z) <= cap + TOL;
    for (std::size_t i = 0; i < Columns; ++i)
      feasible &= std::isfinite(res.iterate.z(i));
    for (std::size_t i = 0; i < N; ++i)
      actual_cost += 0.5 * P(i, i) * res.iterate.z(i) * res.iterate.z(i) +
                     q(i) * res.iterate.z(i);
    feasible &=
        res.iterate.z(N) >= -TOL &&
        res.iterate.z(N) * res.iterate.z(N) <= res.iterate.z(N + 1) + TOL;
    double norm_required = 0.0;
    for (std::size_t i = 0; i < N; ++i)
      norm_required += 0.5 * P(i, i) * res.iterate.z(i) * res.iterate.z(i);
    feasible &= norm_required <= res.iterate.z(N) * res.iterate.z(N) + TOL;
    // Gap is normalized by N+1 orthant coordinates and two SOCs (nu=2 each).
    const double objective_tol = (N + 5) * cfg.duality_gap_tol + TOL;
    const bool ok = res.tag == SolverResult::OPTIMAL && !res.stalled &&
                    feasible &&
                    std::abs(actual_cost - expected_cost) <= objective_tol;
    if (!ok)
      std::printf("trial=%zu regularizer=%d tag=%u stalled=%d feasible=%d "
                  "gap=%.6g cost=%.9g reference=%.9g\n",
                  trial, regularizer, unsigned(res.tag), res.stalled, feasible,
                  res.mu, actual_cost, expected_cost);
    return ok;
  }
}

} // namespace functional_test

int main() {
  using namespace functional_test;
  std::mt19937 rng(0xC0FFEE);
  std::uniform_real_distribution<double> cap_dist(1.0, 3.0), p_dist(0.2, 1.5),
      q_dist(-0.5, 0.5), r_dist(-0.25, 0.25);
  const auto cfg = make_fuzz_config();
  std::printf("constraints=normalized steps=%zu primal_tol=%.3g dual_tol=%.3g "
              "gap_tol=%.3g\n",
              cfg.steps, cfg.primal_feas_tol, cfg.dual_feas_tol,
              cfg.duality_gap_tol);
  std::size_t failures = 0;
  for (std::size_t trial = 0; trial < TRIALS; ++trial) {
    const double cap = cap_dist(rng);
    const auto P = make_diag_psd<6>(rng, p_dist);
    const auto q = make_vec<6>(rng, q_dist);
    const double r = r_dist(rng);
    failures += !check_quad_form<6>(P, q, r, cap, trial, false);
    failures += !check_quad_form<6>(P, VecT<6>{}, r, cap, trial, true);
  }
  std::printf(
      "quad_form_fuzz: %zu/%zu passed (100 QPs + 100 pure regularizers)\n",
      2 * TRIALS - failures, 2 * TRIALS);
  return failures == 0 ? 0 : 1;
}
