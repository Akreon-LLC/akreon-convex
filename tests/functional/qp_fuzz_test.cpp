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

namespace functional_test {

/**
 * @brief qp trial
 *
 * @param range (random range)
 *
 * @return T/F
 */
template <std::size_t N>
bool run_qp_trial(std::mt19937& rng) {
  using Space = SpaceT<N>;
  const auto x = Variable<VarMeta<Space>>();
  const auto prob_init = make_problem(x);
  std::uniform_real_distribution<double> cap_dist(1.0, 3.0);
  std::uniform_real_distribution<double> p_dist(0.2, 1.5);
  std::uniform_real_distribution<double> q_dist(-0.5, 0.5);
  std::uniform_real_distribution<double> r_dist(-0.25, 0.25);
  const double cap = cap_dist(rng);
  const auto prob_constr = make_bounded_nonneg_problem<N>(x, prob_init, cap);

  const auto P = make_diag_psd<N>(rng, p_dist);
  const auto q = make_vec<N>(rng, q_dist);
  const double r = r_dist(rng);
  const auto obj = QuadForm<Space>()(P, q, r, x);
  const auto prob = prob_constr.minimize(obj);

  const auto config = make_fuzz_config();
  const PrimalDualIPM solver(prob, config);
  const auto& res = solver.res;
  if (res.tag != SolverResult::OPTIMAL || res.stalled) return false;

  if (!all_nonnegative<N>(res.iterate.z, TOL)) return false;
  return sum_decision<N>(res.iterate.z) <= cap + TOL;
}

} // namespace functional_test

int main() {
  return functional_test::run_trials("convex_qp_fuzz", functional_test::TRIALS,
                                     [](std::mt19937& rng) {
                                       return functional_test::run_qp_trial<6>(
                                           rng);
                                     })
             ? 0
             : -1;
}
