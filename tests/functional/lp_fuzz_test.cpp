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
 * @brief run lp trial
 *
 * @param rng (random range)
 *
 * @return T/F
 */
template <size_t N>
bool run_lp_trial(std::mt19937& rng) {
  using Space = SpaceT<N>;
  const auto x = Variable<VarMeta<Space>>();
  const auto prob_init = make_problem(x);
  std::uniform_real_distribution<double> cap_dist(1.0, 3.0);
  std::uniform_real_distribution<double> c_dist(0.1, 1.0);
  const double cap = cap_dist(rng);
  const auto prob_constr = make_bounded_nonneg_problem<N>(x, prob_init, cap);

  const auto c = make_vec<N>(rng, c_dist);
  const auto cx = WeightedSum<Space>()(c, x);
  const auto prob = prob_constr.minimize(cx);

  const auto config = make_fuzz_config();
  const PrimalDualIPM solver(prob, config);
  const auto& res = solver.res;
  if (res.tag != SolverResult::OPTIMAL || res.stalled) {
    std::fprintf(stderr, "LP solve failed: tag=%d stalled=%d iter=%zu\n",
                 static_cast<int>(res.tag), res.stalled, res.num_iter);
    return false;
  }

  if (!all_nonnegative<N>(res.iterate.z, TOL)) {
    std::fprintf(stderr, "LP feasibility failed: nonnegative violated\n");
    return false;
  }

  const auto cap_gap = sum_decision<N>(res.iterate.z) - cap;
  if (cap_gap >= TOL) {
    std::fprintf(stderr, "LP feasibility failed: cap gap=%g\n", cap_gap);
    return false;
  }

  return true;
}

}  // namespace functional_test

int main() {
  return functional_test::run_trials(
             "convex_lp_fuzz", functional_test::TRIALS,
             [](std::mt19937& rng) {
               return functional_test::run_lp_trial<6>(rng);
             })
             ? 0
             : -1;
}
