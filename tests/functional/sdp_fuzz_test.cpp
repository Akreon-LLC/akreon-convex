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
 * @brief SDP trial
 *
 * TODO - currently not working well
 *
 * @param random (range)
 *
 * @return T/F
 */
template <size_t N>
bool run_sdp_trial(std::mt19937& rng) {
  using Space = SpaceT<N>;
  const auto x = Variable<VarMeta<Space>>();
  const auto prob_init = make_problem(x);
  const auto prob_nonneg = make_nonneg_problem<N>(x, prob_init);

  std::uniform_real_distribution<double> a_dist(0.5, 1.5);
  std::uniform_real_distribution<double> slack_dist(0.4, 1.2);
  std::uniform_real_distribution<double> c_dist(0.1, 1.0);

  Vector<MatT<N>, N> A{};
  MatT<N> B{};
  VecT<N> ub{};
  VecT<N> c{};

  for (size_t i = 0; i < N; ++i) {
    const double a = a_dist(rng);
    const double slack = slack_dist(rng);

    // LMI is B + sum_i x_i A_i >= 0
    // Choose diagonal structure so:
    //   B_ii = a + slack
    //   A_i(ii) = -a
    // Hence diagonal feasibility implies:
    //   a + slack - a x_i >= 0  =>  x_i <= (a + slack)/a
    A(i)(i)(i) = -a;
    B(i)(i) = a + slack;
    ub(i) = (a + slack) / a;

    c(i) = c_dist(rng);
  }

  const auto lmi = LMI<LMIMeta<Space>, decltype(x)>(A, B, x);
  const auto prob_constr = prob_nonneg.in_set(lmi);

  const auto obj = WeightedSum<Space>()(c, x);
  const auto prob = prob_constr.minimize(obj);

  const auto config = make_fuzz_config();
  const PrimalDualIPM solver(prob, config);
  const auto& res = solver.res;

  if (res.tag != SolverResult::OPTIMAL || res.stalled) {
    std::fprintf(stderr, "SDP solve failed: tag=%d stalled=%d iter=%zu\n",
                 static_cast<int>(res.tag), res.stalled, res.num_iter);
    return false;
  }

  if (!all_nonnegative<N>(res.iterate.z, TOL)) {
    std::fprintf(stderr, "SDP feasibility failed: nonnegative violated\n");
    return false;
  }

  for (size_t i = 0; i < N; ++i) {
    if (res.iterate.z(i) > ub(i) + TOL) {
      std::fprintf(stderr, "SDP feasibility failed: z(%zu)=%g exceeds ub=%g\n",
                   i, res.iterate.z(i), ub(i));
      return false;
    }
  }

  // Since c > 0 and x = 0 is feasible, optimum should be near zero.
  for (size_t i = 0; i < N; ++i) {
    if (constexpr_abs(res.iterate.z(i)) > 5e-3) {
      std::fprintf(stderr,
                   "SDP optimality sanity failed: z(%zu)=%g not near zero\n", i,
                   res.iterate.z(i));
      return false;
    }
  }

  return true;
}

}  // namespace functional_test

int main() {
  return functional_test::run_trials(
             "convex_sdp_fuzz", functional_test::TRIALS,
             [](std::mt19937& rng) {
               return functional_test::run_sdp_trial<2>(rng);
             })
             ? 0
             : -1;
}
