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

#include <catch2/catch_test_macros.hpp>

#include "control.hpp"
#include "math.hpp"
#include "test_helpers.hpp"

using namespace linalg;
using namespace convex;
using namespace control;
using namespace unit_test;
using namespace config;

TEST_CASE("Small LQR 2x2", "[lqr_test]") {
  using Space = Matrixd<2, 2>;
  constexpr auto P = Variable<PsdVarMeta<Space>>();
  constexpr auto prob_init = make_problem(P);

  constexpr auto MatBuilder = mat_traits<Space>();

  SECTION("LQR case 1") {
    constexpr auto sqrt2 = constexpr_sqrt_safe(2.0);
    constexpr auto Q = MatBuilder(1.0, 0.0, 0.0, 1.0);
    constexpr auto R = MatBuilder(1.0, 0.0, 0.0, 1.0);
    constexpr auto A = MatBuilder(1.0, 0.0, 0.0, 1.0);
    constexpr auto B = MatBuilder(1.0, 0.0, 0.0, 1.0);
    constexpr auto C = MatBuilder(1.0, 0.0, 0.0, 1.0);
    constexpr auto ss =
        StateSpace<TimeDomainType::CONTINUOUS, Space, Space, Space>(A, B, C);
    constexpr auto lqr = LQR<decltype(ss), decltype(Q), decltype(R)>();
    constexpr auto lmi = lqr(ss, Q, R, P);
    constexpr auto prob_constr = prob_init.in_set(lmi);
    constexpr auto prob_min = prob_constr.feasibility();
    STATIC_REQUIRE(is_problem_set_v<decltype(prob_min)>);
    STATIC_REQUIRE(
        std::tuple_size<decltype(prob_min.constraint_cones())>::value == 2);

    constexpr auto F_mat = mat_traits<Matrixd<13, 3>>();
    constexpr auto F_exp =
        F_mat(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, -2.0, 0.0, 0.0, 0.0,
              -2.0 * sqrt2, 0.0, sqrt2, 0.0, 0.0, 0.0, sqrt2, 0.0, 0.0, 0.0,
              -2.0, 0.0, sqrt2, 0.0, 0.0, 0.0, sqrt2, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0);
    constexpr auto F = prob_min.constraint_map().linear_part();
    STATIC_REQUIRE(mat_rows_v<decltype(F)> == 13);
    STATIC_REQUIRE(mat_cols_v<decltype(F)> == 3);
    STATIC_REQUIRE(all_equal(F, F_exp));

    constexpr auto g = prob_min.constraint_map().translation();
    STATIC_REQUIRE(vec_size_v<decltype(g)> == 13);
    STATIC_REQUIRE(g == Vectord<13>{0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0,
                                    0.0, 0.0, 1.0, 0.0, 1.0});

    STATIC_REQUIRE(prob_min.linear_objective() == Vectord<3>{});

    PrimalDualIPM solver(prob_min);
    auto res = solver.res;
    //REQUIRE(res.tag == SolverResult::OPTIMAL);
    //REQUIRE(res.num_iter < PD_IPM_STEPS);

#if AKREON_DEBUG
    auto trace = solver.trace;
    //REQUIRE(trace[res.num_iter - 1].tag == config::SolverResult::OPTIMAL);
    log(res, trace);
#endif
  }
}
