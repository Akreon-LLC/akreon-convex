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

#include <algorithm>
#include <catch2/catch_test_macros.hpp>

#include "math.hpp"
#include "test_helpers.hpp"

using namespace linalg;
using namespace convex;
using namespace affine;
using namespace conic;
using namespace unit_test;
using namespace config;

static constexpr auto rel_tol{1e-3};

TEST_CASE("Minimal SOCP 2 Vars", "[socp_test]") {
  using Space = Matrixd<1, 2>;
  using Singleton = Matrixd<1, 1>;
  constexpr auto x = Variable<VarMeta<Space>>();
  constexpr auto x_trans = x.add(Vectord<2>{-1.0, 2.0});
  constexpr auto prob_init = make_problem(x_trans);
  constexpr auto norm =
      Norm<NormMeta<Space, Singleton>, decltype(x_trans), L2<Vectord<2>>>(
          x_trans);
  constexpr auto square = Square(norm);
  constexpr auto half_square = square.mul<0.5>();
  constexpr auto inv_sqrt2 = 1.0 / constexpr_sqrt_safe(2.0);
  constexpr auto sqrt5 = constexpr_sqrt_safe(5.0);

  constexpr auto F_mat = mat_traits<Matrixd<6, 4>>();
  constexpr auto F_exp =
      F_mat(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -inv_sqrt2, 0.0, 0.0, 0.0, inv_sqrt2);
  constexpr Vectord<6> g_exp{-1.0,           2.0, 0.0, 0.0, 0.5 * inv_sqrt2,
                             0.5 * inv_sqrt2};
  constexpr Vectord<2> x_unconstrained_exp{1.0, -2.0};
  constexpr Vectord<2> x_ball_exp{1.0 / sqrt5, -2.0 / sqrt5};

  SECTION("Unconstrained SOCP") {
    constexpr auto prob_min = prob_init.minimize(half_square);

    STATIC_REQUIRE(is_problem_set_v<decltype(prob_min)>);
    STATIC_REQUIRE(prob_min.validate());
    STATIC_REQUIRE(
        std::tuple_size<decltype(prob_min.constraint_cones())>::value == 2);

    constexpr auto F = prob_min.constraint_map().linear_part();
    STATIC_REQUIRE(mat_rows_v<decltype(F)> == 6);
    STATIC_REQUIRE(mat_cols_v<decltype(F)> == 4);
    STATIC_REQUIRE(all_equal(F, F_exp));

    constexpr auto g = prob_min.constraint_map().translation();
    STATIC_REQUIRE(vec_size_v<decltype(g)> == 6);
    STATIC_REQUIRE(g == g_exp);

    constexpr auto c = prob_min.linear_objective();
    STATIC_REQUIRE(c == Vectord<4>{0.0, 0.0, 0.0, 0.5});

    constexpr auto config = [] {
      PrimalDualIPMConfig<double> cfg{};
      cfg.duality_gap_tol = 1e-4;
      return cfg;
    }();
    constexpr PrimalDualIPM solver(prob_min, config);
    constexpr auto res = solver.res;
#ifdef AKREON_REPORT_OUTPUT
    constexpr auto trace = solver.trace;
#endif

    STATIC_REQUIRE(res.tag == SolverResult::OPTIMAL);
    STATIC_REQUIRE(res.num_iter > 0 && res.num_iter <= PD_IPM_STEPS);
#ifdef AKREON_REPORT_OUTPUT
    STATIC_REQUIRE(trace[res.num_iter - 1].tag ==
                   config::SolverResult::OPTIMAL);
#endif
    STATIC_REQUIRE(!res.stalled);
    STATIC_REQUIRE(constexpr_abs(res.iterate.z(0) - x_unconstrained_exp(0)) <
                   rel_tol);
    STATIC_REQUIRE(constexpr_abs(res.iterate.z(1) - x_unconstrained_exp(1)) <
                   rel_tol);
    STATIC_REQUIRE(constexpr_abs(res.convex_obj) < 1e-2);
  }

  SECTION("L2 Ball Constrained SOCP") {
    constexpr auto r = 1.0;
    constexpr auto ball =
        NormBall<NormBallMeta<Space>, decltype(x), NormBallType::L2>(
            Vectord<2>{}, r, x);
    constexpr auto prob = prob_init.in_set(ball);
    constexpr auto prob_min = prob.minimize(half_square);

    STATIC_REQUIRE(is_problem_set_v<decltype(prob_min)>);
    STATIC_REQUIRE(prob_min.validate());
    STATIC_REQUIRE(
        std::tuple_size<decltype(prob_min.constraint_cones())>::value == 3);

    constexpr auto F = prob_min.constraint_map().linear_part();
    STATIC_REQUIRE(mat_rows_v<decltype(F)> == 9);
    STATIC_REQUIRE(mat_cols_v<decltype(F)> == 4);

    constexpr auto F_ball_mat = mat_traits<Matrixd<9, 4>>();
    constexpr auto F_ball_exp = F_ball_mat(
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, -inv_sqrt2, 0.0, 0.0, 0.0, inv_sqrt2);
    STATIC_REQUIRE(all_equal(F, F_ball_exp));

    constexpr auto g = prob_min.constraint_map().translation();
    STATIC_REQUIRE(vec_size_v<decltype(g)> == 9);
    STATIC_REQUIRE(g == Vectord<9>{0.0, 0.0, 1.0, -1.0, 2.0, 0.0, 0.0,
                                   0.5 * inv_sqrt2, 0.5 * inv_sqrt2});

    constexpr auto c = prob_min.linear_objective();
    STATIC_REQUIRE(c == Vectord<4>{0.0, 0.0, 0.0, 0.5});

    constexpr PrimalDualIPM solver(prob_min);
    constexpr auto res = solver.res;
#ifdef AKREON_REPORT_OUTPUT
    constexpr auto trace = solver.trace;
#endif

    STATIC_REQUIRE(res.tag == SolverResult::OPTIMAL);
    STATIC_REQUIRE(res.num_iter > 0 && res.num_iter <= PD_IPM_STEPS);
#ifdef AKREON_REPORT_OUTPUT
    STATIC_REQUIRE(trace[res.num_iter - 1].tag ==
                   config::SolverResult::OPTIMAL);
#endif
    STATIC_REQUIRE(!res.stalled);
    STATIC_REQUIRE(constexpr_abs(res.iterate.z(0) - x_ball_exp(0)) < rel_tol);
    STATIC_REQUIRE(constexpr_abs(res.iterate.z(1) - x_ball_exp(1)) < rel_tol);
    STATIC_REQUIRE(constexpr_abs(res.iterate.z(2) - (sqrt5 - 1.0)) < rel_tol);
    STATIC_REQUIRE(constexpr_abs(res.iterate.z(3) - (6.0 - 2.0 * sqrt5)) <
                   rel_tol);
  }
}

TEST_CASE("Warm Start SOCP Test", "[socp_test]") {
  using Space = Matrixd<1, 2>;
  using Singleton = Matrixd<1, 1>;

  SECTION("Warm start reuses SOCP iterate on L2 ball constrained problem") {
    constexpr auto x = Variable<VarMeta<Space>>();
    constexpr auto x_trans = x.add(Vectord<2>{-1.0, 2.0});
    constexpr auto prob_init = make_problem(x_trans);
    constexpr auto norm =
        Norm<NormMeta<Space, Singleton>, decltype(x_trans), L2<Vectord<2>>>(
            x_trans);
    constexpr auto square = Square(norm);
    constexpr auto half_square = square.mul<0.5>();
    constexpr auto sqrt5 = constexpr_sqrt_safe(5.0);
    constexpr auto x_exp = Vectord<2>{1.0 / sqrt5, -2.0 / sqrt5};

    constexpr auto ball =
        NormBall<NormBallMeta<Space>, decltype(x), NormBallType::L2>(
            Vectord<2>{}, 1.0, x);
    constexpr auto prob = prob_init.in_set(ball);
    constexpr auto prob_min = prob.minimize(half_square);

    constexpr PrimalDualIPM cold_solver(prob_min);
    constexpr auto cold_res = cold_solver.res;
#ifdef AKREON_REPORT_OUTPUT
    constexpr auto cold_trace = cold_solver.trace;
#endif

    constexpr PrimalDualIPM warm_solver(prob_min, cold_res.iterate);
    constexpr auto warm_res = warm_solver.res;
#ifdef AKREON_REPORT_OUTPUT
    constexpr auto warm_trace = warm_solver.trace;
#endif

    STATIC_REQUIRE(cold_res.tag == SolverResult::OPTIMAL);
    STATIC_REQUIRE(warm_res.tag == SolverResult::OPTIMAL);
#ifdef AKREON_REPORT_OUTPUT
    STATIC_REQUIRE(cold_trace[cold_res.num_iter - 1].tag ==
                   config::SolverResult::OPTIMAL);
#endif
#ifdef AKREON_REPORT_OUTPUT
    STATIC_REQUIRE(warm_trace[warm_res.num_iter - 1].tag ==
                   config::SolverResult::OPTIMAL);
#endif
    STATIC_REQUIRE(!warm_res.stalled);
    STATIC_REQUIRE(warm_res.num_iter > 0 && warm_res.num_iter <= PD_IPM_STEPS);
    STATIC_REQUIRE(warm_res.num_iter <= cold_res.num_iter);
    STATIC_REQUIRE(constexpr_abs(warm_res.iterate.z(0) - x_exp(0)) < rel_tol);
    STATIC_REQUIRE(constexpr_abs(warm_res.iterate.z(1) - x_exp(1)) < rel_tol);
    STATIC_REQUIRE(constexpr_abs(warm_res.iterate.z(2) - (sqrt5 - 1.0)) <
                   rel_tol);
    STATIC_REQUIRE(constexpr_abs(warm_res.iterate.z(3) - (6.0 - 2.0 * sqrt5)) <
                   rel_tol);
    STATIC_REQUIRE(
        constexpr_abs(warm_res.iterate.z(0) - cold_res.iterate.z(0)) < rel_tol);
    STATIC_REQUIRE(
        constexpr_abs(warm_res.iterate.z(1) - cold_res.iterate.z(1)) < rel_tol);
  }
}
