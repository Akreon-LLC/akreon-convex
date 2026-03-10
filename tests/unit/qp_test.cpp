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
#include "unit_test_helpers.hpp"

using namespace linalg;
using namespace convex;
using namespace affine;
using namespace conic;
using namespace unit_test;
using namespace config;

static constexpr auto rel_tol{1e-3};

TEST_CASE("Minimal QP 2 Vars", "[qp_test]") {
  using Space = Matrixd<1, 2>;
  constexpr auto x = Variable<VarMeta<Space>>();
  constexpr auto prob_init = make_problem(x);
  constexpr auto r = 1.0;
  constexpr Vectord<2> c{3.0, 1.0};
  constexpr Vectord<2> x_exp{2.29289322, 0.29289322};

  SECTION("Min c^T x s.t. L2 ball") {
    constexpr auto ball =
        NormBall<NormBallMeta<Space>, decltype(x), NormBallType::L2>(c, r, x);
    constexpr auto prob_constr = prob_init.in_set(ball);

    constexpr Vectord<3> g_exp{-c(0), -c(1), r};

    constexpr auto F_mat = mat_traits<Matrixd<3, 2>>();
    constexpr auto F_exp = F_mat(1.0, 0.0, 0.0, 1.0, 0.0, 0.0);

    constexpr Vectord<2> weights{1.0, 1.0};
    constexpr auto cx = weighted_sum(weights, x);
    constexpr auto prob_min = prob_constr.minimize(cx);

    STATIC_REQUIRE(is_problem_set_v<decltype(prob_min)>);
    STATIC_REQUIRE(
        std::tuple_size<decltype(prob_min.constraint_cones())>::value == 1);

    constexpr auto F = prob_min.constraint_map().linear_part();
    STATIC_REQUIRE(mat_rows_v<decltype(F)> == 3);
    STATIC_REQUIRE(mat_cols_v<decltype(F)> == 2);
    STATIC_REQUIRE(all_equal(F, F_exp));

    constexpr auto g = prob_min.constraint_map().translation();
    STATIC_REQUIRE(vec_size_v<decltype(g)> == 3);
    STATIC_REQUIRE(g_exp == g);

    constexpr PrimalDualIPM solver(prob_min);
    constexpr auto res = solver.res;
    constexpr auto trace = solver.trace;

    STATIC_REQUIRE(res.tag == SolverResult::OPTIMAL);
    STATIC_REQUIRE(res.num_iter > 0 && res.num_iter < PD_IPM_STEPS);
    STATIC_REQUIRE(trace[res.num_iter - 1].tag ==
                   config::SolverResult::OPTIMAL);
    STATIC_REQUIRE(!res.stalled);
    STATIC_REQUIRE(constexpr_abs(res.z(0) - x_exp(0)) < rel_tol);
    STATIC_REQUIRE(constexpr_abs(res.z(1) - x_exp(1)) < rel_tol);
  }

  using Singleton = Matrixd<1, 1>;
  SECTION("Min c^T x s.t. ||x|| <= 1") {
    constexpr Vectord<2> weights{1.0, 1.0};
    constexpr auto x_trans = x.add(Vectord<2>{-c(0), -c(1)});

    constexpr auto norm =
        Norm<NormMeta<Space, Singleton>, decltype(x), L2<Vectord<2>>>(x_trans);
    constexpr auto leq =
        Inequality<InequalityType::LEQ, decltype(norm), decltype(r)>(norm, r);
    constexpr auto prob_constr = prob_init.add_constraint(leq);

    constexpr auto cx = weighted_sum(weights, x);
    constexpr auto prob_min = prob_constr.minimize(cx);

    STATIC_REQUIRE(is_problem_set_v<decltype(prob_min)>);
    STATIC_REQUIRE(
        std::tuple_size<decltype(prob_min.constraint_cones())>::value == 2);

    constexpr auto F = prob_min.constraint_map().linear_part();
    STATIC_REQUIRE(mat_rows_v<decltype(F)> == 4);
    STATIC_REQUIRE(mat_cols_v<decltype(F)> == 3);

    constexpr auto F_mat = mat_traits<Matrixd<4, 3>>();
    constexpr auto F_exp =
        F_mat(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0);
    STATIC_REQUIRE(all_equal(F, F_exp));

    constexpr Vectord<4> g_exp{-c(0), -c(1), 0.0, 1.0};
    constexpr auto g = prob_min.constraint_map().translation();
    STATIC_REQUIRE(vec_size_v<decltype(g)> == 4);
    STATIC_REQUIRE(g == g_exp);

    constexpr PrimalDualIPM solver(prob_min);
    constexpr auto res = solver.res;
    constexpr auto trace = solver.trace;

    STATIC_REQUIRE(constexpr_abs(res.z(0) - x_exp(0)) < rel_tol);
    STATIC_REQUIRE(constexpr_abs(res.z(1) - x_exp(1)) < rel_tol);
    STATIC_REQUIRE(res.tag == SolverResult::OPTIMAL);
    STATIC_REQUIRE(res.num_iter > 0 && res.num_iter <= PD_IPM_STEPS);
    STATIC_REQUIRE(trace[res.num_iter - 1].tag ==
                   config::SolverResult::OPTIMAL);
    STATIC_REQUIRE(!res.stalled);
  }
}

TEST_CASE("Minimal QP 1 Var (RSOC)", "[qp_test]") {
  using Space = Matrixd<1, 1>;
  using Singleton = Matrixd<1, 1>;
  constexpr auto x = Variable<VarMeta<Space>>();
  constexpr auto prob_init = make_problem(x);
  constexpr auto x0 = x(0);

  constexpr auto inv_sqrt2 = 1.0 / constexpr_sqrt_safe(2.0);

  SECTION("Min c^T x s.t. (x - 1)^2 <= 5") {
    constexpr auto sq =
        Square<SquareMeta<Singleton>, decltype(x0)>(x0.add(-1.0));
    constexpr auto leq =
        Inequality<InequalityType::LEQ, decltype(sq), double>(sq, 5.0);
    constexpr auto prob_constr = prob_init.add_constraint(leq);

    constexpr auto prob_min = prob_constr.minimize(x0.mul(0.0));

    STATIC_REQUIRE(is_problem_set_v<decltype(prob_min)>);
    STATIC_REQUIRE(
        std::tuple_size<decltype(prob_min.constraint_cones())>::value == 2);

    constexpr mat_traits<Matrixd<4, 2>> F1_mat;
    constexpr auto F_exp =
        F1_mat(1.0, 0.0, 0.0, -inv_sqrt2, 0.0, inv_sqrt2, 0.0, -1.0);
    constexpr auto F = prob_min.constraint_map().linear_part();
    STATIC_REQUIRE(mat_rows_v<decltype(F)> == 4);
    STATIC_REQUIRE(mat_cols_v<decltype(F)> == 2);
    STATIC_REQUIRE(all_equal(F, F_exp));

    constexpr auto g = prob_min.constraint_map().translation();
    STATIC_REQUIRE(vec_size_v<decltype(g)> == 4);
    STATIC_REQUIRE(g ==
                   Vectord<4>{-1.0, 0.5 * inv_sqrt2, 0.5 * inv_sqrt2, 5.0});
    STATIC_REQUIRE(prob_min.validate());

    constexpr PrimalDualIPM solver(prob_min);
    constexpr auto res = solver.res;
    constexpr auto trace = solver.trace;

    STATIC_REQUIRE(res.tag == SolverResult::OPTIMAL);
    STATIC_REQUIRE(res.num_iter > 0 && res.num_iter <= PD_IPM_STEPS);
    STATIC_REQUIRE(trace[res.num_iter - 1].tag ==
                   config::SolverResult::OPTIMAL);
    STATIC_REQUIRE(!res.stalled);
    STATIC_REQUIRE(constexpr_abs(res.z(0) - 1.0) < rel_tol);
  }

#if 0
  SECTION("Min c^T x s.t. 1/x <= 1") {
    constexpr auto inv_pos = InvPos<InvPosMeta<Singleton>, decltype(x0)>(x0);
    constexpr auto leq =
        Inequality<InequalityType::LEQ, decltype(inv_pos), double>(inv_pos,
                                                                   1.0);
    constexpr auto prob_constr = prob_init.add_constraint(leq);

    constexpr auto prob_min = prob_constr.minimize(x0.mul(0.0));

    STATIC_REQUIRE(is_problem_set_v<decltype(prob_min)>);
    STATIC_REQUIRE(
        std::tuple_size<decltype(prob_min.constraint_cones())>::value == 2);

    constexpr auto F = prob_min.constraint_map().linear_part();
    STATIC_REQUIRE(mat_rows_v<decltype(F)> == 4);
    STATIC_REQUIRE(mat_cols_v<decltype(F)> == 2);

    constexpr mat_traits<Matrixd<4, 2>> F2_mat;
    constexpr auto F_exp = F2_mat(0.0, 0.0, inv_sqrt2, -inv_sqrt2,
                                  inv_sqrt2, inv_sqrt2, 0.0, -1.0);
    STATIC_REQUIRE(all_equal(F, F_exp));

    constexpr auto g = prob_min.constraint_map().translation();
    STATIC_REQUIRE(vec_size_v<decltype(g)> == 4);
    STATIC_REQUIRE(
        g == Vectord<4>{constexpr_sqrt_safe(2.0), 0.0, 0.0, 1.0});
    STATIC_REQUIRE(prob_min.validate());

    constexpr PrimalDualIPM solver(prob_min);
    constexpr auto res = solver.res;
    constexpr auto trace = solver.trace;
#if 0
    STATIC_REQUIRE(res.tag == SolverResult::OPTIMAL);
    STATIC_REQUIRE(res.num_iter > 0 && res.num_iter <= PD_IPM_STEPS);
    STATIC_REQUIRE(trace[res.num_iter - 1].tag ==
                   config::SolverResult::OPTIMAL);
    STATIC_REQUIRE(!res.stalled);
    STATIC_REQUIRE(constexpr_abs(res.z(0) - 1.0) < rel_tol);
#endif
    log(res, trace);
  }
#endif
}
