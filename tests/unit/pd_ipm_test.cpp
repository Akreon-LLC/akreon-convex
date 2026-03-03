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
#include <iostream>

#include "math.hpp"
#include "unit_test_helpers.hpp"

using namespace linalg;
using namespace convex;
using namespace affine;
using namespace conic;
using namespace unit_test;
using namespace config;

/**
 * @brief Simple LP UT
 * A x = b
 * A = [1 1; 2 1; -1 2]
 * b = [4 5 2]
 *
 * (1) min x0 s.t. A x = b
 * (2) min x1 s.t. A x = b
 * (3) min x0 + x1 s.t. A x = b
 * (4) min c^T x s.t. A x = b
 */
TEST_CASE("Minimal LP 2 Vars", "[pd_ipm_test]") {
  using Space = Matrixd<1, 2>;
  using Codom = Matrixd<1, 3>;
  constexpr auto x = Variable<NonnegVarMeta<Space>>();
  constexpr auto prob_init = make_problem(x);

  constexpr auto A_mat = mat_traits<Matrixd<3, 2>>();
  constexpr auto A = A_mat(1.0, 1.0, 2.0, 1.0, -1.0, 2.0);
  constexpr auto b = Vectord<3>{4.0, 5.0, 2.0};
  constexpr Polyhedron<PolyhedronMeta<Space, Codom>, decltype(x)> poly(A, b, x);
  constexpr auto prob_constr = prob_init.in_set(poly);

  constexpr auto F_mat = mat_traits<Matrixd<5, 2>>();
  constexpr auto F_exp =
      F_mat(1.0, 0.0, 0.0, 1.0, -1.0, -1.0, -2.0, -1.0, 1.0, -2.0);

  constexpr auto g_exp = Vectord<5>{0.0, 0.0, 4.0, 5.0, 2.0};

  constexpr auto res_tol = 1e-2;

  SECTION("Min x0") {
    constexpr auto prob_min = prob_constr.minimize(x(0).value());
    STATIC_REQUIRE(is_problem_set_v<decltype(prob_min)>);
    STATIC_REQUIRE(
        std::tuple_size<decltype(prob_min.constraint_cones())>::value == 2);
    constexpr auto F = prob_min.constraint_map().linear_part();
    STATIC_REQUIRE(mat_rows_v<decltype(F)> == 5);
    STATIC_REQUIRE(mat_cols_v<decltype(F)> == 2);

    STATIC_REQUIRE(all_equal(F_exp, F));

    constexpr auto g = prob_min.constraint_map().translation();
    STATIC_REQUIRE(g == g_exp);

    constexpr auto c = prob_min.linear_objective();
    STATIC_REQUIRE(c == Vectord<2>{1.0, 0.0});

    constexpr PrimalDualIPM solver(prob_min);
    constexpr auto res = solver.res;
    constexpr auto trace = solver.trace;
    STATIC_REQUIRE(constexpr_abs(res.z(0)) < res_tol);
    STATIC_REQUIRE(res.tag == SolverResult::OPTIMAL);
    STATIC_REQUIRE(res.num_iter > 0 && res.num_iter < PD_IPM_STEPS);
    STATIC_REQUIRE(trace[res.num_iter - 1].tag ==
                   config::SolverResult::OPTIMAL);
  }

  SECTION("Min x1") {
    constexpr auto prob_min = prob_constr.minimize(x(1).value());
    STATIC_REQUIRE(is_problem_set_v<decltype(prob_min)>);
    STATIC_REQUIRE(
        std::tuple_size<decltype(prob_min.constraint_cones())>::value == 2);
    constexpr auto F = prob_min.constraint_map().linear_part();
    STATIC_REQUIRE(mat_rows_v<decltype(F)> == 5);
    STATIC_REQUIRE(mat_cols_v<decltype(F)> == 2);

    STATIC_REQUIRE(all_equal(F_exp, F));

    constexpr auto g = prob_min.constraint_map().translation();
    STATIC_REQUIRE(g == g_exp);

    constexpr auto c = prob_min.linear_objective();
    STATIC_REQUIRE(c == Vectord<2>{0.0, 1.0});

    constexpr PrimalDualIPM solver(prob_min);
    constexpr auto res = solver.res;
    constexpr auto trace = solver.trace;
    STATIC_REQUIRE(constexpr_abs(res.z(1)) < res_tol);
    STATIC_REQUIRE(res.tag == SolverResult::OPTIMAL);
    STATIC_REQUIRE(res.num_iter > 0 && res.num_iter < PD_IPM_STEPS);
    STATIC_REQUIRE(trace[res.num_iter - 1].tag ==
                   config::SolverResult::OPTIMAL);
  }

  SECTION("Min sum(x.split())") {
    static constexpr auto x_tup = x.split();
    constexpr auto x_sum =
        Sum<SumMeta<Space, Matrixd<1, 1>>, decltype(std::get<0>(x_tup)),
            decltype(std::get<1>(x_tup))>(std::get<0>(x_tup),
                                          std::get<1>(x_tup));

    constexpr auto prob_min = prob_constr.minimize(x_sum);
    STATIC_REQUIRE(is_problem_set_v<decltype(prob_min)>);
    STATIC_REQUIRE(
        std::tuple_size<decltype(prob_min.constraint_cones())>::value == 2);
    constexpr auto F = prob_min.constraint_map().linear_part();
    STATIC_REQUIRE(mat_rows_v<decltype(F)> == 5);
    STATIC_REQUIRE(mat_cols_v<decltype(F)> == 2);

    STATIC_REQUIRE(all_equal(F_exp, F));

    constexpr auto g = prob_min.constraint_map().translation();
    STATIC_REQUIRE(g == g_exp);

    constexpr auto c = prob_min.linear_objective();
    STATIC_REQUIRE(c == Vectord<2>{1.0, 1.0});

    constexpr PrimalDualIPM solver(prob_min);
    constexpr auto res = solver.res;
    constexpr auto trace = solver.trace;
    STATIC_REQUIRE(res.tag == SolverResult::OPTIMAL);
    STATIC_REQUIRE(res.num_iter > 0 && res.num_iter < PD_IPM_STEPS);
    STATIC_REQUIRE(trace[res.num_iter - 1].tag ==
                   config::SolverResult::OPTIMAL);
  }

  SECTION("Min c^T x") {
    constexpr auto c = Vectord<2>{2.0, -1.0};
    constexpr auto cx = weighted_sum(c, x);
    constexpr auto prob_min = prob_constr.minimize(cx);
    STATIC_REQUIRE(is_problem_set_v<decltype(prob_min)>);
    STATIC_REQUIRE(
        std::tuple_size<decltype(prob_min.constraint_cones())>::value == 2);
    constexpr auto F = prob_min.constraint_map().linear_part();
    STATIC_REQUIRE(mat_rows_v<decltype(F)> == 5);
    STATIC_REQUIRE(mat_cols_v<decltype(F)> == 2);

    STATIC_REQUIRE(all_equal(F_exp, F));

    constexpr auto g = prob_min.constraint_map().translation();
    STATIC_REQUIRE(g == g_exp);

    constexpr auto obj = prob_min.linear_objective();
    STATIC_REQUIRE(obj == c);

    constexpr PrimalDualIPM solver(prob_min);
    constexpr auto res = solver.res;
    constexpr auto trace = solver.trace;
    STATIC_REQUIRE(res.tag == SolverResult::OPTIMAL);
    STATIC_REQUIRE(res.num_iter > 0 && res.num_iter < PD_IPM_STEPS);
    STATIC_REQUIRE(trace[res.num_iter - 1].tag ==
                   config::SolverResult::OPTIMAL);

#if 0
    char buf[64000];
    trace_to_cstr(res.num_iter, trace, buf, sizeof(buf));
    std::cout << buf << std::endl;

    memset(buf, '\0', sizeof(buf));
    result_to_cstr(res, buf, sizeof(buf));
    std::cout << buf << std::endl;
#endif
  }
}

/**
 * @brief Simple QP
 *
 * (1) c^T x s.t. ||x|| <= 1
 */
TEST_CASE("Minimal QP 2 Vars", "[pd_ipm_test]") {
  using Space = Matrixd<1, 2>;
  constexpr auto x = Variable<FreeVarMeta<Space>>();
  constexpr auto prob_init = make_problem(x);
  constexpr auto r = 1.0;
  constexpr Vectord<2> c{3.0, 1.0};
  constexpr auto ball =
      NormBall<NormBallMeta<Space>, decltype(x), NormBallType::L2>(c, r, x);
  constexpr auto prob_constr = prob_init.in_set(ball);

  SECTION("Min c^T x") {
    constexpr Vectord<2> weights{1.0, 1.0};
    constexpr auto cx = weighted_sum(weights, x);
    constexpr auto prob_min = prob_constr.minimize(cx);

    constexpr PrimalDualIPM solver(prob_min);
    constexpr auto res = solver.res;
    constexpr auto trace = solver.trace;
#if 0
    STATIC_REQUIRE(res.tag == SolverResult::PRIMAL_OPTIMAL);
    STATIC_REQUIRE(res.num_iter > 0 && res.num_iter < PD_IPM_STEPS);
    STATIC_REQUIRE(trace[res.num_iter - 1].tag ==
                   config::SolverResult::PRIMAL_OPTIMAL);
    STATIC_REQUIRE(res.stalled);
#endif

    char buf[64000];
    trace_to_cstr(res.num_iter, trace, buf, sizeof(buf));
    std::cout << buf << std::endl;

    memset(buf, '\0', sizeof(buf));
    result_to_cstr(res, buf, sizeof(buf));
    std::cout << buf << std::endl;
  }
}

/**
 * @brief Simple SOCP
 *
 * (1) min 0.5 ||x + a||_2^2
 * (2) min 0.5 ||x + a||_2^2
 *     s.t. ||x|| <= 1
 */
#if 0
TEST_CASE("Minimal SOCP 2 Vars", "[pd_ipm_test]") {
  using Space = Matrixd<1, 2>;
  using Singleton = Matrixd<1, 1>;
  constexpr auto x = Variable<FreeVarMeta<Space>>();
  constexpr auto x_trans = x.add(Vectord<2>{-1.0, 2.0});
  constexpr auto prob_init = make_problem(x_trans);
  constexpr auto norm =
      Norm<NormMeta<Space, Singleton>, decltype(x_trans), L2<Vectord<2>>>(
          x_trans);
  constexpr auto square = Square<SquareMeta<Singleton>, decltype(norm)>(norm);
  constexpr auto half_square = square.mul<0.5>();

  constexpr auto F_mat = mat_traits<Matrixd<6, 4>>();
  constexpr auto F_exp =
      F_mat(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
  constexpr Vectord<6> g_exp{-1.0, 2.0, 0.0, 0.0, 0.5, 0.0};

  SECTION("Unconstrained SOCP") {
    constexpr auto prob_min = prob_init.minimize(half_square);

    STATIC_REQUIRE(is_problem_set_v<decltype(prob_min)>);
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

    constexpr PrimalDualIPM solver(prob_min);

    constexpr auto res = solver.res;
    constexpr auto trace = solver.trace;
    STATIC_REQUIRE(res.num_iter > 0);
    STATIC_REQUIRE(res.tag != SolverResult::INVALID);
    // STATIC_REQUIRE(trace[res.num_iter - 1].tag ==
    // SolverResult::PRIMAL_FEASIBLE);
    char buf[64000];
    trace_to_cstr(res.num_iter, trace, buf, sizeof(buf));
    std::cout << buf << std::endl;

    memset(buf, '\0', sizeof(buf));
    result_to_cstr(res, buf, sizeof(buf));
    std::cout << buf << std::endl;
  }

  SECTION("L2 Ball Constrained SOCP") {
    constexpr auto r = 1.0;
    constexpr auto ball =
        NormBall<NormBallMeta<Space>, decltype(x), NormBallType::L2>(
            Vectord<2>{}, r, x);
    constexpr auto prob = prob_init.in_set(ball);
    constexpr auto prob_min = prob.minimize(half_square);

    STATIC_REQUIRE(is_problem_set_v<decltype(prob_min)>);

#if 0
    STATIC_REQUIRE(
        std::tuple_size<decltype(prob_min.constraint_cones())>::value == 2);

    constexpr auto F = prob_min.constraint_map().linear_part();
    STATIC_REQUIRE(mat_rows_v<decltype(F)> == 6);
    STATIC_REQUIRE(mat_cols_v<decltype(F)> == 4);
    STATIC_REQUIRE(all_equal(F, F_exp));

    constexpr auto g = prob_min.constraint_map().translation();
    STATIC_REQUIRE(vec_size_v<decltype(g)> == 6);
    STATIC_REQUIRE(g == g_exp);
#endif

    constexpr auto c = prob_min.linear_objective();
    STATIC_REQUIRE(c == Vectord<4>{0.0, 0.0, 0.0, 0.5});

    constexpr PrimalDualIPM solver(prob_min);

    constexpr auto res = solver.res;
    constexpr auto trace = solver.trace;
    STATIC_REQUIRE(res.num_iter > 0);
    STATIC_REQUIRE(res.tag != SolverResult::INVALID);
    STATIC_REQUIRE(trace[res.num_iter - 1].tag ==
                   SolverResult::PRIMAL_RESIDUAL_FEASIBLE_CONE_VIOLATION);
    char buf[16000];
    trace_to_cstr(res.num_iter, trace, buf, sizeof(buf));
    std::cout << buf << std::endl;

    memset(buf, '\0', sizeof(buf));
    result_to_cstr(res, buf, sizeof(buf));
    std::cout << buf << std::endl;
  }
}
#endif
