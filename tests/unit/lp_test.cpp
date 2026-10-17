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

constexpr double rel_tol = 1e-3;

TEST_CASE("LP canonicalizes and solves polyhedron objectives", "[lp_test]") {
  using Space = Matrixd<1, 2>;
  constexpr auto x = Variable<VarMeta<Space>>();
  constexpr auto prob_init = make_problem(x);

  constexpr auto A_mat = mat_traits<Matrixd<5, 2>>();
  constexpr auto A = A_mat(-1.0, 0.0, 0.0, -1.0, 1.0, 1.0, 2.0, 1.0, -1.0, 2.0);
  constexpr auto b = Vectord<5>{0.0, 0.0, 4.0, 5.0, 2.0};
  constexpr auto poly = Polyhedron(A, b, x);
  constexpr auto prob_constr = prob_init.in_set(poly);

  constexpr auto F_mat = mat_traits<Matrixd<5, 2>>();
  constexpr auto F_exp =
      F_mat(1.0, 0.0, 0.0, 1.0, -1.0, -1.0, -2.0, -1.0, 1.0, -2.0);

  constexpr auto g_exp = Vectord<5>{0.0, 0.0, 4.0, 5.0, 2.0};

  SECTION("Min x0") {
    constexpr auto prob_min = prob_constr.minimize(x(0));
    STATIC_REQUIRE(is_problem_set_v<decltype(prob_min)>);
    STATIC_REQUIRE(
        std::tuple_size<decltype(prob_min.constraint_cones())>::value == 1);
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

    STATIC_REQUIRE(constexpr_abs(res.iterate.z(0)) < rel_tol);
    STATIC_REQUIRE(res.tag == SolverResult::OPTIMAL);
    STATIC_REQUIRE(res.num_iter > 0 && res.num_iter < PD_IPM_STEPS);

#if AKREON_REPORT_OUTPUT
    constexpr auto trace = solver.trace;
    STATIC_REQUIRE(trace[res.num_iter - 1].tag ==
                   config::SolverResult::OPTIMAL);
#endif
  }

  SECTION("Min x1") {
    constexpr auto prob_min = prob_constr.minimize(x(1));
    STATIC_REQUIRE(is_problem_set_v<decltype(prob_min)>);
    STATIC_REQUIRE(
        std::tuple_size<decltype(prob_min.constraint_cones())>::value == 1);
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

    STATIC_REQUIRE(constexpr_abs(res.iterate.z(1)) < rel_tol);
    STATIC_REQUIRE(res.tag == SolverResult::OPTIMAL);
    STATIC_REQUIRE(res.num_iter > 0 && res.num_iter < PD_IPM_STEPS);
#ifdef AKREON_REPORT_OUTPUT
    constexpr auto trace = solver.trace;
    STATIC_REQUIRE(trace[res.num_iter - 1].tag ==
                   config::SolverResult::OPTIMAL);
#endif
  }

  SECTION("Min sum(x.split())") {
    static constexpr auto x_tup = x.split();
    constexpr auto x_sum = Sum(std::get<0>(x_tup), std::get<1>(x_tup));

    constexpr auto prob_min = prob_constr.minimize(x_sum);
    STATIC_REQUIRE(is_problem_set_v<decltype(prob_min)>);
    STATIC_REQUIRE(
        std::tuple_size<decltype(prob_min.constraint_cones())>::value == 1);
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

    STATIC_REQUIRE(res.tag == SolverResult::OPTIMAL);
    STATIC_REQUIRE(res.num_iter > 0 && res.num_iter < PD_IPM_STEPS);
#ifdef AKREON_REPORT_OUTPUT
    constexpr auto trace = solver.trace;
    STATIC_REQUIRE(trace[res.num_iter - 1].tag ==
                   config::SolverResult::OPTIMAL);
#endif
    STATIC_REQUIRE(res.iterate.z(0) < rel_tol);
    STATIC_REQUIRE(res.iterate.z(1) < rel_tol);
  }

  SECTION("Min c^T x") {
    constexpr auto c = Vectord<2>{2.0, -1.0};
    constexpr auto cx = WeightedSum<Space>()(c, x);
    constexpr auto prob_min = prob_constr.minimize(cx);
    STATIC_REQUIRE(is_problem_set_v<decltype(prob_min)>);
    STATIC_REQUIRE(
        std::tuple_size<decltype(prob_min.constraint_cones())>::value == 1);
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

    STATIC_REQUIRE(res.tag == SolverResult::OPTIMAL);
    STATIC_REQUIRE(res.num_iter > 0 && res.num_iter < PD_IPM_STEPS);
#ifdef AKREON_REPORT_OUTPUT
    constexpr auto trace = solver.trace;
    STATIC_REQUIRE(trace[res.num_iter - 1].tag ==
                   config::SolverResult::OPTIMAL);
#endif
    STATIC_REQUIRE(res.iterate.z(0) < rel_tol);
    STATIC_REQUIRE(linalg::constexpr_abs(res.iterate.z(1) - 1.0) < rel_tol);
  }
}

TEST_CASE("LP solves intersected halfspace and hyperplane constraints",
          "[lp_test]") {
  using Space = Matrixd<1, 2>;

  constexpr auto x = Variable<VarMeta<Space>>();
  constexpr auto prob_init = make_problem(x);

  constexpr mat_traits<Matrixd<2, 2>> A_nonneg_mat;
  constexpr auto A_nonneg = A_nonneg_mat(-1.0, 0.0, 0.0, -1.0);
  constexpr auto b_nonneg = Vectord<2>{0.0, 0.0};
  constexpr auto nonneg = Polyhedron(A_nonneg, b_nonneg, x);

  constexpr Halfspace<HalfspaceMeta<Space>, decltype(x), HalfspaceType::LH>
      x0_upper_bound(Vectord<2>{1.0, 0.0}, 0.5, x);
  constexpr auto simplex_face = Hyperplane(Vectord<2>{1.0, 1.0}, 1.0, x);

  constexpr auto feasible_set =
      Intersection(nonneg, x0_upper_bound, simplex_face);
  constexpr auto prob_constr = prob_init.in_set(feasible_set);

  constexpr auto c = Vectord<2>{2.0, -1.0};
  constexpr auto cx = WeightedSum<Space>()(c, x);
  constexpr auto prob_min = prob_constr.minimize(cx);

  constexpr mat_traits<Matrixd<3, 2>> F_mat;
  constexpr auto F_exp = F_mat(1.0, 0.0, 0.0, 1.0, -1.0, 0.0);
  constexpr auto g_exp = Vectord<3>{0.0, 0.0, 0.5};
  constexpr mat_traits<Matrixd<1, 2>> Aeq_mat;
  constexpr auto Aeq = Aeq_mat(1.0, 1.0);

  STATIC_REQUIRE(is_problem_set_v<decltype(prob_min)>);
  STATIC_REQUIRE(prob_min.validate());
  STATIC_REQUIRE(
      std::tuple_size<decltype(prob_min.constraint_cones())>::value == 2);
  STATIC_REQUIRE(
      mat_rows_v<decltype(prob_min.constraint_map().linear_part())> == 3);
  STATIC_REQUIRE(
      mat_cols_v<decltype(prob_min.constraint_map().linear_part())> == 2);
  STATIC_REQUIRE(all_equal(prob_min.constraint_map().linear_part(), F_exp));
  STATIC_REQUIRE(prob_min.constraint_map().translation() == g_exp);
  STATIC_REQUIRE(all_equal(prob_min.equality_map().linear_part(), Aeq));
  STATIC_REQUIRE(prob_min.equality_map().translation() == Vectord<1>{1.0});
  STATIC_REQUIRE(prob_min.linear_objective() == c);
}

TEST_CASE("LP solves intersected polyhedron and affine subspace constraints",
          "[lp_test]") {
  using Space = Matrixd<1, 2>;

  constexpr auto x = Variable<VarMeta<Space>>();
  constexpr auto prob_init = make_problem(x);

  constexpr mat_traits<Matrixd<3, 2>> A_poly_mat;
  constexpr auto A_poly = A_poly_mat(-1.0, 0.0, 0.0, -1.0, 1.0, 0.0);
  constexpr auto b_poly = Vectord<3>{0.0, 0.0, 0.5};
  constexpr auto poly = Polyhedron(A_poly, b_poly, x);

  constexpr auto A_aff = make_identity<double, 2>();
  constexpr auto b_aff = Vectord<2>{0.0, 1.0};
  constexpr auto affine_set = AffineSubspace(A_aff, b_aff, x);

  constexpr auto feasible_set = Intersection(poly, affine_set);
  constexpr auto prob_constr = prob_init.in_set(feasible_set);

  constexpr auto c = Vectord<2>{2.0, -1.0};
  constexpr auto cx = WeightedSum<Space>()(c, x);
  constexpr auto prob_min = prob_constr.minimize(cx);

  constexpr mat_traits<Matrixd<3, 2>> F_mat;
  constexpr auto F_exp = F_mat(1.0, 0.0, 0.0, 1.0, -1.0, 0.0);
  constexpr auto g_exp = Vectord<3>{0.0, 0.0, 0.5};

  STATIC_REQUIRE(is_problem_set_v<decltype(prob_min)>);
  STATIC_REQUIRE(prob_min.validate());
  STATIC_REQUIRE(
      std::tuple_size<decltype(prob_min.constraint_cones())>::value == 1);
  STATIC_REQUIRE(
      mat_rows_v<decltype(prob_min.constraint_map().linear_part())> == 3);
  STATIC_REQUIRE(
      mat_cols_v<decltype(prob_min.constraint_map().linear_part())> == 2);
  STATIC_REQUIRE(all_equal(prob_min.constraint_map().linear_part(), F_exp));
  STATIC_REQUIRE(prob_min.constraint_map().translation() == g_exp);
  STATIC_REQUIRE(all_equal(prob_min.equality_map().linear_part(),
                           make_identity<double, 2>()));
  STATIC_REQUIRE(prob_min.equality_map().translation() == Vectord<2>{0.0, 1.0});
  STATIC_REQUIRE(prob_min.linear_objective() == c);

  constexpr PrimalDualIPM solver(prob_min);
  constexpr auto res = solver.res;

  STATIC_REQUIRE(res.tag == SolverResult::OPTIMAL);
  STATIC_REQUIRE(res.num_iter > 0 && res.num_iter < PD_IPM_STEPS);
#ifdef AKREON_REPORT_OUTPUT
  constexpr auto trace = solver.trace;
  STATIC_REQUIRE(trace[res.num_iter - 1].tag == config::SolverResult::OPTIMAL);
#endif
  STATIC_REQUIRE(linalg::constexpr_abs(res.iterate.z(0)) < rel_tol);
  STATIC_REQUIRE(linalg::constexpr_abs(res.iterate.z(1) - 1.0) < rel_tol);
}

TEST_CASE("LP minimizes a linear objective over an L1 ball", "[lp_test]") {
  using Space = Matrixd<1, 2>;

  constexpr auto x = Variable<VarMeta<Space>>();
  constexpr auto prob_init = make_problem(x);

  constexpr auto center = Vectord<2>{0.0, 0.0};
  constexpr auto radius = 1.5;
  constexpr auto ball =
      NormBall<NormBallMeta<Space>, decltype(x), NormBallType::L1>(center,
                                                                   radius, x);
  constexpr auto prob_constr = prob_init.in_set(ball);

  constexpr auto c = Vectord<2>{2.0, -1.0};
  constexpr auto cx = WeightedSum<Space>()(c, x);
  constexpr auto prob_min = prob_constr.minimize(cx);

  constexpr mat_traits<Matrixd<7, 4>> F_mat;
  constexpr auto F_exp = F_mat(
      1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 0.0, 0.0, -1.0,
      0.0, 1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
  constexpr auto g_exp = Vectord<7>{0.0, 0.0, 0.0, 0.0, radius, 0.0, 0.0};
  constexpr auto obj_exp = Vectord<4>{2.0, -1.0, 0.0, 0.0};

  STATIC_REQUIRE(is_problem_set_v<decltype(prob_min)>);
  STATIC_REQUIRE(prob_min.validate());
  STATIC_REQUIRE(
      std::tuple_size<decltype(prob_min.constraint_cones())>::value == 4);
  STATIC_REQUIRE(
      mat_rows_v<decltype(prob_min.constraint_map().linear_part())> == 7);
  STATIC_REQUIRE(
      mat_cols_v<decltype(prob_min.constraint_map().linear_part())> == 4);
  STATIC_REQUIRE(all_equal(prob_min.constraint_map().linear_part(), F_exp));
  STATIC_REQUIRE(prob_min.constraint_map().translation() == g_exp);
  STATIC_REQUIRE(prob_min.linear_objective() == obj_exp);

  constexpr PrimalDualIPM solver(prob_min);
  constexpr auto res = solver.res;

  STATIC_REQUIRE(res.tag == SolverResult::OPTIMAL);
  STATIC_REQUIRE(res.num_iter > 0 && res.num_iter < PD_IPM_STEPS);
#ifdef AKREON_REPORT_OUTPUT
  constexpr auto trace = solver.trace;
  STATIC_REQUIRE(trace[res.num_iter - 1].tag == config::SolverResult::OPTIMAL);
#endif
  STATIC_REQUIRE(linalg::constexpr_abs(res.iterate.z(0) + radius) < rel_tol);
  STATIC_REQUIRE(linalg::constexpr_abs(res.iterate.z(1)) < rel_tol);
}

TEST_CASE("LP minimizes a linear objective over a Linf ball", "[lp_test]") {
  using Space = Matrixd<1, 2>;

  constexpr auto x = Variable<VarMeta<Space>>();
  constexpr auto prob_init = make_problem(x);

  constexpr auto center = Vectord<2>{0.0, 0.0};
  constexpr auto radius = 1.5;
  constexpr auto ball =
      NormBall<NormBallMeta<Space>, decltype(x), NormBallType::Linf>(center,
                                                                     radius, x);
  constexpr auto prob_constr = prob_init.in_set(ball);

  constexpr auto c = Vectord<2>{2.0, -1.0};
  constexpr auto cx = WeightedSum<Space>()(c, x);
  constexpr auto prob_min = prob_constr.minimize(cx);

  constexpr mat_traits<Matrixd<4, 2>> F_mat;
  constexpr auto F_exp = F_mat(1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0);
  constexpr auto g_exp = Vectord<4>{radius, radius, radius, radius};

  STATIC_REQUIRE(is_problem_set_v<decltype(prob_min)>);
  STATIC_REQUIRE(prob_min.validate());
  STATIC_REQUIRE(
      std::tuple_size<decltype(prob_min.constraint_cones())>::value == 2);
  STATIC_REQUIRE(
      mat_rows_v<decltype(prob_min.constraint_map().linear_part())> == 4);
  STATIC_REQUIRE(
      mat_cols_v<decltype(prob_min.constraint_map().linear_part())> == 2);
  STATIC_REQUIRE(all_equal(prob_min.constraint_map().linear_part(), F_exp));
  STATIC_REQUIRE(prob_min.constraint_map().translation() == g_exp);
  STATIC_REQUIRE(prob_min.linear_objective() == c);

  constexpr PrimalDualIPM solver(prob_min);
  constexpr auto res = solver.res;

  STATIC_REQUIRE(res.tag == SolverResult::OPTIMAL);
  STATIC_REQUIRE(res.num_iter > 0 && res.num_iter < PD_IPM_STEPS);
#ifdef AKREON_REPORT_OUTPUT
  constexpr auto trace = solver.trace;
  STATIC_REQUIRE(trace[res.num_iter - 1].tag == config::SolverResult::OPTIMAL);
#endif
  STATIC_REQUIRE(linalg::constexpr_abs(res.iterate.z(0) + radius) < rel_tol);
  STATIC_REQUIRE(linalg::constexpr_abs(res.iterate.z(1) - radius) < rel_tol);
}

TEST_CASE("LP solves an L1 epigraph with a lower bound", "[lp_test]") {
  using Space = Matrixd<1, 2>;
  using Codom = Matrixd<1, 1>;

  constexpr auto x = Variable<VarMeta<Space>>();
  constexpr auto prob_init = make_problem(x);
  constexpr auto x0 = x(0);
  constexpr auto geq =
      Inequality<InequalityType::GEQ, decltype(x0), double>(x0, 1.0);
  constexpr auto prob_constr = prob_init.add_constraint(geq);

  constexpr auto l1 =
      Norm<NormMeta<Space, Codom>, decltype(x), L1<Vectord<2>>>(x);
  constexpr auto prob_min = prob_constr.minimize(l1);

  constexpr mat_traits<Matrixd<8, 5>> F_mat;
  constexpr auto F_exp = F_mat(
      1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0,
      -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0,
      1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
  constexpr auto g_exp = Vectord<8>{-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  constexpr auto obj_exp = Vectord<5>{0.0, 0.0, 0.0, 0.0, 1.0};

  STATIC_REQUIRE(is_problem_set_v<decltype(prob_min)>);
  STATIC_REQUIRE(prob_min.validate());
  STATIC_REQUIRE(
      std::tuple_size<decltype(prob_min.constraint_cones())>::value == 5);
  STATIC_REQUIRE(
      mat_rows_v<decltype(prob_min.constraint_map().linear_part())> == 8);
  STATIC_REQUIRE(
      mat_cols_v<decltype(prob_min.constraint_map().linear_part())> == 5);
  STATIC_REQUIRE(all_equal(prob_min.constraint_map().linear_part(), F_exp));
  STATIC_REQUIRE(prob_min.constraint_map().translation() == g_exp);
  STATIC_REQUIRE(prob_min.linear_objective() == obj_exp);

  constexpr PrimalDualIPM solver(prob_min);
  constexpr auto res = solver.res;

  STATIC_REQUIRE(res.tag == SolverResult::OPTIMAL);
  STATIC_REQUIRE(res.num_iter > 0 && res.num_iter < PD_IPM_STEPS);
#ifdef AKREON_REPORT_OUTPUT
  constexpr auto trace = solver.trace;
  STATIC_REQUIRE(trace[res.num_iter - 1].tag == config::SolverResult::OPTIMAL);
#endif
  STATIC_REQUIRE(linalg::constexpr_abs(res.iterate.z(0) - 1.0) < rel_tol);
  STATIC_REQUIRE(linalg::constexpr_abs(res.iterate.z(1)) < rel_tol);
  STATIC_REQUIRE(linalg::constexpr_abs(res.iterate.z(4) - 1.0) < rel_tol);
}

TEST_CASE("LP solves a Linf epigraph with a lower bound", "[lp_test]") {
  using Space = Matrixd<1, 2>;
  using Codom = Matrixd<1, 1>;

  constexpr auto x = Variable<VarMeta<Space>>();
  constexpr auto prob_init = make_problem(x);
  constexpr auto x1 = x(1);
  constexpr auto leq =
      Inequality<InequalityType::LEQ, decltype(x1), double>(x1, -2.0);
  constexpr auto prob_constr = prob_init.add_constraint(leq);

  constexpr auto linf =
      Norm<NormMeta<Space, Codom>, decltype(x), Linf<Vectord<2>>>(x);
  constexpr auto prob_min = prob_constr.minimize(linf);

  constexpr mat_traits<Matrixd<5, 3>> F_mat;
  constexpr auto F_exp = F_mat(0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0,
                               -1.0, 0.0, 1.0, 0.0, -1.0, 1.0);
  constexpr auto g_exp = Vectord<5>{-2.0, 0.0, 0.0, 0.0, 0.0};
  constexpr auto obj_exp = Vectord<3>{0.0, 0.0, 1.0};

  STATIC_REQUIRE(is_problem_set_v<decltype(prob_min)>);
  STATIC_REQUIRE(prob_min.validate());
  STATIC_REQUIRE(
      std::tuple_size<decltype(prob_min.constraint_cones())>::value == 3);
  STATIC_REQUIRE(
      mat_rows_v<decltype(prob_min.constraint_map().linear_part())> == 5);
  STATIC_REQUIRE(
      mat_cols_v<decltype(prob_min.constraint_map().linear_part())> == 3);
  STATIC_REQUIRE(all_equal(prob_min.constraint_map().linear_part(), F_exp));
  STATIC_REQUIRE(prob_min.constraint_map().translation() == g_exp);
  STATIC_REQUIRE(prob_min.linear_objective() == obj_exp);

  constexpr PrimalDualIPM solver(prob_min);
  constexpr auto res = solver.res;

  STATIC_REQUIRE(res.tag == SolverResult::OPTIMAL);
  STATIC_REQUIRE(res.num_iter > 0 && res.num_iter < PD_IPM_STEPS);
#ifdef AKREON_REPORT_OUTPUT
  constexpr auto trace = solver.trace;
  STATIC_REQUIRE(trace[res.num_iter - 1].tag == config::SolverResult::OPTIMAL);
#endif
  STATIC_REQUIRE(linalg::constexpr_abs(res.iterate.z(1) + 2.0) < rel_tol);
  STATIC_REQUIRE(linalg::constexpr_abs(res.iterate.z(2) - 2.0) < rel_tol);
  STATIC_REQUIRE(linalg::constexpr_abs(res.iterate.z(0)) <= 2.0 + rel_tol);
}

TEST_CASE("LP minimizes Max with componentwise lower bounds", "[lp_test]") {
  using Space = Matrixd<1, 2>;
  constexpr auto x = Variable<VarMeta<Space>>();
  constexpr auto prob_init = make_problem(x);
  constexpr auto x0 = x(0);
  constexpr auto x1 = x(1);
  constexpr auto x0_geq =
      Inequality<InequalityType::GEQ, decltype(x0), double>(x0, 1.5);
  constexpr auto x1_geq =
      Inequality<InequalityType::GEQ, decltype(x1), double>(x1, 1.5);
  constexpr auto prob_constr =
      prob_init.add_constraint(x0_geq).add_constraint(x1_geq);

  constexpr auto mx = Max(x);
  constexpr auto prob_min = prob_constr.minimize(mx);

  constexpr mat_traits<Matrixd<4, 3>> F_mat;
  constexpr auto F_exp =
      F_mat(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 1.0);
  constexpr auto g_exp = Vectord<4>{-1.5, -1.5, 0.0, 0.0};
  constexpr auto obj_exp = Vectord<3>{0.0, 0.0, 1.0};

  STATIC_REQUIRE(is_problem_set_v<decltype(prob_min)>);
  STATIC_REQUIRE(prob_min.validate());
  STATIC_REQUIRE(
      std::tuple_size<decltype(prob_min.constraint_cones())>::value == 3);
  STATIC_REQUIRE(
      mat_rows_v<decltype(prob_min.constraint_map().linear_part())> == 4);
  STATIC_REQUIRE(
      mat_cols_v<decltype(prob_min.constraint_map().linear_part())> == 3);
  STATIC_REQUIRE(all_equal(prob_min.constraint_map().linear_part(), F_exp));
  STATIC_REQUIRE(prob_min.constraint_map().translation() == g_exp);
  STATIC_REQUIRE(prob_min.linear_objective() == obj_exp);

  constexpr PrimalDualIPM solver(prob_min);
  constexpr auto res = solver.res;

  STATIC_REQUIRE(res.tag == SolverResult::OPTIMAL);
  STATIC_REQUIRE(res.num_iter > 0 && res.num_iter < PD_IPM_STEPS);
#ifdef AKREON_REPORT_OUTPUT
  constexpr auto trace = solver.trace;
  STATIC_REQUIRE(trace[res.num_iter - 1].tag == config::SolverResult::OPTIMAL);
#endif
  STATIC_REQUIRE(linalg::constexpr_abs(res.iterate.z(0) - 1.5) < rel_tol);
  STATIC_REQUIRE(linalg::constexpr_abs(res.iterate.z(1) - 1.5) < rel_tol);
}

TEST_CASE("LP canonicalizes and solves Sum equality and inequality constraints",
          "[lp_test]") {
  using Space = Matrixd<1, 2>;

  SECTION("sum(x) == 1") {
    constexpr auto x = Variable<VarMeta<Space>>();
    constexpr auto prob_init = make_problem(x);
    constexpr mat_traits<Matrixd<2, 2>> A_nonneg_mat;
    constexpr auto A_nonneg = A_nonneg_mat(-1.0, 0.0, 0.0, -1.0);
    constexpr auto b_nonneg = Vectord<2>{0.0, 0.0};
    constexpr auto nonneg = Polyhedron(A_nonneg, b_nonneg, x);
    constexpr auto sum_eq_one = Hyperplane(Vectord<2>{1.0, 1.0}, 1.0, x);
    constexpr auto feasible_set = Intersection(nonneg, sum_eq_one);
    constexpr auto prob_constr = prob_init.in_set(feasible_set);

    constexpr auto c = Vectord<2>{2.0, -1.0};
    constexpr auto cx = WeightedSum<Space>()(c, x);
    constexpr auto prob_min = prob_constr.minimize(cx);

    constexpr mat_traits<Matrixd<2, 2>> F_mat;
    constexpr auto F_exp = F_mat(1.0, 0.0, 0.0, 1.0);
    constexpr auto g_exp = Vectord<2>{0.0, 0.0};
    constexpr mat_traits<Matrixd<1, 2>> Aeq_mat;
    constexpr auto Aeq = Aeq_mat(1.0, 1.0);

    STATIC_REQUIRE(is_problem_set_v<decltype(prob_min)>);
    STATIC_REQUIRE(prob_min.validate());
    STATIC_REQUIRE(
        std::tuple_size<decltype(prob_min.constraint_cones())>::value == 1);
    STATIC_REQUIRE(
        mat_rows_v<decltype(prob_min.constraint_map().linear_part())> == 2);
    STATIC_REQUIRE(
        mat_cols_v<decltype(prob_min.constraint_map().linear_part())> == 2);
    STATIC_REQUIRE(all_equal(prob_min.constraint_map().linear_part(), F_exp));
    STATIC_REQUIRE(prob_min.constraint_map().translation() == g_exp);
    STATIC_REQUIRE(all_equal(prob_min.equality_map().linear_part(), Aeq));
    STATIC_REQUIRE(prob_min.equality_map().translation() == Vectord<1>{1.0});
    STATIC_REQUIRE(prob_min.linear_objective() == c);

    constexpr PrimalDualIPM solver(prob_min);
    constexpr auto res = solver.res;

    STATIC_REQUIRE(res.tag == SolverResult::OPTIMAL);
    STATIC_REQUIRE(res.num_iter > 0 && res.num_iter < PD_IPM_STEPS);
#ifdef AKREON_REPORT_OUTPUT
    constexpr auto trace = solver.trace;
    STATIC_REQUIRE(trace[res.num_iter - 1].tag ==
                   config::SolverResult::OPTIMAL);
#endif
    STATIC_REQUIRE(linalg::constexpr_abs(res.iterate.z(0)) < rel_tol);
    STATIC_REQUIRE(linalg::constexpr_abs(res.iterate.z(1) - 1.0) < rel_tol);
  }

  SECTION("sum(x) <= 1") {
    constexpr auto x = Variable<VarMeta<Space>>();
    constexpr auto prob_init = make_problem(x);
    constexpr mat_traits<Matrixd<3, 2>> A_poly_mat;
    constexpr auto A_poly = A_poly_mat(-1.0, 0.0, 0.0, -1.0, 1.0, 1.0);
    constexpr auto b_poly = Vectord<3>{0.0, 0.0, 1.0};
    constexpr auto poly = Polyhedron(A_poly, b_poly, x);
    constexpr auto prob_constr = prob_init.in_set(poly);

    constexpr auto c = Vectord<2>{2.0, -1.0};
    constexpr auto cx = WeightedSum<Space>()(c, x);
    constexpr auto prob_min = prob_constr.minimize(cx);

    constexpr mat_traits<Matrixd<3, 2>> F_mat;
    constexpr auto F_exp = F_mat(1.0, 0.0, 0.0, 1.0, -1.0, -1.0);
    constexpr auto g_exp = Vectord<3>{0.0, 0.0, 1.0};

    STATIC_REQUIRE(is_problem_set_v<decltype(prob_min)>);
    STATIC_REQUIRE(prob_min.validate());
    STATIC_REQUIRE(
        std::tuple_size<decltype(prob_min.constraint_cones())>::value == 1);
    STATIC_REQUIRE(
        mat_rows_v<decltype(prob_min.constraint_map().linear_part())> == 3);
    STATIC_REQUIRE(
        mat_cols_v<decltype(prob_min.constraint_map().linear_part())> == 2);
    STATIC_REQUIRE(all_equal(prob_min.constraint_map().linear_part(), F_exp));
    STATIC_REQUIRE(prob_min.constraint_map().translation() == g_exp);
    STATIC_REQUIRE(prob_min.linear_objective() == c);

    constexpr PrimalDualIPM solver(prob_min);
    constexpr auto res = solver.res;

    STATIC_REQUIRE(res.tag == SolverResult::OPTIMAL);
    STATIC_REQUIRE(res.num_iter > 0 && res.num_iter < PD_IPM_STEPS);
#ifdef AKREON_REPORT_OUTPUT
    constexpr auto trace = solver.trace;
    STATIC_REQUIRE(trace[res.num_iter - 1].tag ==
                   config::SolverResult::OPTIMAL);
#endif
    STATIC_REQUIRE(linalg::constexpr_abs(res.iterate.z(0)) < rel_tol);
    STATIC_REQUIRE(linalg::constexpr_abs(res.iterate.z(1) - 1.0) < rel_tol);
  }
}

TEST_CASE("LP minimizes Abs with a lower bound", "[lp_test]") {
  using Space = Matrixd<1, 1>;

  constexpr auto x = Variable<VarMeta<Space>>();
  constexpr auto prob_init = make_problem(x);
  constexpr auto abs_x = Abs(x);
  constexpr auto x0 = x(0);
  constexpr auto geq =
      Inequality<InequalityType::GEQ, decltype(x0), double>(x0, 1.0);
  constexpr auto prob_constr = prob_init.add_constraint(geq);
  constexpr auto prob_min = prob_constr.minimize(abs_x);

  constexpr auto obj_exp = Vectord<3>{0.0, 0.0, 1.0};

  STATIC_REQUIRE(is_problem_set_v<decltype(prob_min)>);
  STATIC_REQUIRE(prob_min.validate());
  STATIC_REQUIRE(
      std::tuple_size<decltype(prob_min.constraint_cones())>::value == 6);
  STATIC_REQUIRE(
      mat_rows_v<decltype(prob_min.constraint_map().linear_part())> == 6);
  STATIC_REQUIRE(
      mat_cols_v<decltype(prob_min.constraint_map().linear_part())> == 3);
  STATIC_REQUIRE(prob_min.linear_objective() == obj_exp);

  constexpr PrimalDualIPM solver(prob_min);
  constexpr auto res = solver.res;

  STATIC_REQUIRE(res.tag == SolverResult::OPTIMAL);
  STATIC_REQUIRE(res.num_iter > 0 && res.num_iter < PD_IPM_STEPS);
#ifdef AKREON_REPORT_OUTPUT
  constexpr auto trace = solver.trace;
  STATIC_REQUIRE(trace[res.num_iter - 1].tag == config::SolverResult::OPTIMAL);
#endif
  STATIC_REQUIRE(linalg::constexpr_abs(res.iterate.z(0) - 1.0) < rel_tol);

  // log(res, trace);
}

TEST_CASE("LP warm start reuses an iterate on a polyhedron", "[lp_test]") {
  using Space = Matrixd<1, 2>;

  constexpr auto x = Variable<VarMeta<Space>>();
  constexpr auto prob_init = make_problem(x);

  constexpr auto A_mat = mat_traits<Matrixd<5, 2>>();
  constexpr auto A = A_mat(-1.0, 0.0, 0.0, -1.0, 1.0, 1.0, 2.0, 1.0, -1.0, 2.0);
  constexpr auto b = Vectord<5>{0.0, 0.0, 4.0, 5.0, 2.0};
  constexpr auto poly = Polyhedron(A, b, x);
  constexpr auto prob_constr = prob_init.in_set(poly);

  constexpr auto c = Vectord<2>{2.0, -1.0};
  constexpr auto cx = WeightedSum<Space>()(c, x);
  constexpr auto prob_min = prob_constr.minimize(cx);

  constexpr PrimalDualIPM cold_solver(prob_min);
  constexpr auto cold_res = cold_solver.res;

  constexpr PrimalDualIPM warm_solver(prob_min, cold_res.iterate);
  constexpr auto warm_res = warm_solver.res;

  STATIC_REQUIRE(cold_res.tag == SolverResult::OPTIMAL);
  STATIC_REQUIRE(warm_res.tag == SolverResult::OPTIMAL);
#ifdef AKREON_REPORT_OUTPUT
  constexpr auto cold_trace = cold_solver.trace;
  constexpr auto warm_trace = warm_solver.trace;
  STATIC_REQUIRE(cold_trace[cold_res.num_iter - 1].tag ==
                 config::SolverResult::OPTIMAL);
  STATIC_REQUIRE(warm_trace[warm_res.num_iter - 1].tag ==
                 config::SolverResult::OPTIMAL);
#endif
  STATIC_REQUIRE(warm_res.num_iter > 0 && warm_res.num_iter < PD_IPM_STEPS);
  STATIC_REQUIRE(warm_res.num_iter <= cold_res.num_iter);
  STATIC_REQUIRE(constexpr_abs(warm_res.iterate.z(0)) < rel_tol);
  STATIC_REQUIRE(constexpr_abs(warm_res.iterate.z(1) - 1.0) < rel_tol);
  STATIC_REQUIRE(constexpr_abs(warm_res.iterate.z(0) - cold_res.iterate.z(0)) <
                 rel_tol);
  STATIC_REQUIRE(constexpr_abs(warm_res.iterate.z(1) - cold_res.iterate.z(1)) <
                 rel_tol);
}
