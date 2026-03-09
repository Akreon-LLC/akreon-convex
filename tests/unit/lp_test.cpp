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

TEST_CASE("Polyhedron Test", "[lp_test]") {
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
  constexpr auto rel_tol = 1e-3;

  SECTION("Min x0") {
    constexpr auto prob_min = prob_constr.minimize(x(0));
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

    STATIC_REQUIRE(constexpr_abs(res.z(0)) < rel_tol);
    STATIC_REQUIRE(res.tag == SolverResult::OPTIMAL);
    STATIC_REQUIRE(res.num_iter > 0 && res.num_iter < PD_IPM_STEPS);
    STATIC_REQUIRE(trace[res.num_iter - 1].tag ==
                   config::SolverResult::OPTIMAL);
  }

  SECTION("Min x1") {
    constexpr auto prob_min = prob_constr.minimize(x(1));
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

    STATIC_REQUIRE(constexpr_abs(res.z(1)) < rel_tol);
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
    STATIC_REQUIRE(res.z(0) < rel_tol);
    STATIC_REQUIRE(res.z(1) < rel_tol);
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
    STATIC_REQUIRE(res.z(0) < rel_tol);
    STATIC_REQUIRE(linalg::constexpr_abs(res.z(1) - 1.0) < rel_tol);
  }
}

TEST_CASE("Halfspace and Hyperplane LP Test", "[lp_test]") {
  using Space = Matrixd<1, 2>;

  SECTION("Free Var With Scalar GEQ") {
    constexpr auto x = Variable<NonnegVarMeta<Space>>();
    constexpr auto prob_init = make_problem(x);

    constexpr Halfspace<HalfspaceMeta<Space>, decltype(x), HalfspaceType::LH>
        x0_upper_bound(Vectord<2>{1.0, 0.0}, 0.5, x);
    constexpr Hyperplane<HyperplaneMeta<Space>, decltype(x)> simplex_face(
        Vectord<2>{1.0, 1.0}, 1.0, x);

    constexpr auto feasible_set =
        Intersection<IntersectionMeta<Space>, decltype(x0_upper_bound),
                     decltype(simplex_face)>(x0_upper_bound, simplex_face);
    constexpr auto prob_constr = prob_init.in_set(feasible_set);

    constexpr auto c = Vectord<2>{2.0, -1.0};
    constexpr auto cx = weighted_sum(c, x);
    constexpr auto prob_min = prob_constr.minimize(cx);

    constexpr mat_traits<Matrixd<5, 2>> F_mat;
    constexpr auto F_exp =
        F_mat(1.0, 0.0, 0.0, 1.0, -1.0, 0.0, -1.0, -1.0, 1.0, 1.0);
    constexpr auto g_exp = Vectord<5>{0.0, 0.0, 0.5, 1.0, -1.0};

    STATIC_REQUIRE(is_problem_set_v<decltype(prob_min)>);
    STATIC_REQUIRE(prob_min.validate());
    STATIC_REQUIRE(
        std::tuple_size<decltype(prob_min.constraint_cones())>::value == 3);
    STATIC_REQUIRE(
        mat_rows_v<decltype(prob_min.constraint_map().linear_part())> == 5);
    STATIC_REQUIRE(
        mat_cols_v<decltype(prob_min.constraint_map().linear_part())> == 2);
    STATIC_REQUIRE(all_equal(prob_min.constraint_map().linear_part(), F_exp));
    STATIC_REQUIRE(prob_min.constraint_map().translation() == g_exp);
    STATIC_REQUIRE(prob_min.linear_objective() == c);
  }
}

TEST_CASE("Polyhedron and Affine Subspace LP Test", "[lp_test]") {
  using Space = Matrixd<1, 2>;
  using PolyCodom = Matrixd<1, 1>;

  constexpr auto rel_tol = 1e-3;

  SECTION("Nonneg Var") {
    constexpr auto x = Variable<NonnegVarMeta<Space>>();
    constexpr auto prob_init = make_problem(x);

    constexpr mat_traits<Matrixd<1, 2>> A_poly_mat;
    constexpr auto A_poly = A_poly_mat(1.0, 0.0);
    constexpr auto b_poly = Vectord<1>{0.5};
    constexpr Polyhedron<PolyhedronMeta<Space, PolyCodom>, decltype(x)> poly(
        A_poly, b_poly, x);

    constexpr auto A_aff = make_identity<double, 2>();
    constexpr auto b_aff = Vectord<2>{0.0, 1.0};
    constexpr auto affine_set =
        AffineSubspace<AffineMeta<Space, Space>, decltype(x)>(A_aff, b_aff, x);

    constexpr auto feasible_set =
        Intersection<IntersectionMeta<Space>, decltype(poly),
                     decltype(affine_set)>(poly, affine_set);
    constexpr auto prob_constr = prob_init.in_set(feasible_set);

    constexpr auto c = Vectord<2>{2.0, -1.0};
    constexpr auto cx = weighted_sum(c, x);
    constexpr auto prob_min = prob_constr.minimize(cx);

    constexpr mat_traits<Matrixd<7, 2>> F_mat;
    constexpr auto F_exp = F_mat(1.0, 0.0, 0.0, 1.0, -1.0, 0.0, -1.0, 0.0, 0.0,
                                 -1.0, 1.0, 0.0, 0.0, 1.0);
    constexpr auto g_exp = Vectord<7>{0.0, 0.0, 0.5, 0.0, 1.0, 0.0, -1.0};

    STATIC_REQUIRE(is_problem_set_v<decltype(prob_min)>);
    STATIC_REQUIRE(prob_min.validate());
    STATIC_REQUIRE(
        std::tuple_size<decltype(prob_min.constraint_cones())>::value == 3);
    STATIC_REQUIRE(
        mat_rows_v<decltype(prob_min.constraint_map().linear_part())> == 7);
    STATIC_REQUIRE(
        mat_cols_v<decltype(prob_min.constraint_map().linear_part())> == 2);
    STATIC_REQUIRE(all_equal(prob_min.constraint_map().linear_part(), F_exp));
    STATIC_REQUIRE(prob_min.constraint_map().translation() == g_exp);
    STATIC_REQUIRE(prob_min.linear_objective() == c);

    constexpr PrimalDualIPM solver(prob_min);
    constexpr auto res = solver.res;
    constexpr auto trace = solver.trace;

    STATIC_REQUIRE(res.tag == SolverResult::OPTIMAL);
    STATIC_REQUIRE(res.num_iter > 0 && res.num_iter < PD_IPM_STEPS);
    STATIC_REQUIRE(trace[res.num_iter - 1].tag ==
                   config::SolverResult::OPTIMAL);
    STATIC_REQUIRE(linalg::constexpr_abs(res.z(0)) < rel_tol);
    STATIC_REQUIRE(linalg::constexpr_abs(res.z(1) - 1.0) < rel_tol);
  }
}

TEST_CASE("L1 Ball LP Test", "[lp_test]") {
  using Space = Matrixd<1, 2>;

  constexpr auto rel_tol = 1e-3;

  SECTION("Free Var") {
    constexpr auto x = Variable<FreeVarMeta<Space>>();
    constexpr auto prob_init = make_problem(x);

    constexpr auto center = Vectord<2>{0.0, 0.0};
    constexpr auto radius = 1.5;
    constexpr auto ball =
        NormBall<NormBallMeta<Space>, decltype(x), NormBallType::L1>(center,
                                                                     radius, x);
    constexpr auto prob_constr = prob_init.in_set(ball);

    constexpr auto c = Vectord<2>{2.0, -1.0};
    constexpr auto cx = weighted_sum(c, x);
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
    constexpr auto trace = solver.trace;

    STATIC_REQUIRE(res.tag == SolverResult::OPTIMAL);
    STATIC_REQUIRE(res.num_iter > 0 && res.num_iter < PD_IPM_STEPS);
    STATIC_REQUIRE(trace[res.num_iter - 1].tag ==
                   config::SolverResult::OPTIMAL);
    STATIC_REQUIRE(linalg::constexpr_abs(res.z(0) + radius) < rel_tol);
    STATIC_REQUIRE(linalg::constexpr_abs(res.z(1)) < rel_tol);
  }
}

TEST_CASE("Linf Ball LP Test", "[lp_test]") {
  using Space = Matrixd<1, 2>;

  constexpr auto rel_tol = 1e-3;

  SECTION("Free Var") {
    constexpr auto x = Variable<FreeVarMeta<Space>>();
    constexpr auto prob_init = make_problem(x);

    constexpr auto center = Vectord<2>{0.0, 0.0};
    constexpr auto radius = 1.5;
    constexpr auto ball =
        NormBall<NormBallMeta<Space>, decltype(x), NormBallType::Linf>(
            center, radius, x);
    constexpr auto prob_constr = prob_init.in_set(ball);

    constexpr auto c = Vectord<2>{2.0, -1.0};
    constexpr auto cx = weighted_sum(c, x);
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
    constexpr auto trace = solver.trace;

    STATIC_REQUIRE(res.tag == SolverResult::OPTIMAL);
    STATIC_REQUIRE(res.num_iter > 0 && res.num_iter < PD_IPM_STEPS);
    STATIC_REQUIRE(trace[res.num_iter - 1].tag ==
                   config::SolverResult::OPTIMAL);
    STATIC_REQUIRE(linalg::constexpr_abs(res.z(0) + radius) < rel_tol);
    STATIC_REQUIRE(linalg::constexpr_abs(res.z(1) - radius) < rel_tol);
  }
}

TEST_CASE("L1 Function LP Test", "[lp_test]") {
  using Space = Matrixd<1, 2>;
  using Codom = Matrixd<1, 1>;

  constexpr auto rel_tol = 1e-3;

  SECTION("Free Var With Scalar GEQ") {
    constexpr auto x = Variable<FreeVarMeta<Space>>();
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
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0,
        0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0,
        -1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
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
    constexpr auto trace = solver.trace;

    STATIC_REQUIRE(res.tag == SolverResult::OPTIMAL);
    STATIC_REQUIRE(res.num_iter > 0 && res.num_iter < PD_IPM_STEPS);
    STATIC_REQUIRE(trace[res.num_iter - 1].tag ==
                   config::SolverResult::OPTIMAL);
    STATIC_REQUIRE(linalg::constexpr_abs(res.z(0) - 1.0) < rel_tol);
    STATIC_REQUIRE(linalg::constexpr_abs(res.z(1)) < rel_tol);
    STATIC_REQUIRE(linalg::constexpr_abs(res.z(4) - 1.0) < rel_tol);
  }
}

TEST_CASE("Linf Function LP Test", "[lp_test]") {
  using Space = Matrixd<1, 2>;
  using Codom = Matrixd<1, 1>;

  constexpr auto rel_tol = 1e-3;

  SECTION("Free Var With Scalar LEQ") {
    constexpr auto x = Variable<FreeVarMeta<Space>>();
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
    constexpr auto trace = solver.trace;

    STATIC_REQUIRE(res.tag == SolverResult::OPTIMAL);
    STATIC_REQUIRE(res.num_iter > 0 && res.num_iter < PD_IPM_STEPS);
    STATIC_REQUIRE(trace[res.num_iter - 1].tag ==
                   config::SolverResult::OPTIMAL);
    STATIC_REQUIRE(linalg::constexpr_abs(res.z(1) + 2.0) < rel_tol);
    STATIC_REQUIRE(linalg::constexpr_abs(res.z(2) - 2.0) < rel_tol);
    STATIC_REQUIRE(linalg::constexpr_abs(res.z(0)) <= 2.0 + rel_tol);
  }
}

TEST_CASE("Max Function LP Test", "[lp_test]") {
  using Space = Matrixd<1, 2>;
  using Codom = Matrixd<1, 1>;

  SECTION("Nonneg Var With Scalar GEQs") {
    constexpr auto x = Variable<NonnegVarMeta<Space>>();
    constexpr auto prob_init = make_problem(x);
    constexpr auto x0 = x(0);
    constexpr auto x1 = x(1);
    constexpr auto x0_geq =
        Inequality<InequalityType::GEQ, decltype(x0), double>(x0, 1.5);
    constexpr auto x1_geq =
        Inequality<InequalityType::GEQ, decltype(x1), double>(x1, 1.5);
    constexpr auto prob_constr =
        prob_init.add_constraint(x0_geq).add_constraint(x1_geq);

    constexpr auto mx = Max<MaxMeta<Space, Codom>, decltype(x)>(x);
    constexpr auto prob_min = prob_constr.minimize(mx);

    constexpr mat_traits<Matrixd<6, 3>> F_mat;
    constexpr auto F_exp = F_mat(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0,
                                 0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 1.0);
    constexpr auto g_exp = Vectord<6>{0.0, 0.0, -1.5, -1.5, 0.0, 0.0};
    constexpr auto obj_exp = Vectord<3>{0.0, 0.0, 1.0};

    STATIC_REQUIRE(is_problem_set_v<decltype(prob_min)>);
    STATIC_REQUIRE(prob_min.validate());
    STATIC_REQUIRE(
        std::tuple_size<decltype(prob_min.constraint_cones())>::value == 5);
    STATIC_REQUIRE(
        mat_rows_v<decltype(prob_min.constraint_map().linear_part())> == 6);
    STATIC_REQUIRE(
        mat_cols_v<decltype(prob_min.constraint_map().linear_part())> == 3);
    STATIC_REQUIRE(all_equal(prob_min.constraint_map().linear_part(), F_exp));
    STATIC_REQUIRE(prob_min.constraint_map().translation() == g_exp);
    STATIC_REQUIRE(prob_min.linear_objective() == obj_exp);

    constexpr PrimalDualIPM solver(prob_min);
    constexpr auto res = solver.res;
    constexpr auto trace = solver.trace;

    log(res, trace);
  }
}

TEST_CASE("Sum Constraint LP Test", "[lp_test]") {
  using Space = Matrixd<1, 2>;

  constexpr auto rel_tol = 1e-3;

  /// (dakre) TODO - Previous zero order cone -> now two-sided Rn+ doesnt have
  /// as clean of a KKT as emplacing the equality blocks directly. However
  /// this requires a much greater refactor as it impacts modeling & solver
  /// layers
  SECTION("sum(x) == 1") {
    constexpr auto x = Variable<NonnegVarMeta<Space>>();
    constexpr auto prob_init = make_problem(x);
    constexpr Hyperplane<HyperplaneMeta<Space>, decltype(x)> sum_eq_one(
        Vectord<2>{1.0, 1.0}, 1.0, x);
    constexpr auto prob_constr = prob_init.in_set(sum_eq_one);

    constexpr auto c = Vectord<2>{2.0, -1.0};
    constexpr auto cx = weighted_sum(c, x);
    constexpr auto prob_min = prob_constr.minimize(cx);

    constexpr mat_traits<Matrixd<4, 2>> F_mat;
    constexpr auto F_exp = F_mat(1.0, 0.0, 0.0, 1.0, -1.0, -1.0, 1.0, 1.0);
    constexpr auto g_exp = Vectord<4>{0.0, 0.0, 1.0, -1.0};

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

    constexpr PrimalDualIPMConfig<double> config{.dual_feas_tol = 2e-3};
    constexpr PrimalDualIPM solver(prob_min, config);
    constexpr auto res = solver.res;
    constexpr auto trace = solver.trace;

    STATIC_REQUIRE(res.tag == SolverResult::OPTIMAL);
    STATIC_REQUIRE(res.num_iter > 0 && res.num_iter < PD_IPM_STEPS);
    STATIC_REQUIRE(trace[res.num_iter - 1].tag ==
                   config::SolverResult::OPTIMAL);
    STATIC_REQUIRE(linalg::constexpr_abs(res.z(0)) < rel_tol);
    STATIC_REQUIRE(linalg::constexpr_abs(res.z(1) - 1.0) < rel_tol);
  }

  SECTION("sum(x) <= 1") {
    constexpr auto x = Variable<NonnegVarMeta<Space>>();
    constexpr auto prob_init = make_problem(x);
    constexpr Halfspace<HalfspaceMeta<Space>, decltype(x), HalfspaceType::LH>
        sum_leq_one(Vectord<2>{1.0, 1.0}, 1.0, x);
    constexpr auto prob_constr = prob_init.in_set(sum_leq_one);

    constexpr auto c = Vectord<2>{2.0, -1.0};
    constexpr auto cx = weighted_sum(c, x);
    constexpr auto prob_min = prob_constr.minimize(cx);

    constexpr mat_traits<Matrixd<3, 2>> F_mat;
    constexpr auto F_exp = F_mat(1.0, 0.0, 0.0, 1.0, -1.0, -1.0);
    constexpr auto g_exp = Vectord<3>{0.0, 0.0, 1.0};

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
    STATIC_REQUIRE(prob_min.linear_objective() == c);

    constexpr PrimalDualIPMConfig<double> config{.dual_feas_tol = 1e-3};
    constexpr PrimalDualIPM solver(prob_min, config);
    constexpr auto res = solver.res;
    constexpr auto trace = solver.trace;

    STATIC_REQUIRE(res.tag == SolverResult::OPTIMAL);
    STATIC_REQUIRE(res.num_iter > 0 && res.num_iter < PD_IPM_STEPS);
    STATIC_REQUIRE(trace[res.num_iter - 1].tag ==
                   config::SolverResult::OPTIMAL);
    STATIC_REQUIRE(linalg::constexpr_abs(res.z(0)) < rel_tol);
    STATIC_REQUIRE(linalg::constexpr_abs(res.z(1) - 1.0) < rel_tol);
  }
}

TEST_CASE("Abs Function LP Test", "[lp_test]") {
  using Space = Matrixd<1, 1>;

  constexpr auto rel_tol = 1e-3;

  SECTION("Nonneg Var") {
    constexpr auto x = Variable<FreeVarMeta<Space>>();
    constexpr auto prob_init = make_problem(x);
    constexpr auto abs_x = Abs<AbsMeta<Space>, decltype(x)>(x);
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

    constexpr PrimalDualIPMConfig<double> config{.primal_feas_tol = 1e-3};
    constexpr PrimalDualIPM solver(prob_min, config);
    constexpr auto res = solver.res;
    constexpr auto trace = solver.trace;

    STATIC_REQUIRE(res.tag == SolverResult::OPTIMAL);
    STATIC_REQUIRE(res.num_iter > 0 && res.num_iter < PD_IPM_STEPS);
    STATIC_REQUIRE(trace[res.num_iter - 1].tag ==
                   config::SolverResult::OPTIMAL);
    STATIC_REQUIRE(linalg::constexpr_abs(res.z(0) - 1.0) < rel_tol);
  }
}
