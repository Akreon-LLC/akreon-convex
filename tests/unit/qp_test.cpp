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
  constexpr auto x = Variable<NonnegVarMeta<Space>>();
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

  SECTION("Min c^T x s.t. ellipsoid") {
    constexpr auto P = r * r * make_identity<double, 2>();
    constexpr auto ellipsoid =
        Ellipsoid<EllipsoidMeta<Space>, decltype(x)>(c, P, x);
    constexpr auto prob_constr = prob_init.in_set(ellipsoid);

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

TEST_CASE("Minimal QP 2 Vars (Square Objective)", "[qp_test]") {
  using Space = Matrixd<1, 2>;
  using Singleton = Matrixd<1, 1>;
  constexpr auto x = Variable<PosVarMeta<Space>>();
  constexpr auto prob_init = make_problem(x);
  constexpr auto x0 = x(0);
  constexpr auto x1 = x(1);
  constexpr auto inv_sqrt2 = 1.0 / constexpr_sqrt_safe(2.0);

  SECTION("Min square(x0 - 1) s.t. x0 >= 0, x1 == 0") {
    constexpr auto x0_geq =
        Inequality<InequalityType::GEQ, decltype(x0), double>(x0, 0.0);
    constexpr auto x1_eq =
        Inequality<InequalityType::EQ, decltype(x1), double>(x1, 0.0);
    constexpr auto prob_constr =
        prob_init.add_constraint(x0_geq).add_constraint(x1_eq);
    constexpr auto sq =
        Square<SquareMeta<Singleton>, decltype(x0)>(x0.add(-1.0));
    constexpr auto prob_min = prob_constr.minimize(sq);

    STATIC_REQUIRE(is_problem_set_v<decltype(prob_min)>);
    STATIC_REQUIRE(prob_min.validate());
    STATIC_REQUIRE(
        std::tuple_size<decltype(prob_min.constraint_cones())>::value == 3);

    constexpr mat_traits<Matrixd<6, 3>> F_mat;
    constexpr auto F_exp =
        F_mat(1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0,
              0.0, -inv_sqrt2, 0.0, 0.0, inv_sqrt2);
    constexpr auto F = prob_min.constraint_map().linear_part();
    STATIC_REQUIRE(mat_rows_v<decltype(F)> == 6);
    STATIC_REQUIRE(mat_cols_v<decltype(F)> == 3);
    STATIC_REQUIRE(all_equal(F, F_exp));

    constexpr auto g = prob_min.constraint_map().translation();
    STATIC_REQUIRE(vec_size_v<decltype(g)> == 6);
    STATIC_REQUIRE(
        g == Vectord<6>{0.0, 0.0, 0.0, -1.0, 0.5 * inv_sqrt2, 0.5 * inv_sqrt2});
    STATIC_REQUIRE(prob_min.linear_objective() == Vectord<3>{0.0, 0.0, 1.0});

    constexpr PrimalDualIPM solver(prob_min);
    constexpr auto res = solver.res;
    constexpr auto trace = solver.trace;

    STATIC_REQUIRE(res.tag == SolverResult::OPTIMAL);
    STATIC_REQUIRE(res.num_iter > 0 && res.num_iter <= PD_IPM_STEPS);
    STATIC_REQUIRE(trace[res.num_iter - 1].tag ==
                   config::SolverResult::OPTIMAL);
    STATIC_REQUIRE(!res.stalled);
    STATIC_REQUIRE(constexpr_abs(res.z(0) - 1.0) < rel_tol);
    STATIC_REQUIRE(constexpr_abs(res.z(1)) < rel_tol);
    STATIC_REQUIRE(constexpr_abs(res.z(2)) < rel_tol);
  }
}

TEST_CASE("Minimal QP 1 Var (Square Root Objective)", "[qp_test]") {
  using Space = Matrixd<1, 1>;
  using Singleton = Matrixd<1, 1>;
  constexpr auto x = Variable<PosVarMeta<Space>>();
  constexpr auto prob_init = make_problem(x);
  constexpr auto x0 = x(0);
  constexpr auto inv_sqrt2 = 1.0 / constexpr_sqrt_safe(2.0);

  SECTION("Max sqrt(x0) s.t. 0 <= x0 <= 1") {
    constexpr auto x0_leq =
        Inequality<InequalityType::LEQ, decltype(x0), double>(x0, 1.0);
    constexpr auto x0_geq =
        Inequality<InequalityType::GEQ, decltype(x0), double>(x0, 0.0);
    constexpr auto prob_constr =
        prob_init.add_constraint(x0_leq).add_constraint(x0_geq);
    constexpr auto sqrt_x0 = Sqrt<SqrtMeta<Singleton>, decltype(x0)>(x0);
    constexpr auto prob_max = prob_constr.maximize(sqrt_x0);

    STATIC_REQUIRE(is_problem_set_v<decltype(prob_max)>);
    STATIC_REQUIRE(prob_max.validate());
    STATIC_REQUIRE(
        std::tuple_size<decltype(prob_max.constraint_cones())>::value == 3);

    constexpr mat_traits<Matrixd<5, 2>> F_mat;
    constexpr auto F_exp =
        F_mat(-1.0, 0.0, 1.0, 0.0, 0.0, -1.0, -inv_sqrt2, 0.0, inv_sqrt2, 0.0);
    constexpr auto F = prob_max.constraint_map().linear_part();
    STATIC_REQUIRE(mat_rows_v<decltype(F)> == 5);
    STATIC_REQUIRE(mat_cols_v<decltype(F)> == 2);
    STATIC_REQUIRE(all_equal(F, F_exp));

    constexpr auto g = prob_max.constraint_map().translation();
    STATIC_REQUIRE(vec_size_v<decltype(g)> == 5);
    STATIC_REQUIRE(g ==
                   Vectord<5>{1.0, 0.0, 0.0, 0.5 * inv_sqrt2, 0.5 * inv_sqrt2});
    STATIC_REQUIRE(prob_max.linear_objective() == Vectord<2>{0.0, 1.0});

    constexpr PrimalDualIPM solver(prob_max);
    constexpr auto res = solver.res;
    constexpr auto trace = solver.trace;

    STATIC_REQUIRE(res.tag == SolverResult::OPTIMAL);
    STATIC_REQUIRE(res.num_iter > 0 && res.num_iter <= PD_IPM_STEPS);
    STATIC_REQUIRE(trace[res.num_iter - 1].tag ==
                   config::SolverResult::OPTIMAL);
    STATIC_REQUIRE(!res.stalled);
    STATIC_REQUIRE(constexpr_abs(res.z(0) - 1.0) < rel_tol);
    STATIC_REQUIRE(constexpr_abs(res.z(1) + 1.0) < rel_tol);
  }
}

TEST_CASE("Minimal QP 2 Vars (Harmonic Mean Objective)", "[qp_test]") {
  using Space = Matrixd<1, 2>;
  using Singleton = Matrixd<1, 1>;
  constexpr auto x = Variable<VarMeta<Space>>();
  constexpr auto prob_init = make_problem(x);
  constexpr auto hmean_x =
      HarmonicMean<HarmonicMeanMeta<Space, Singleton>, decltype(x)>(x);
  constexpr auto inv_sqrt2 = 1.0 / constexpr_sqrt_safe(2.0);

  SECTION("Max hmean(x) s.t. x >= 0, x0 + x1 == 2") {
    constexpr mat_traits<Matrixd<2, 2>> A_nonneg_mat;
    constexpr auto A_nonneg = A_nonneg_mat(-1.0, 0.0, 0.0, -1.0);
    constexpr auto b_nonneg = Vectord<2>{0.0, 0.0};
    constexpr Polyhedron<PolyhedronMeta<Space, Space>, decltype(x)> nonneg(
        A_nonneg, b_nonneg, x);
    constexpr Hyperplane<HyperplaneMeta<Space>, decltype(x)> simplex_face(
        Vectord<2>{1.0, 1.0}, 2.0, x);
    constexpr auto feasible_set =
        Intersection<IntersectionMeta<Space>, decltype(nonneg),
                     decltype(simplex_face)>(nonneg, simplex_face);
    constexpr auto prob_constr = prob_init.in_set(feasible_set);
    constexpr auto prob_max = prob_constr.maximize(hmean_x);

    STATIC_REQUIRE(is_problem_set_v<decltype(prob_max)>);
    STATIC_REQUIRE(prob_max.validate());
    STATIC_REQUIRE(
        std::tuple_size<decltype(prob_max.constraint_cones())>::value == 3);

    constexpr mat_traits<Matrixd<7, 3>> F_mat;
    constexpr auto F_exp =
        F_mat(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, -1.0, 0.0, 1.0, 1.0, 0.0, 0.0,
              0.0, -inv_sqrt2, inv_sqrt2, -inv_sqrt2, 0.0, inv_sqrt2, inv_sqrt2,
              inv_sqrt2);
    constexpr auto F = prob_max.constraint_map().linear_part();
    STATIC_REQUIRE(mat_rows_v<decltype(F)> == 7);
    STATIC_REQUIRE(mat_cols_v<decltype(F)> == 3);
    STATIC_REQUIRE(all_equal(F, F_exp));

    constexpr auto g = prob_max.constraint_map().translation();
    STATIC_REQUIRE(vec_size_v<decltype(g)> == 7);
    STATIC_REQUIRE(g == Vectord<7>{0.0, 0.0, 2.0, -2.0, 0.0, 0.0, 0.0});
    STATIC_REQUIRE(prob_max.linear_objective() == Vectord<3>{0.0, 0.0, 1.0});

    constexpr PrimalDualIPMConfig config{.duality_gap_tol = 1e-4};
    constexpr PrimalDualIPM solver(prob_max, config);
    constexpr auto res = solver.res;
    constexpr auto trace = solver.trace;

    STATIC_REQUIRE(res.tag == SolverResult::OPTIMAL);
    STATIC_REQUIRE(res.num_iter > 0 && res.num_iter <= PD_IPM_STEPS);
    STATIC_REQUIRE(trace[res.num_iter - 1].tag ==
                   config::SolverResult::OPTIMAL);
    STATIC_REQUIRE(!res.stalled);
    STATIC_REQUIRE(constexpr_abs(res.z(0) - 1.0) < rel_tol);
    STATIC_REQUIRE(constexpr_abs(res.z(1) - 1.0) < rel_tol);
    STATIC_REQUIRE(constexpr_abs(res.z(2) + 1.0) < rel_tol);
  }
}

TEST_CASE("Minimal QP 1 Var (RSOC)", "[qp_test]") {
  using Space = Matrixd<1, 1>;
  using Singleton = Matrixd<1, 1>;
  constexpr auto x = Variable<VarMeta<Space>>();
  constexpr auto prob_init = make_problem(x);
  constexpr auto x0 = x(0);

  constexpr auto inv_sqrt2 = 1.0 / constexpr_sqrt_safe(2.0);

  SECTION("Feasibility min c^T x s.t. (x - 1)^2 <= 5") {
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

  SECTION("Min x0 s.t. 1/x <= 1, x >= 0") {
    constexpr auto inv_pos = InvPos<InvPosMeta<Singleton>, decltype(x0)>(x0);
    constexpr auto leq =
        Inequality<InequalityType::LEQ, decltype(inv_pos), double>(inv_pos,
                                                                   1.0);
    constexpr auto geq =
        Inequality<InequalityType::GEQ, decltype(x0), double>(x0, 0.0);
    constexpr auto prob_constr =
        prob_init.add_constraint(leq).add_constraint(geq);
    constexpr auto prob_min = prob_constr.minimize(x0);

    STATIC_REQUIRE(is_problem_set_v<decltype(prob_min)>);
    STATIC_REQUIRE(
        std::tuple_size<decltype(prob_min.constraint_cones())>::value == 3);

    constexpr auto F = prob_min.constraint_map().linear_part();
    STATIC_REQUIRE(mat_rows_v<decltype(F)> == 5);
    STATIC_REQUIRE(mat_cols_v<decltype(F)> == 2);

    constexpr mat_traits<Matrixd<5, 2>> F2_mat;
    constexpr auto F_exp = F2_mat(0.0, 0.0, inv_sqrt2, -inv_sqrt2, inv_sqrt2,
                                  inv_sqrt2, 0.0, -1.0, 1.0, 0.0);
    STATIC_REQUIRE(all_equal(F, F_exp));

    constexpr auto g = prob_min.constraint_map().translation();
    STATIC_REQUIRE(vec_size_v<decltype(g)> == 5);
    STATIC_REQUIRE(g ==
                   Vectord<5>{constexpr_sqrt_safe(2.0), 0.0, 0.0, 1.0, 0.0});
    STATIC_REQUIRE(prob_min.validate());
    STATIC_REQUIRE(prob_min.linear_objective() == Vectord<2>{1.0, 0.0});

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
}

TEST_CASE("Minimal QP 1 Var (Huber Objective)", "[qp_test]") {
  using Space = Matrixd<1, 1>;
  constexpr auto x = Variable<VarMeta<Space>>();
  constexpr auto prob_init = make_problem(x);
  constexpr auto x0 = x(0);
  constexpr auto delta = 2.0;
  constexpr auto huber_x0 = Huber<HuberMeta<Space>, decltype(x0)>(delta, x0);
  constexpr auto inv_sqrt2 = 1.0 / constexpr_sqrt_safe(2.0);

  SECTION("Min huber(x0) s.t. x0 >= 0") {
    constexpr auto x0_geq =
        Inequality<InequalityType::GEQ, decltype(x0), double>(x0, 0.0);
    constexpr auto prob_constr = prob_init.add_constraint(x0_geq);
    constexpr auto prob_min = prob_constr.minimize(huber_x0);

    STATIC_REQUIRE(is_problem_set_v<decltype(prob_min)>);
    STATIC_REQUIRE(prob_min.validate());
    STATIC_REQUIRE(
        std::tuple_size<decltype(prob_min.constraint_cones())>::value == 3);

    constexpr mat_traits<Matrixd<10, 6>> F_mat;
    constexpr auto F_exp =
        F_mat(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, -1.0, 0.0, 0.0, 0.0,
              -1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0,
              -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
              0.0, -2.0, -1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, -inv_sqrt2, 0.0, 0.0, 0.0, 0.0, 0.0, inv_sqrt2, 0.0);
    constexpr auto F = prob_min.constraint_map().linear_part();
    STATIC_REQUIRE(mat_rows_v<decltype(F)> == 10);
    STATIC_REQUIRE(mat_cols_v<decltype(F)> == 6);
    STATIC_REQUIRE(all_equal(F, F_exp));

    constexpr auto g = prob_min.constraint_map().translation();
    STATIC_REQUIRE(vec_size_v<decltype(g)> == 10);
    STATIC_REQUIRE(g == Vectord<10>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                    0.5 * inv_sqrt2, 0.5 * inv_sqrt2});
    STATIC_REQUIRE(prob_min.linear_objective() ==
                   as_row_vector(Vectord<6>{0.0, 0.0, 0.0, 0.0, 0.0, 1.0}));

    constexpr PrimalDualIPMConfig config{.duality_gap_tol = 1e-4};
    constexpr PrimalDualIPM solver(prob_min, config);
    constexpr auto res = solver.res;
    constexpr auto trace = solver.trace;

    STATIC_REQUIRE(res.tag == SolverResult::OPTIMAL);
    STATIC_REQUIRE(res.num_iter > 0 && res.num_iter <= PD_IPM_STEPS);
    STATIC_REQUIRE(trace[res.num_iter - 1].tag ==
                   config::SolverResult::OPTIMAL);
    STATIC_REQUIRE(!res.stalled);
    STATIC_REQUIRE(constexpr_abs(res.z(0)) < 2e-2);
  }
}
