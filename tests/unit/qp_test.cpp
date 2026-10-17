/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
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

// Keep all three APIs: direct sets and a lifted Norm inequality have distinct
// canonicalization paths even though they describe the same feasible region.
TEST_CASE("Equivalent L2 ball formulations canonicalize and solve",
          "[qp_test]") {
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
    constexpr auto cx = WeightedSum<Space>()(weights, x);
    constexpr auto prob_min = prob_constr.minimize(cx);

    STATIC_REQUIRE(is_problem_set_v<decltype(prob_min)>);
    STATIC_REQUIRE(
        std::tuple_size<decltype(prob_min.constraint_cones())>::value == 1);

    constexpr auto F = prob_min.constraint_map().linear_part();
    STATIC_REQUIRE(mat_rows_v<decltype(F)> == 3);
    STATIC_REQUIRE(mat_cols_v<decltype(F)> == 2);
    STATIC_REQUIRE(all_equal(F, F_exp));

    constexpr auto g = prob_min.constraint_map().translation();
    STATIC_REQUIRE(vector_dim_v<decltype(g)> == 3);
    STATIC_REQUIRE(g_exp == g);

    constexpr PrimalDualIPM solver(prob_min);
    constexpr auto res = solver.res;
#ifdef AKREON_REPORT_OUTPUT
    constexpr auto trace = solver.trace;
#endif

    STATIC_REQUIRE(res.tag == SolverResult::OPTIMAL);
    STATIC_REQUIRE(res.num_iter > 0 && res.num_iter < PD_IPM_STEPS);
#ifdef AKREON_REPORT_OUTPUT
    STATIC_REQUIRE(trace[res.num_iter - 1].tag ==
                   config::SolverResult::OPTIMAL);
#endif
    STATIC_REQUIRE(!res.stalled);
    STATIC_REQUIRE(constexpr_abs(res.iterate.z(0) - x_exp(0)) < rel_tol);
    STATIC_REQUIRE(constexpr_abs(res.iterate.z(1) - x_exp(1)) < rel_tol);
  }

  SECTION("Min c^T x s.t. ellipsoid") {
    constexpr auto P = r * r * make_identity<double, 2>();
    constexpr auto ellipsoid = Ellipsoid(c, P, x);
    constexpr auto prob_constr = prob_init.in_set(ellipsoid);

    constexpr Vectord<3> g_exp{-c(0), -c(1), r};

    constexpr auto F_mat = mat_traits<Matrixd<3, 2>>();
    constexpr auto F_exp = F_mat(1.0, 0.0, 0.0, 1.0, 0.0, 0.0);

    constexpr Vectord<2> weights{1.0, 1.0};
    constexpr auto cx = WeightedSum<Space>()(weights, x);
    constexpr auto prob_min = prob_constr.minimize(cx);

    STATIC_REQUIRE(is_problem_set_v<decltype(prob_min)>);
    STATIC_REQUIRE(
        std::tuple_size<decltype(prob_min.constraint_cones())>::value == 1);

    constexpr auto F = prob_min.constraint_map().linear_part();
    STATIC_REQUIRE(mat_rows_v<decltype(F)> == 3);
    STATIC_REQUIRE(mat_cols_v<decltype(F)> == 2);
    STATIC_REQUIRE(all_equal(F, F_exp));

    constexpr auto g = prob_min.constraint_map().translation();
    STATIC_REQUIRE(vector_dim_v<decltype(g)> == 3);
    STATIC_REQUIRE(all_equal(g_exp, g));

    constexpr PrimalDualIPM solver(prob_min);
    constexpr auto res = solver.res;
#ifdef AKREON_REPORT_OUTPUT
    constexpr auto trace = solver.trace;
#endif

    STATIC_REQUIRE(res.tag == SolverResult::OPTIMAL);
    STATIC_REQUIRE(res.num_iter > 0 && res.num_iter < PD_IPM_STEPS);
#ifdef AKREON_REPORT_OUTPUT
    STATIC_REQUIRE(trace[res.num_iter - 1].tag ==
                   config::SolverResult::OPTIMAL);
#endif
    STATIC_REQUIRE(!res.stalled);
    STATIC_REQUIRE(constexpr_abs(res.iterate.z(0) - x_exp(0)) < rel_tol);
    STATIC_REQUIRE(constexpr_abs(res.iterate.z(1) - x_exp(1)) < rel_tol);
  }

  using Singleton = Matrixd<1, 1>;
  SECTION("Min sum(x) with Norm(x - center) <= radius") {
    constexpr Vectord<2> weights{1.0, 1.0};
    constexpr auto x_trans = x.add(Vectord<2>{-c(0), -c(1)});

    constexpr auto norm =
        Norm<NormMeta<Space, Singleton>, decltype(x), L2<Vectord<2>>>(x_trans);
    constexpr auto leq =
        Inequality<InequalityType::LEQ, decltype(norm), decltype(r)>(norm, r);
    constexpr auto prob_constr = prob_init.add_constraint(leq);

    constexpr auto cx = WeightedSum<Space>()(weights, x);
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
    STATIC_REQUIRE(vector_dim_v<decltype(g)> == 4);
    STATIC_REQUIRE(g == g_exp);

    constexpr PrimalDualIPM solver(prob_min);
    constexpr auto res = solver.res;
#ifdef AKREON_REPORT_OUTPUT
    constexpr auto trace = solver.trace;
#endif

    STATIC_REQUIRE(constexpr_abs(res.iterate.z(0) - x_exp(0)) < rel_tol);
    STATIC_REQUIRE(constexpr_abs(res.iterate.z(1) - x_exp(1)) < rel_tol);
    STATIC_REQUIRE(res.tag == SolverResult::OPTIMAL);
    STATIC_REQUIRE(res.num_iter > 0 && res.num_iter <= PD_IPM_STEPS);
#ifdef AKREON_REPORT_OUTPUT
    STATIC_REQUIRE(trace[res.num_iter - 1].tag ==
                   config::SolverResult::OPTIMAL);
#endif
    STATIC_REQUIRE(!res.stalled);
  }
}

TEST_CASE("Square objective solves with inequality and equality constraints",
          "[qp_test]") {
  using Space = Matrixd<1, 2>;
  constexpr auto x = Variable<PosVarMeta<Space>>();
  constexpr auto prob_init = make_problem(x);
  constexpr auto x0 = x(0);
  constexpr auto x1 = x(1);
  constexpr auto inv_sqrt2 = inv_sqrt2_v<double>;

  constexpr auto x0_geq =
      Inequality<InequalityType::GEQ, decltype(x0), double>(x0, 0.0);
  constexpr auto x1_eq =
      Inequality<InequalityType::EQ, decltype(x1), double>(x1, 0.0);
  constexpr auto prob_constr =
      prob_init.add_constraint(x0_geq).add_constraint(x1_eq);
  constexpr auto sq = Square(x0.add(-1.0));
  constexpr auto prob_min = prob_constr.minimize(sq);

  STATIC_REQUIRE(is_problem_set_v<decltype(prob_min)>);
  STATIC_REQUIRE(prob_min.validate());
  STATIC_REQUIRE(
      std::tuple_size<decltype(prob_min.constraint_cones())>::value == 2);

  constexpr mat_traits<Matrixd<4, 3>> F_mat;
  constexpr auto F_exp = F_mat(1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                               -inv_sqrt2, 0.0, 0.0, inv_sqrt2);
  constexpr auto F = prob_min.constraint_map().linear_part();
  STATIC_REQUIRE(mat_rows_v<decltype(F)> == 4);
  STATIC_REQUIRE(mat_cols_v<decltype(F)> == 3);
  STATIC_REQUIRE(all_equal(F, F_exp));

  constexpr auto g = prob_min.constraint_map().translation();
  STATIC_REQUIRE(vector_dim_v<decltype(g)> == 4);
  STATIC_REQUIRE(g == Vectord<4>{0.0, -1.0, 0.5 * inv_sqrt2, 0.5 * inv_sqrt2});
  constexpr mat_traits<Matrixd<1, 3>> Aeq_mat;
  constexpr auto Aeq = Aeq_mat(0.0, 1.0, 0.0);
  STATIC_REQUIRE(all_equal(prob_min.equality_map().linear_part(), Aeq));
  STATIC_REQUIRE(prob_min.equality_map().translation() == Vectord<1>{0.0});
  STATIC_REQUIRE(prob_min.linear_objective() == Vectord<3>{0.0, 0.0, 1.0});

  constexpr PrimalDualIPM solver(prob_min);
  constexpr auto res = solver.res;
#ifdef AKREON_REPORT_OUTPUT
  constexpr auto trace = solver.trace;
#endif

  STATIC_REQUIRE(res.tag == SolverResult::OPTIMAL);
  STATIC_REQUIRE(res.num_iter > 0 && res.num_iter <= PD_IPM_STEPS);
#ifdef AKREON_REPORT_OUTPUT
  STATIC_REQUIRE(trace[res.num_iter - 1].tag == config::SolverResult::OPTIMAL);
#endif
  STATIC_REQUIRE(!res.stalled);
  STATIC_REQUIRE(constexpr_abs(res.iterate.z(0) - 1.0) < rel_tol);
  STATIC_REQUIRE(constexpr_abs(res.iterate.z(1)) < rel_tol);
  STATIC_REQUIRE(constexpr_abs(res.iterate.z(2)) < rel_tol);
}

TEST_CASE("Square objective solves with a free scalar variable", "[qp_test]") {
  using Space = Matrixd<1, 1>;
  constexpr auto x = Variable<VarMeta<Space>>();
  constexpr auto prob_init = make_problem(x);
  constexpr auto x0 = x(0);
  constexpr auto inv_sqrt2 = inv_sqrt2_v<double>;

  constexpr auto sq = Square(x0.add(-1.0));
  constexpr auto prob_min = prob_init.minimize(sq);

  STATIC_REQUIRE(is_problem_set_v<decltype(prob_min)>);
  STATIC_REQUIRE(prob_min.validate());
  STATIC_REQUIRE(
      std::tuple_size<decltype(prob_min.constraint_cones())>::value == 1);

  constexpr mat_traits<Matrixd<3, 2>> F_mat;
  constexpr auto F_exp = F_mat(1.0, 0.0, 0.0, -inv_sqrt2, 0.0, inv_sqrt2);
  constexpr auto F = prob_min.constraint_map().linear_part();
  STATIC_REQUIRE(mat_rows_v<decltype(F)> == 3);
  STATIC_REQUIRE(mat_cols_v<decltype(F)> == 2);
  STATIC_REQUIRE(all_equal(F, F_exp));

  constexpr auto g = prob_min.constraint_map().translation();
  STATIC_REQUIRE(vector_dim_v<decltype(g)> == 3);
  STATIC_REQUIRE(g == Vectord<3>{-1.0, 0.5 * inv_sqrt2, 0.5 * inv_sqrt2});
  STATIC_REQUIRE(prob_min.linear_objective() == Vectord<2>{0.0, 1.0});

  constexpr PrimalDualIPM solver(prob_min);
  constexpr auto res = solver.res;
#ifdef AKREON_REPORT_OUTPUT
  constexpr auto trace = solver.trace;
#endif

  STATIC_REQUIRE(res.tag == SolverResult::OPTIMAL);
  STATIC_REQUIRE(res.num_iter > 0 && res.num_iter <= PD_IPM_STEPS);
#ifdef AKREON_REPORT_OUTPUT
  STATIC_REQUIRE(trace[res.num_iter - 1].tag == config::SolverResult::OPTIMAL);
#endif
  STATIC_REQUIRE(!res.stalled);
  STATIC_REQUIRE(constexpr_abs(res.iterate.z(0) - 1.0) < rel_tol);
  STATIC_REQUIRE(constexpr_abs(res.iterate.z(1)) < rel_tol);
}

TEST_CASE("Sqrt maximization solves on the unit interval", "[qp_test]") {
  using Space = Matrixd<1, 1>;
  constexpr auto x = Variable<PosVarMeta<Space>>();
  constexpr auto prob_init = make_problem(x);
  constexpr auto x0 = x(0);
  constexpr auto inv_sqrt2 = inv_sqrt2_v<double>;

  constexpr auto x0_leq =
      Inequality<InequalityType::LEQ, decltype(x0), double>(x0, 1.0);
  constexpr auto x0_geq =
      Inequality<InequalityType::GEQ, decltype(x0), double>(x0, 0.0);
  constexpr auto prob_constr =
      prob_init.add_constraint(x0_leq).add_constraint(x0_geq);
  constexpr auto sqrt_x0 = Sqrt(x0);
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
  STATIC_REQUIRE(vector_dim_v<decltype(g)> == 5);
  STATIC_REQUIRE(g ==
                 Vectord<5>{1.0, 0.0, 0.0, 0.5 * inv_sqrt2, 0.5 * inv_sqrt2});
  STATIC_REQUIRE(prob_max.linear_objective() == Vectord<2>{0.0, 1.0});

  constexpr PrimalDualIPM solver(prob_max);
  constexpr auto res = solver.res;
#ifdef AKREON_REPORT_OUTPUT
  constexpr auto trace = solver.trace;
#endif

  STATIC_REQUIRE(res.tag == SolverResult::OPTIMAL);
  STATIC_REQUIRE(res.num_iter > 0 && res.num_iter <= PD_IPM_STEPS);
#ifdef AKREON_REPORT_OUTPUT
  STATIC_REQUIRE(trace[res.num_iter - 1].tag == config::SolverResult::OPTIMAL);
#endif
  STATIC_REQUIRE(!res.stalled);
  STATIC_REQUIRE(constexpr_abs(res.iterate.z(0) - 1.0) < rel_tol);
  STATIC_REQUIRE(constexpr_abs(res.iterate.z(1) + 1.0) < rel_tol);
}

TEST_CASE("Harmonic mean maximization solves with a fixed component sum",
          "[qp_test]") {
  using Space = Matrixd<1, 2>;
  constexpr auto x = Variable<VarMeta<Space>>();
  constexpr auto prob_init = make_problem(x);
  constexpr auto hmean_x = HarmonicMean(x);
  constexpr auto inv_sqrt2 = inv_sqrt2_v<double>;

  constexpr mat_traits<Matrixd<2, 2>> A_nonneg_mat;
  constexpr auto A_nonneg = A_nonneg_mat(-1.0, 0.0, 0.0, -1.0);
  constexpr auto b_nonneg = Vectord<2>{0.0, 0.0};
  constexpr auto nonneg = Polyhedron(A_nonneg, b_nonneg, x);
  constexpr auto simplex_face = Hyperplane(Vectord<2>{1.0, 1.0}, 2.0, x);
  constexpr auto feasible_set = Intersection(nonneg, simplex_face);
  constexpr auto prob_constr = prob_init.in_set(feasible_set);
  constexpr auto prob_max = prob_constr.maximize(hmean_x);

  STATIC_REQUIRE(is_problem_set_v<decltype(prob_max)>);
  STATIC_REQUIRE(prob_max.validate());
  STATIC_REQUIRE(
      std::tuple_size<decltype(prob_max.constraint_cones())>::value == 2);

  constexpr mat_traits<Matrixd<5, 3>> F_mat;
  constexpr auto F_exp =
      F_mat(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -inv_sqrt2, inv_sqrt2,
            -inv_sqrt2, 0.0, inv_sqrt2, inv_sqrt2, inv_sqrt2);
  constexpr auto F = prob_max.constraint_map().linear_part();
  STATIC_REQUIRE(mat_rows_v<decltype(F)> == 5);
  STATIC_REQUIRE(mat_cols_v<decltype(F)> == 3);
  STATIC_REQUIRE(all_equal(F, F_exp));

  constexpr auto g = prob_max.constraint_map().translation();
  STATIC_REQUIRE(vector_dim_v<decltype(g)> == 5);
  STATIC_REQUIRE(g == Vectord<5>{0.0, 0.0, 0.0, 0.0, 0.0});
  constexpr mat_traits<Matrixd<1, 3>> Aeq_mat;
  constexpr auto Aeq = Aeq_mat(1.0, 1.0, 0.0);
  STATIC_REQUIRE(all_equal(prob_max.equality_map().linear_part(), Aeq));
  STATIC_REQUIRE(prob_max.equality_map().translation() == Vectord<1>{2.0});
  STATIC_REQUIRE(prob_max.linear_objective() == Vectord<3>{0.0, 0.0, 1.0});

  constexpr auto config = [] {
    PrimalDualIPMConfig<double> cfg{};
    cfg.duality_gap_tol = 1e-4;
    return cfg;
  }();
  constexpr PrimalDualIPM solver(prob_max, config);
  constexpr auto res = solver.res;
#ifdef AKREON_REPORT_OUTPUT
  constexpr auto trace = solver.trace;
#endif

  STATIC_REQUIRE(res.tag == SolverResult::OPTIMAL);
  STATIC_REQUIRE(res.num_iter > 0 && res.num_iter <= PD_IPM_STEPS);
#ifdef AKREON_REPORT_OUTPUT
  STATIC_REQUIRE(trace[res.num_iter - 1].tag == config::SolverResult::OPTIMAL);
#endif
  STATIC_REQUIRE(!res.stalled);
  STATIC_REQUIRE(constexpr_abs(res.iterate.z(0) - 1.0) < rel_tol);
  STATIC_REQUIRE(constexpr_abs(res.iterate.z(1) - 1.0) < rel_tol);
  STATIC_REQUIRE(constexpr_abs(res.iterate.z(2) + 1.0) < rel_tol);
}

TEST_CASE("Rotated SOC constraints solve square and reciprocal problems",
          "[qp_test]") {
  using Space = Matrixd<1, 1>;
  constexpr auto x = Variable<VarMeta<Space>>();
  constexpr auto prob_init = make_problem(x);
  constexpr auto x0 = x(0);

  constexpr auto inv_sqrt2 = inv_sqrt2_v<double>;

  SECTION("Feasibility min c^T x s.t. (x - 1)^2 <= 5") {
    constexpr auto sq = Square(x0.add(-1.0));
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
    STATIC_REQUIRE(vector_dim_v<decltype(g)> == 4);
    STATIC_REQUIRE(g ==
                   Vectord<4>{-1.0, 0.5 * inv_sqrt2, 0.5 * inv_sqrt2, 5.0});
    STATIC_REQUIRE(prob_min.validate());

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
    STATIC_REQUIRE(constexpr_abs(res.iterate.z(0) - 1.0) < rel_tol);
  }

  SECTION("Min x0 s.t. 1/x <= 1, x >= 0") {
    constexpr auto inv_pos = InvPos(x0);
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
    STATIC_REQUIRE(vector_dim_v<decltype(g)> == 5);
    STATIC_REQUIRE(g == Vectord<5>{sqrt2_v<double>, 0.0, 0.0, 1.0, 0.0});
    STATIC_REQUIRE(prob_min.validate());
    STATIC_REQUIRE(prob_min.linear_objective() == Vectord<2>{1.0, 0.0});

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
    STATIC_REQUIRE(constexpr_abs(res.iterate.z(0) - 1.0) < rel_tol);
  }
}

TEST_CASE("Huber minimization solves with a nonnegative lower bound",
          "[qp_test]") {
  using Space = Matrixd<1, 1>;
  constexpr auto x = Variable<VarMeta<Space>>();
  constexpr auto prob_init = make_problem(x);
  constexpr auto x0 = x(0);
  constexpr auto delta = 2.0;
  constexpr auto huber_x0 = Huber(delta, x0);
  constexpr auto inv_sqrt2 = inv_sqrt2_v<double>;

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
      F_mat(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, -1.0, 0.0, 0.0, 0.0, -1.0,
            1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0,
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            -2.0, -1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            -inv_sqrt2, 0.0, 0.0, 0.0, 0.0, 0.0, inv_sqrt2, 0.0);
  constexpr auto F = prob_min.constraint_map().linear_part();
  STATIC_REQUIRE(mat_rows_v<decltype(F)> == 10);
  STATIC_REQUIRE(mat_cols_v<decltype(F)> == 6);
  STATIC_REQUIRE(all_equal(F, F_exp));

  constexpr auto g = prob_min.constraint_map().translation();
  STATIC_REQUIRE(vector_dim_v<decltype(g)> == 10);
  STATIC_REQUIRE(g == Vectord<10>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                  0.5 * inv_sqrt2, 0.5 * inv_sqrt2});
  STATIC_REQUIRE(prob_min.linear_objective() ==
                 Vectord<6>{0.0, 0.0, 0.0, 0.0, 0.0, 1.0});

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
  STATIC_REQUIRE(trace[res.num_iter - 1].tag == config::SolverResult::OPTIMAL);
#endif
  STATIC_REQUIRE(!res.stalled);
  STATIC_REQUIRE(constexpr_abs(res.iterate.z(0)) < 2e-2);
}

TEST_CASE("QP warm start reuses an iterate on an L2 ball", "[qp_test]") {
  using Space = Matrixd<1, 2>;

  constexpr auto x = Variable<NonnegVarMeta<Space>>();
  constexpr auto prob_init = make_problem(x);
  constexpr auto center = Vectord<2>{3.0, 1.0};
  constexpr auto r = 1.0;
  constexpr auto x_exp = Vectord<2>{2.29289322, 0.29289322};

  constexpr auto ball =
      NormBall<NormBallMeta<Space>, decltype(x), NormBallType::L2>(center, r,
                                                                   x);
  constexpr auto prob_constr = prob_init.in_set(ball);

  constexpr auto weights = Vectord<2>{1.0, 1.0};
  constexpr auto cx = WeightedSum<Space>()(weights, x);
  constexpr auto prob_min = prob_constr.minimize(cx);

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
  STATIC_REQUIRE(warm_res.num_iter > 0 && warm_res.num_iter < PD_IPM_STEPS);
  STATIC_REQUIRE(warm_res.num_iter <= cold_res.num_iter);
  STATIC_REQUIRE(constexpr_abs(warm_res.iterate.z(0) - x_exp(0)) < rel_tol);
  STATIC_REQUIRE(constexpr_abs(warm_res.iterate.z(1) - x_exp(1)) < rel_tol);
  STATIC_REQUIRE(constexpr_abs(warm_res.iterate.z(0) - cold_res.iterate.z(0)) <
                 rel_tol);
  STATIC_REQUIRE(constexpr_abs(warm_res.iterate.z(1) - cold_res.iterate.z(1)) <
                 rel_tol);
}

TEST_CASE("QuadForm keeps the norm auxiliary coupled to its square",
          "[qp_test]") {
  constexpr auto x = Variable<VarMeta<Matrixd<1, 2>>>();
  constexpr auto P = mat_traits<Matrixd<2, 2>>{}(2.0, 1.0, 1.0, 2.5);
  constexpr auto q = Vectord<2>{-4.0, -6.0};
  constexpr auto prob = make_problem(x).minimize(make_quad_form(P, q, 0.75, x));
  STATIC_REQUIRE(mat_cols_v<decltype(prob.constraint_map().linear_part())> ==
                 4);
  STATIC_REQUIRE(prob.linear_objective() == Vectord<4>{-4.0, -6.0, 0.0, 1.0});
  STATIC_REQUIRE(prob.builder.coord_map.translation() == Vectord<1>{0.75});
  // At x=[1,2], 0.5*x^T*P*x=8. Both cone bounds must share u=sqrt(8).
  const auto z = Vectord<4>{1.0, 2.0, std::sqrt(8.0), 8.0};
  const auto mapped = prob.constraint_map()(z);
  REQUIRE(mapped(2) >= std::hypot(mapped(0), mapped(1)) - 1e-8);
  REQUIRE(mapped(5) >= std::hypot(mapped(3), mapped(4)) - 1e-8);
  const auto disconnected =
      prob.constraint_map()(Vectord<4>{1.0, 2.0, std::sqrt(8.0), 0.0});
  REQUIRE(disconnected(5) < std::hypot(disconnected(3), disconnected(4)));
}

TEST_CASE("Corrector cross term uses dual coordinates", "[qp_test]") {
  using Vec = Vectord<2>;
  const auto F = make_identity<double, 2>();
  const Matrix<double, 0, 2> A{};
  const auto cones = std::tuple(NonnegOrthantCone<Vec>{});
  const NoEqualityCoupling<Vec, Vector<double, 0>> eq;
  const PredictorCorrector<1, decltype(F), decltype(A), Vec, Vec> pc;

  for (double scale : {1.0, 7.0}) {
    const auto x = scale * Vec{2.0, 4.0};
    const Vec y{3.0, 5.0};
    const auto px = scale * Vec{0.5, -0.25};
    const Vec py{-0.75, 0.5};
    const auto factor = LDL(make_diag(Vec{1.5 / scale, 1.25 / scale}));
#ifdef AKREON_REPORT_OUTPUT
    PdIpmIterTrace<double, 1> trace{};
#endif
    const auto [rhs, step, event] =
        pc.corrector(Vec{}, px, py, cones, factor, eq, F, Vec{}, Vec{},
                     Vector<double, 0>{}, x, y
#ifdef AKREON_REPORT_OUTPUT
                     ,
                     trace
#endif
        );
    REQUIRE(event == NumericalEvent::NONE);
    // mu=13; rhs=-(y-mu/x+px*py/x), independent of primal scaling.
    REQUIRE(all_equal(rhs, Vec{3.6875, -1.71875}, 1e-10));
    REQUIRE(all_equal(step.py, Vec{}, 1e-10));
  }
}

TEST_CASE("Corrector centers a product cone with a shared gap", "[qp_test]") {
  using Vec = Vectord<5>;
  const auto F = make_identity<double, 5>();
  const Matrix<double, 0, 5> A{};
  const auto cones = std::tuple(NonnegOrthantCone<Vectord<2>>{},
                                SecondOrderCone<Vectord<3>>{});
  const NoEqualityCoupling<Vec, Vector<double, 0>> eq;
  const PredictorCorrector<2, decltype(F), decltype(A), Vec, Vec> pc;
  const auto factor = LDL(make_diag(Vec{1.5, 1.25, 7.0 / 3, 7.0 / 3, 7.0 / 3}));
#ifdef AKREON_REPORT_OUTPUT
  PdIpmIterTrace<double, 2> trace{};
#endif
  const auto [rhs, step, event] =
      pc.corrector(Vec{}, Vec{}, Vec{}, cones, factor, eq, F, Vec{}, Vec{},
                   Vector<double, 0>{}, Vec{2.0, 4.0, 0.0, 0.0, 3.0},
                   Vec{3.0, 5.0, 0.0, 0.0, 7.0}
#ifdef AKREON_REPORT_OUTPUT
                   ,
                   trace
#endif
      );
  REQUIRE(event == NumericalEvent::NONE);
  // <x,y>/nu=47/4. SOC's Jordan trace contributes twice its Euclidean dot.
  REQUIRE(all_equal(rhs, Vec{2.875, -2.0625, 0.0, 0.0, 5.0 / 6}, 1e-9));
  REQUIRE(all_equal(step.py, Vec{}, 1e-9));
}
