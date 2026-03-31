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

TEST_CASE("Minimal SDP", "[sdp_test]") {
  using Space = Matrixd<1, 2>;
  constexpr auto x = Variable<VarMeta<Space>>();
  constexpr auto prob_init = make_problem(x);

  SECTION("Min c^T x s.t. LMI") {
    constexpr auto A0 =
        Matrixd<2, 2>{Vectord<2>{-1.0, 0.0}, Vectord<2>{0.0, -2.0}};
    constexpr auto A1 =
        Matrixd<2, 2>{Vectord<2>{-2.0, 0.0}, Vectord<2>{0.0, -1.0}};
    constexpr auto B =
        Matrixd<2, 2>{Vectord<2>{1.0, 0.0}, Vectord<2>{0.0, 1.0}};
    constexpr Vector<Matrixd<2, 2>, 2> A{A0, A1};
    constexpr auto feasible_set = LMI(A, B, x);
    constexpr auto prob_constr = prob_init.in_set(feasible_set);

    constexpr auto geq_x0 =
        Inequality<InequalityType::GEQ, decltype(x(0)), double>(x(0), 0.0);
    constexpr auto geq_x1 =
        Inequality<InequalityType::GEQ, decltype(x(1)), double>(x(1), 0.0);
    constexpr auto x_constr =
        prob_constr.add_constraint(geq_x0).add_constraint(geq_x1);

    constexpr Vectord<2> c{1.0, 1.0};
    constexpr auto cx = WeightedSum<Space>()(c, x);
    constexpr auto prob_min = x_constr.minimize(cx);

    STATIC_REQUIRE(is_problem_set_v<decltype(prob_min)>);
    STATIC_REQUIRE(prob_min.validate());
    STATIC_REQUIRE(
        std::tuple_size<decltype(prob_min.constraint_cones())>::value == 3);

    constexpr auto F_mat = mat_traits<Matrixd<5, 2>>();
    constexpr auto F_exp =
        F_mat(1.0, 2.0, 0.0, 0.0, 2.0, 1.0, 1.0, 0.0, 0.0, 1.0);
    constexpr auto F = prob_min.constraint_map().linear_part();
    STATIC_REQUIRE(mat_rows_v<decltype(F)> == 5);
    STATIC_REQUIRE(mat_cols_v<decltype(F)> == 2);
    STATIC_REQUIRE(all_equal(F, F_exp));

    constexpr auto g = prob_min.constraint_map().translation();
    STATIC_REQUIRE(vec_size_v<decltype(g)> == 5);
    STATIC_REQUIRE(g == Vectord<5>{1.0, 0.0, 1.0, 0.0, 0.0});
    STATIC_REQUIRE(prob_min.linear_objective() == Matrixd<1, 2>{c});

    /// (dakre) moving this to runtime path as compile time is slow
    PrimalDualIPMConfig<double> cfg{};
    cfg.steps = 64;
    cfg.primal_feas_tol = 1e-4;
    cfg.dual_feas_tol = 1e-4;
    cfg.duality_gap_tol = 1e-4;
    const PrimalDualIPM solver(prob_min, cfg);
    const auto& res = solver.res;
#ifdef AKREON_DEBUG
    const auto& trace = solver.trace;
#endif

#ifdef AKREON_DEBUG
    if (res.tag != SolverResult::OPTIMAL || res.stalled) log(res, trace);
#endif
    REQUIRE(res.tag == SolverResult::OPTIMAL);
    REQUIRE(res.num_iter > 0);
    REQUIRE(res.num_iter <= PD_IPM_STEPS);
#ifdef AKREON_DEBUG
    REQUIRE(trace[res.num_iter - 1].tag == config::SolverResult::OPTIMAL);
#endif
    REQUIRE(!res.stalled);
    REQUIRE(constexpr_abs(res.iterate.z(0)) < rel_tol);
    REQUIRE(constexpr_abs(res.iterate.z(1)) < rel_tol);
  }

  SECTION("Min c^T x s.t. LMI and x0 == x1") {
    constexpr auto A0 =
        Matrixd<2, 2>{Vectord<2>{-1.0, 0.0}, Vectord<2>{0.0, -2.0}};
    constexpr auto A1 =
        Matrixd<2, 2>{Vectord<2>{-2.0, 0.0}, Vectord<2>{0.0, -1.0}};
    constexpr auto B =
        Matrixd<2, 2>{Vectord<2>{1.0, 0.0}, Vectord<2>{0.0, 1.0}};
    constexpr Vector<Matrixd<2, 2>, 2> A{A0, A1};
    constexpr auto lmi = LMI(A, B, x);
    constexpr auto eq_x0_x1 = Hyperplane(Vectord<2>{1.0, -1.0}, 0.0, x);

    constexpr auto prob_constr = prob_init.in_set(lmi).in_set(eq_x0_x1);

    constexpr auto geq_x0 =
        Inequality<InequalityType::GEQ, decltype(x(0)), double>(x(0), 0.0);
    constexpr auto geq_x1 =
        Inequality<InequalityType::GEQ, decltype(x(1)), double>(x(1), 0.0);
    constexpr auto x_constr =
        prob_constr.add_constraint(geq_x0).add_constraint(geq_x1);

    constexpr Vectord<2> c{1.0, 1.0};
    constexpr auto cx = WeightedSum<Space>()(c, x);
    constexpr auto prob_min = x_constr.minimize(cx);

    STATIC_REQUIRE(is_problem_set_v<decltype(prob_min)>);
    STATIC_REQUIRE(prob_min.validate());
    STATIC_REQUIRE(
        std::tuple_size<decltype(prob_min.constraint_cones())>::value == 3);

    constexpr auto Aeq_mat = mat_traits<Matrixd<1, 2>>();
    constexpr auto Aeq = Aeq_mat(1.0, -1.0);
    STATIC_REQUIRE(all_equal(prob_min.equality_map().linear_part(), Aeq));
    STATIC_REQUIRE(prob_min.equality_map().translation() == Vectord<1>{0.0});

    PrimalDualIPMConfig<double> cfg{};
    cfg.steps = 64;
    cfg.primal_feas_tol = 1e-4;
    cfg.dual_feas_tol = 1e-4;
    cfg.duality_gap_tol = 1e-4;
    const PrimalDualIPM solver(prob_min, cfg);
    const auto& res = solver.res;
#ifdef AKREON_DEBUG
    const auto& trace = solver.trace;
    if (res.tag != SolverResult::OPTIMAL || res.stalled) log(res, trace);
#endif

    REQUIRE(res.tag == SolverResult::OPTIMAL);
    REQUIRE(res.num_iter > 0);
    REQUIRE(res.num_iter <= PD_IPM_STEPS);
#ifdef AKREON_DEBUG
    REQUIRE(trace[res.num_iter - 1].tag == config::SolverResult::OPTIMAL);
#endif
    REQUIRE(!res.stalled);
    REQUIRE(constexpr_abs(res.iterate.z(0)) < rel_tol);
    REQUIRE(constexpr_abs(res.iterate.z(1)) < rel_tol);
  }

  SECTION("Min c^T x s.t. 3x3 LMI") {
    constexpr auto A0 = Matrixd<3, 3>{
        Vectord<3>{-1.0, 0.0, 0.0},
        Vectord<3>{0.0, -2.0, 0.0},
        Vectord<3>{0.0, 0.0, -3.0},
    };
    constexpr auto A1 = Matrixd<3, 3>{
        Vectord<3>{-3.0, 0.0, 0.0},
        Vectord<3>{0.0, -2.0, 0.0},
        Vectord<3>{0.0, 0.0, -1.0},
    };
    constexpr auto B = Matrixd<3, 3>{
        Vectord<3>{1.0, 0.0, 0.0},
        Vectord<3>{0.0, 1.0, 0.0},
        Vectord<3>{0.0, 0.0, 1.0},
    };

    constexpr Vector<Matrixd<3, 3>, 2> A{A0, A1};
    constexpr auto feasible_set = LMI(A, B, x);
    constexpr auto prob_constr = prob_init.in_set(feasible_set);

    constexpr auto geq_x0 =
        Inequality<InequalityType::GEQ, decltype(x(0)), double>(x(0), 0.0);
    constexpr auto geq_x1 =
        Inequality<InequalityType::GEQ, decltype(x(1)), double>(x(1), 0.0);
    constexpr auto x_constr =
        prob_constr.add_constraint(geq_x0).add_constraint(geq_x1);

    constexpr Vectord<2> c{1.0, 1.0};
    constexpr auto cx = WeightedSum<Space>()(c, x);
    constexpr auto prob_min = x_constr.minimize(cx);

    STATIC_REQUIRE(is_problem_set_v<decltype(prob_min)>);
    STATIC_REQUIRE(prob_min.validate());
    STATIC_REQUIRE(
        std::tuple_size<decltype(prob_min.constraint_cones())>::value == 3);

    // PSD(3) contributes 6 lower-triangular coordinates, then 2 R_+
    // constraints.
    constexpr auto F_exp = Matrixd<8, 2>{
        Vectord<2>{1.0, 3.0},  // (0,0)
        Vectord<2>{0.0, 0.0},  // (1,0)
        Vectord<2>{0.0, 0.0},  // (2,0)
        Vectord<2>{2.0, 2.0},  // (1,1)
        Vectord<2>{0.0, 0.0},  // (2,1)
        Vectord<2>{3.0, 1.0},  // (2,2)
        Vectord<2>{1.0, 0.0},  // x0 >= 0
        Vectord<2>{0.0, 1.0},  // x1 >= 0
    };

    const auto F = prob_min.constraint_map().linear_part();
    STATIC_REQUIRE(mat_rows_v<decltype(F)> == 8);
    STATIC_REQUIRE(mat_cols_v<decltype(F)> == 2);
    REQUIRE(all_equal(F, F_exp));

    const auto g = prob_min.constraint_map().translation();
    STATIC_REQUIRE(vec_size_v<decltype(g)> == 8);
    REQUIRE(g == Vectord<8>{1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0});

    STATIC_REQUIRE(prob_min.linear_objective() == Matrixd<1, 2>{c});

    SUCCEED("3x3 LMI with 2-dim decision variable canonicalizes correctly");
  }
}
