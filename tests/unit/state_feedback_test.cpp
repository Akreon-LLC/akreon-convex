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

static constexpr auto rel_tol{1e-3};

TEST_CASE("State Feedback 2x2", "[state_feedback_test]") {
  using Space = Matrixd<2, 2>;
  using QSpace = SymmetricMatrix<Space>;
  constexpr auto Q = Variable<PsdVarMeta<Space>>();
  constexpr auto I = make_identity<double, 2>();
  constexpr auto Z = Matrixd<2, 2>{};
  constexpr auto MatBuilder = mat_traits<Space>();

  SECTION("Hand-checkable canonical LMI") {
    constexpr auto A = Z;
    constexpr auto B = I;
    constexpr auto C = I;
    constexpr auto ss =
        StateSpace<TimeDomainType::CONTINUOUS, Space, Space, Space>(A, B, C);

    constexpr auto sf = StateFeedback<decltype(ss)>();
    constexpr auto lmi = sf(ss, Q);

    STATIC_REQUIRE(lmi.is_well_formed());

    constexpr auto builder = conic::create_empty_cblk_builder<QSpace>();
    constexpr auto cone_block =
        lmi.template canonicalize<AtomType::SET>(builder).cb;

    STATIC_REQUIRE(all_equal(cone_block.map.linear_part(), Matrixd<3, 3>{}));
    STATIC_REQUIRE(cone_block.map.translation() == Vectord<3>{1.0, 0.0, 1.0});

    constexpr auto cones = cone_block.cones;
    constexpr auto sz = std::tuple_size<decltype(cones)>::value;
    STATIC_REQUIRE(sz == 1);
    STATIC_REQUIRE(cone_block.validate());
  }

  SECTION("Solver finds feasible/stabilizing Q on unstable full-actuation plant") {
    constexpr auto A = MatBuilder(0.2, 0.0, 0.0, 0.1);
    constexpr auto B = I;
    constexpr auto C = I;
    constexpr auto ss =
        StateSpace<TimeDomainType::CONTINUOUS, Space, Space, Space>(A, B, C);

    constexpr auto Q_geq =
        Inequality<InequalityType::GEQ, decltype(Q), Space>(Q, 1e-6 * I);
    constexpr auto sf = StateFeedback<decltype(ss)>();
    constexpr auto lmi = sf(ss, Q);
    constexpr auto prob_constr =
        make_problem(Q).add_constraint(Q_geq).in_set(lmi);
    constexpr auto prob = prob_constr.feasibility();

    STATIC_REQUIRE(is_problem_set_v<decltype(prob)>);
    STATIC_REQUIRE(prob.validate());
    STATIC_REQUIRE(
        std::tuple_size<decltype(prob.constraint_cones())>::value == 2);

    const PrimalDualIPM solver(prob);
    const auto& res = solver.res;
#ifdef AKREON_DEBUG
    const auto& trace = solver.trace;
    if (res.tag != SolverResult::OPTIMAL || res.stalled) log(res, trace);
#endif

    REQUIRE(res.tag == SolverResult::OPTIMAL);
    REQUIRE(!res.stalled);

    const auto Q_res = make_smat<2>(res.iterate.z);
    REQUIRE(all_equal(Q_res, transpose(Q_res), rel_tol));
    REQUIRE(Q_res(0)(0) > 1e-6);
    REQUIRE(Q_res(1)(1) > 1e-6);

    const auto [K_res, K_event] = sf.K(ss, Q_res);
    REQUIRE(K_event == linalg::NumericalEvent::NONE);
    REQUIRE(K_res(0)(0) < 0.0);
    REQUIRE(K_res(1)(1) < 0.0);
    REQUIRE(constexpr_abs(K_res(0)(1)) < 1e-3);
    REQUIRE(constexpr_abs(K_res(1)(0)) < 1e-3);

    const auto closed_loop = A + B * K_res;
    REQUIRE(closed_loop(0)(0) < 0.0);
    REQUIRE(closed_loop(1)(1) < 0.0);

    const auto cert = (-1.0) * (A * Q_res + Q_res * transpose(A)) + B * transpose(B);
    const auto cert_sym = linalg::SymmetricMatrix(cert);
    REQUIRE(cert_sym.is_valid());
    REQUIRE(cert_sym.is_psd());
  }

  SECTION("Explicit feasible point satisfies canonical constraints") {
    constexpr auto A = MatBuilder(0.2, 0.0, 0.0, 0.1);
    constexpr auto B = I;
    constexpr auto C = I;
    constexpr auto ss =
        StateSpace<TimeDomainType::CONTINUOUS, Space, Space, Space>(A, B, C);

    constexpr auto Q_geq =
        Inequality<InequalityType::GEQ, decltype(Q), Space>(Q, 1e-6 * I);
    constexpr auto sf = StateFeedback<decltype(ss)>();
    constexpr auto lmi = sf(ss, Q);
    constexpr auto prob =
        make_problem(Q).add_constraint(Q_geq).in_set(lmi).feasibility();

    STATIC_REQUIRE(is_problem_set_v<decltype(prob)>);
    STATIC_REQUIRE(prob.validate());

    const PrimalDualIPM solver(prob);
    const auto& res = solver.res;
#ifdef AKREON_DEBUG
    const auto& trace = solver.trace;
    if (res.tag != SolverResult::OPTIMAL || res.stalled) log(res, trace);
#endif

    REQUIRE(res.tag == SolverResult::OPTIMAL);
    REQUIRE(!res.stalled);

    const auto Q_feas = I;
    const auto z_feas = get_svec(Q_feas);
    const auto x_feas =
        prob.constraint_map().linear_part() * z_feas +
        prob.constraint_map().translation();

    const auto Q_slack = make_smat<2>(Vectord<3>{x_feas(0), x_feas(1), x_feas(2)});
    REQUIRE(Q_slack(0)(0) > 0.0);
    REQUIRE(Q_slack(1)(1) > 0.0);

    const auto cert_slack = make_smat<2>(
        Vectord<3>{x_feas(3), x_feas(4), x_feas(5)});
    const auto cert_sym = SymmetricMatrix(cert_slack, 1e-6);
    REQUIRE(cert_sym.is_valid());
    REQUIRE(cert_sym.is_psd());
  }
}

TEST_CASE("State Feedback 3x3", "[state_feedback_test]") {
  using Space = Matrixd<3, 3>;
  constexpr auto Q = Variable<PsdVarMeta<Space>>();
  constexpr auto I = make_identity<double, 3>();
  constexpr auto MatBuilder = mat_traits<Space>();

  SECTION("Solver finds feasible Q on coupled stable plant") {
    constexpr auto A =
        MatBuilder(-1.0, 0.2, 0.0, 0.0, -0.5, 0.3, 0.0, 0.0, -2.0);
    constexpr auto B = I;
    constexpr auto C = I;
    constexpr auto ss =
        StateSpace<TimeDomainType::CONTINUOUS, Space, Space, Space>(A, B, C);

    constexpr auto Q_geq =
        Inequality<InequalityType::GEQ, decltype(Q), Space>(Q, 1e-6 * I);
    constexpr auto sf = StateFeedback<decltype(ss)>();
    constexpr auto lmi = sf(ss, Q);
    constexpr auto prob_constr =
        make_problem(Q).add_constraint(Q_geq).in_set(lmi);
    constexpr auto prob = prob_constr.feasibility();

    STATIC_REQUIRE(is_problem_set_v<decltype(prob)>);
    STATIC_REQUIRE(prob.validate());
    STATIC_REQUIRE(
        std::tuple_size<decltype(prob.constraint_cones())>::value == 2);

    const PrimalDualIPM solver(prob);
    const auto& res = solver.res;
#ifdef AKREON_DEBUG
    const auto& trace = solver.trace;
    if (res.tag != SolverResult::OPTIMAL || res.stalled) log(res, trace);
#endif

    REQUIRE(res.tag == SolverResult::OPTIMAL);
    REQUIRE(!res.stalled);

    const auto Q_res = make_smat<3>(res.iterate.z);
    REQUIRE(all_equal(Q_res, transpose(Q_res), rel_tol));
    REQUIRE(Q_res(0)(0) > 1e-6);
    REQUIRE(Q_res(1)(1) > 1e-6);
    REQUIRE(Q_res(2)(2) > 1e-6);

    const auto cert = (-1.0) * (A * Q_res + Q_res * transpose(A)) + B * transpose(B);
    const auto cert_sym = linalg::SymmetricMatrix(cert);
    REQUIRE(cert_sym.is_valid());
    REQUIRE(cert_sym.is_psd());
  }

  SECTION("Explicit feasible 3x3 point satisfies canonical constraints") {
    constexpr auto A =
        MatBuilder(-1.0, 0.2, 0.0, 0.0, -0.5, 0.3, 0.0, 0.0, -2.0);
    constexpr auto B = I;
    constexpr auto C = I;
    constexpr auto ss =
        StateSpace<TimeDomainType::CONTINUOUS, Space, Space, Space>(A, B, C);

    constexpr auto Q_geq =
        Inequality<InequalityType::GEQ, decltype(Q), Space>(Q, 1e-6 * I);
    constexpr auto sf = StateFeedback<decltype(ss)>();
    constexpr auto lmi = sf(ss, Q);
    constexpr auto prob =
        make_problem(Q).add_constraint(Q_geq).in_set(lmi).feasibility();

    STATIC_REQUIRE(is_problem_set_v<decltype(prob)>);
    STATIC_REQUIRE(prob.validate());

    const PrimalDualIPM solver(prob);
    const auto& res = solver.res;
#ifdef AKREON_DEBUG
    const auto& trace = solver.trace;
    if (res.tag != SolverResult::OPTIMAL || res.stalled) log(res, trace);
#endif

    REQUIRE(res.tag == SolverResult::OPTIMAL);
    REQUIRE(!res.stalled);

    const auto Q_feas = I;
    const auto z_feas = get_svec(Q_feas);
    const auto x_feas =
        prob.constraint_map().linear_part() * z_feas +
        prob.constraint_map().translation();

    const auto Q_slack = make_smat<3>(Vectord<6>{x_feas(0), x_feas(1), x_feas(2),
                                                 x_feas(3), x_feas(4), x_feas(5)});
    REQUIRE(Q_slack(0)(0) > 0.0);
    REQUIRE(Q_slack(1)(1) > 0.0);
    REQUIRE(Q_slack(2)(2) > 0.0);

    const auto cert_slack = make_smat<3>(
        Vectord<6>{x_feas(6), x_feas(7), x_feas(8),
                   x_feas(9), x_feas(10), x_feas(11)});
    const auto cert_sym = SymmetricMatrix(cert_slack, 1e-6);
    REQUIRE(cert_sym.is_valid());
    REQUIRE(cert_sym.is_psd());
  }
}
