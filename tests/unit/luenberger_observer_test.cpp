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

#include "control/estimation/luenberger_observer.hpp"
#include "math.hpp"
#include "test_helpers.hpp"

using namespace linalg;
using namespace convex;
using namespace control;
using namespace unit_test;
using namespace config;

static constexpr auto rel_tol{1e-3};

TEST_CASE("Luenberger Observer 2x2", "[luenberger_observer_test]") {
  using Space = Matrixd<2, 2>;
  using QSpace = SymmetricMatrix<Space>;
  constexpr auto P = Variable<SymMatVarMeta<Space>>();
  constexpr auto I = make_identity<double, 2>();
  constexpr auto Z = Matrixd<2, 2>{};
  constexpr auto MatBuilder = mat_traits<Space>();

  SECTION("Hand-checkable canonical LMI") {
    constexpr auto A = Z;
    constexpr auto B = I;
    constexpr auto C = I;
    constexpr auto ss =
        StateSpace<TimeDomainType::CONTINUOUS, Space, Space, Space>(A, B, C);

    constexpr auto obs = LuenbergerObserver<decltype(ss)>();
    constexpr auto lmi = obs(ss, P);

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

  SECTION("Solver finds feasible P and stabilizing observer gain") {
    constexpr auto A = MatBuilder(0.2, 0.0, 0.0, 0.1);
    constexpr auto B = I;
    constexpr auto C = I;
    constexpr auto ss =
        StateSpace<TimeDomainType::CONTINUOUS, Space, Space, Space>(A, B, C);

    constexpr auto obs = LuenbergerObserver<decltype(ss)>();
    constexpr auto prob = make_observer_problem(ss, P);

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

    const auto P_res = make_smat<2>(res.iterate.z);
    REQUIRE(all_equal(P_res, transpose(P_res), rel_tol));
    REQUIRE(P_res(0)(0) > 1e-6);
    REQUIRE(P_res(1)(1) > 1e-6);

    const auto [L_res, L_event] = obs.L(ss, P_res);
    REQUIRE(L_event == linalg::NumericalEvent::NONE);
    REQUIRE(L_res(0)(0) > 0.0);
    REQUIRE(L_res(1)(1) > 0.0);
    REQUIRE(constexpr_abs(L_res(0)(1)) < 1e-3);
    REQUIRE(constexpr_abs(L_res(1)(0)) < 1e-3);

    const auto error_dynamics = A - L_res * C;
    REQUIRE(error_dynamics(0)(0) < 0.0);
    REQUIRE(error_dynamics(1)(1) < 0.0);

    const auto cert =
        (-1.0) * (transpose(A) * P_res + P_res * A) + transpose(C) * C;
    const auto cert_sym = linalg::SymmetricMatrix(cert);
    REQUIRE(cert_sym.is_valid());
    REQUIRE(cert_sym.is_psd());

    const auto lyap =
        (-1.0) * (transpose(error_dynamics) * P_res + P_res * error_dynamics);
    const auto lyap_sym = linalg::SymmetricMatrix(lyap);
    REQUIRE(lyap_sym.is_valid());
    REQUIRE(lyap_sym.is_psd());
  }

  SECTION("Explicit feasible point satisfies canonical constraints") {
    constexpr auto A = MatBuilder(0.2, 0.0, 0.0, 0.1);
    constexpr auto B = I;
    constexpr auto C = I;
    constexpr auto ss =
        StateSpace<TimeDomainType::CONTINUOUS, Space, Space, Space>(A, B, C);

    constexpr auto prob = make_observer_problem(ss, P);

    STATIC_REQUIRE(is_problem_set_v<decltype(prob)>);
    STATIC_REQUIRE(prob.validate());

    const auto P_feas = I;
    const auto z_feas = get_svec(P_feas);
    const auto x_feas =
        prob.constraint_map().linear_part() * z_feas +
        prob.constraint_map().translation();

    const auto P_slack = make_smat<2>(Vectord<3>{x_feas(0), x_feas(1), x_feas(2)});
    REQUIRE(P_slack(0)(0) > 0.0);
    REQUIRE(P_slack(1)(1) > 0.0);

    const auto cert_slack = make_smat<2>(
        Vectord<3>{x_feas(3), x_feas(4), x_feas(5)});
    const auto cert_sym = SymmetricMatrix(cert_slack, 1e-6);
    REQUIRE(cert_sym.is_valid());
    REQUIRE(cert_sym.is_psd());
  }
}

TEST_CASE("Luenberger Observer 3x3", "[luenberger_observer_test]") {
  using Space = Matrixd<3, 3>;
  constexpr auto P = Variable<SymMatVarMeta<Space>>();
  constexpr auto I = make_identity<double, 3>();
  constexpr auto MatBuilder = mat_traits<Space>();

  SECTION("Solver finds feasible P on coupled observable plant") {
    constexpr auto A =
        MatBuilder(0.2, 0.05, 0.0, 0.0, 0.1, 0.04, 0.0, 0.0, 0.15);
    constexpr auto B = I;
    constexpr auto C = I;
    constexpr auto ss =
        StateSpace<TimeDomainType::CONTINUOUS, Space, Space, Space>(A, B, C);

    constexpr auto obs = LuenbergerObserver<decltype(ss)>();
    constexpr auto prob = make_observer_problem(ss, P);

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

    const auto P_res = make_smat<3>(res.iterate.z);
    REQUIRE(all_equal(P_res, transpose(P_res), rel_tol));
    REQUIRE(P_res(0)(0) > 1e-6);
    REQUIRE(P_res(1)(1) > 1e-6);
    REQUIRE(P_res(2)(2) > 1e-6);

    const auto cert =
        (-1.0) * (transpose(A) * P_res + P_res * A) + transpose(C) * C;
    const auto cert_sym = linalg::SymmetricMatrix(cert);
    REQUIRE(cert_sym.is_valid());
    REQUIRE(cert_sym.is_psd());

    const auto [L_res, L_event] = obs.L(ss, P_res);
    REQUIRE(L_event == linalg::NumericalEvent::NONE);

    const auto error_dynamics = A - L_res * C;
    const auto lyap =
        (-1.0) * (transpose(error_dynamics) * P_res + P_res * error_dynamics);
    const auto lyap_sym = linalg::SymmetricMatrix(lyap, 1e-6);
    REQUIRE(lyap_sym.is_valid());
    REQUIRE(lyap_sym.is_psd());
  }

  SECTION("Explicit feasible 3x3 point satisfies canonical constraints") {
    constexpr auto A =
        MatBuilder(0.2, 0.05, 0.0, 0.0, 0.1, 0.04, 0.0, 0.0, 0.15);
    constexpr auto B = I;
    constexpr auto C = I;
    constexpr auto ss =
        StateSpace<TimeDomainType::CONTINUOUS, Space, Space, Space>(A, B, C);

    constexpr auto prob = make_observer_problem(ss, P);

    STATIC_REQUIRE(is_problem_set_v<decltype(prob)>);
    STATIC_REQUIRE(prob.validate());

    const auto P_feas = I;
    const auto z_feas = get_svec(P_feas);
    const auto x_feas =
        prob.constraint_map().linear_part() * z_feas +
        prob.constraint_map().translation();

    const auto P_slack = make_smat<3>(Vectord<6>{x_feas(0), x_feas(1), x_feas(2),
                                                 x_feas(3), x_feas(4), x_feas(5)});
    REQUIRE(P_slack(0)(0) > 0.0);
    REQUIRE(P_slack(1)(1) > 0.0);
    REQUIRE(P_slack(2)(2) > 0.0);

    const auto cert_slack = make_smat<3>(
        Vectord<6>{x_feas(6), x_feas(7), x_feas(8),
                   x_feas(9), x_feas(10), x_feas(11)});
    const auto cert_sym = SymmetricMatrix(cert_slack, 1e-6);
    REQUIRE(cert_sym.is_valid());
    REQUIRE(cert_sym.is_psd());
  }
}
