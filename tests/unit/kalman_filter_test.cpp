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

template <typename T>
static constexpr auto lqe_solver_config() {
  auto cfg = config::PrimalDualIPMConfig<T>{};
  cfg.steps = 64;
  cfg.duality_gap_tol = T(2e-4);
  return cfg;
}

TEST_CASE("Small Kalman Filter 2x2", "[kalman_filter_test]") {
  using Space = Matrixd<2, 2>;
  using PSpace = SymmetricMatrix<Space>;
  constexpr auto P = Variable<SymMatVarMeta<Space>>();
  constexpr auto MatBuilder = mat_traits<Space>();

  SECTION("Canonical LMI matches estimator Riccati Schur form") {
    constexpr auto I = make_identity<double, 2>();
    constexpr auto Z = Matrixd<2, 2>{};
    constexpr auto Q = MatBuilder(1.0, 0.0, 0.0, 1.0);
    constexpr auto R = MatBuilder(1.0, 0.0, 0.0, 1.0);
    constexpr auto A = Z;
    constexpr auto B = I;
    constexpr auto C = I;
    constexpr auto ss =
        StateSpace<TimeDomainType::CONTINUOUS, Space, Space, Space>(A, B, C);

    constexpr auto kf = KalmanFilter<decltype(ss), decltype(Q), decltype(R)>();
    constexpr auto lmi = kf(ss, Q, R, P);

    STATIC_REQUIRE(lmi.is_well_formed());

    constexpr auto builder = conic::create_empty_cblk_builder<PSpace>();
    constexpr auto cone_block =
        lmi.template canonicalize<AtomType::SET>(builder).cb;

    constexpr auto sqrt2 = constexpr_sqrt_safe(2.0);
    constexpr auto F_exp = Matrixd<10, 3>{
        Vectord<3>{0.0, 0.0, 0.0},  Vectord<3>{0.0, 0.0, 0.0},
        Vectord<3>{0.0, 0.0, 0.0},  Vectord<3>{-sqrt2, 0.0, 0.0},
        Vectord<3>{0.0, -1.0, 0.0}, Vectord<3>{0.0, 0.0, 0.0},
        Vectord<3>{0.0, -1.0, 0.0}, Vectord<3>{0.0, 0.0, -sqrt2},
        Vectord<3>{0.0, 0.0, 0.0},  Vectord<3>{0.0, 0.0, 0.0},
    };

    STATIC_REQUIRE(all_equal(cone_block.map.linear_part(), F_exp));
    STATIC_REQUIRE(
        cone_block.map.translation() ==
        Vectord<10>{-1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0});

    constexpr auto cones = cone_block.cones;
    constexpr auto sz = std::tuple_size<decltype(cones)>::value;
    STATIC_REQUIRE(sz == 1);
    STATIC_REQUIRE(cone_block.validate());
  }

  SECTION("Boundary witness satisfies current estimator Schur form") {
    constexpr auto I = make_identity<double, 2>();
    constexpr auto Q = MatBuilder(1.0, 0.0, 0.0, 1.0);
    constexpr auto R = MatBuilder(1.0, 0.0, 0.0, 1.0);
    constexpr auto A = -1.0 * I;
    constexpr auto B = I;
    constexpr auto C = I;
    constexpr auto ss =
        StateSpace<TimeDomainType::CONTINUOUS, Space, Space, Space>(A, B, C);
    constexpr auto kf = KalmanFilter<decltype(ss), decltype(Q), decltype(R)>();

    const auto P_star = I;
    const auto [L_res, L_event] = kf.L(P_star, ss, R);
    REQUIRE(L_event == linalg::NumericalEvent::NONE);
    REQUIRE(all_equal(L_res, I, 1e-6));

    Matrixd<4, 4> lmi_block{};
    lmi_block = set_block<0, 2, 0, 2>(
        lmi_block, (-1.0) * (A * P_star + P_star * transpose(A) + Q));
    lmi_block =
        set_block<0, 2, 2, 4>(lmi_block, (-1.0) * (P_star * transpose(C)));
    lmi_block = set_block<2, 4, 0, 2>(lmi_block, (-1.0) * (C * P_star));
    lmi_block = set_block<2, 4, 2, 4>(lmi_block, R);

    const auto lmi_sym = SymmetricMatrix(lmi_block, 1e-6);
    REQUIRE(lmi_sym.is_valid());
    REQUIRE(lmi_sym.is_psd());

    const auto schur_res = A * P_star + P_star * transpose(A) + Q +
                           P_star * transpose(C) * C * P_star;
    REQUIRE(all_equal(schur_res, Matrixd<2, 2>{}, 1e-6));
  }

  SECTION("Explicit feasible point satisfies estimator Schur complement") {
    constexpr auto I = make_identity<double, 2>();
    constexpr auto Q = MatBuilder(1.0, 0.0, 0.0, 1.0);
    constexpr auto R = MatBuilder(1.0, 0.0, 0.0, 1.0);
    constexpr auto A = -1.0 * I;
    constexpr auto B = I;
    constexpr auto C = I;
    constexpr auto ss =
        StateSpace<TimeDomainType::CONTINUOUS, Space, Space, Space>(A, B, C);

    constexpr auto kf = KalmanFilter<decltype(ss), decltype(Q), decltype(R)>();
    constexpr auto lmi = kf(ss, Q, R, P);

    constexpr auto P_geq =
        Inequality<InequalityType::GEQ, decltype(P), Space>(P, 1e-6 * I);
    constexpr auto prob_constr =
        make_problem(P).add_constraint(P_geq).in_set(lmi);

    STATIC_REQUIRE(prob_constr.validate());

    const auto P_feas = I + 1e-3 * I;

    const auto P_sym = SymmetricMatrix(P_feas, 1e-6);
    REQUIRE(P_sym.is_valid());
    REQUIRE(P_sym.is_pd());

    Matrixd<4, 4> lmi_slack{};
    lmi_slack = set_block<0, 2, 0, 2>(
        lmi_slack, (-1.0) * (A * P_feas + P_feas * transpose(A) + Q));
    lmi_slack =
        set_block<0, 2, 2, 4>(lmi_slack, (-1.0) * (P_feas * transpose(C)));
    lmi_slack = set_block<2, 4, 0, 2>(lmi_slack, (-1.0) * (C * P_feas));
    lmi_slack = set_block<2, 4, 2, 4>(lmi_slack, R);
    const auto lmi_sym = SymmetricMatrix(lmi_slack, 1e-6);
    REQUIRE(lmi_sym.is_valid());
    REQUIRE(lmi_sym.is_psd());
  }

  SECTION("Solver converges on stable fully observed plant") {
    constexpr auto I = make_identity<double, 2>();
    constexpr auto Q = MatBuilder(1.0, 0.0, 0.0, 1.0);
    constexpr auto R = MatBuilder(1.0, 0.0, 0.0, 1.0);
    constexpr auto A = -1.0 * I;
    constexpr auto B = I;
    constexpr auto C = I;
    constexpr auto ss =
        StateSpace<TimeDomainType::CONTINUOUS, Space, Space, Space>(A, B, C);

    constexpr auto kf = KalmanFilter<decltype(ss), decltype(Q), decltype(R)>();
    constexpr auto prob = make_lqe_problem(ss, Q, R, P);

    STATIC_REQUIRE(is_problem_set_v<decltype(prob)>);
    REQUIRE(prob.validate());

    using Solver = PrimalDualIPM<decltype(prob)>;
    const auto cfg = lqe_solver_config<double>();
    const Solver solver(prob, cfg);
    const auto& res = solver.res;
#ifdef AKREON_REPORT_OUTPUT
    const auto& trace = solver.trace;
    if (res.tag != SolverResult::OPTIMAL || res.stalled) log(res, trace);
#endif

    REQUIRE(res.tag == SolverResult::OPTIMAL);
    REQUIRE(!res.stalled);

    const auto P_res = make_smat<2>(res.iterate.z);
    REQUIRE(all_equal(P_res, transpose(P_res), rel_tol));
    REQUIRE(P_res(0)(0) > 1e-6);
    REQUIRE(P_res(1)(1) > 1e-6);

    const auto [L_res, L_event] = kf.L(P_res, ss, R);
    REQUIRE(L_event == linalg::NumericalEvent::NONE);
    REQUIRE(L_res(0)(0) > 0.0);
    REQUIRE(L_res(1)(1) > 0.0);

    const auto cert = (-1.0) * (A * P_res + P_res * transpose(A) + Q) +
                      P_res * transpose(C) * C * P_res;
    const auto cert_sym = SymmetricMatrix(cert, 1e-5);
    REQUIRE(cert_sym.is_valid());
    REQUIRE(cert_sym.is_psd());
  }

  SECTION("Unstable observable plant is infeasible under current P-only Kalman "
          "LMI") {
    constexpr auto I = make_identity<double, 2>();
    constexpr auto Q = MatBuilder(1.0, 0.0, 0.0, 1.0);
    constexpr auto R = MatBuilder(1.0, 0.0, 0.0, 1.0);
    constexpr auto A = 0.2 * I;
    constexpr auto B = I;
    constexpr auto C = I;
    constexpr auto ss =
        StateSpace<TimeDomainType::CONTINUOUS, Space, Space, Space>(A, B, C);

    constexpr auto prob = make_lqe_problem(ss, Q, R, P);

    STATIC_REQUIRE(is_problem_set_v<decltype(prob)>);
    REQUIRE(prob.validate());

    using Solver = PrimalDualIPM<decltype(prob)>;
    const auto cfg = lqe_solver_config<double>();
    const Solver solver(prob, cfg);
    const auto& res = solver.res;
#ifdef AKREON_REPORT_OUTPUT
    const auto& trace = solver.trace;
    if (res.tag == SolverResult::OPTIMAL && !res.stalled) log(res, trace);
#endif

    REQUIRE(res.tag != SolverResult::OPTIMAL);
  }
}

TEST_CASE("Small Kalman Filter 3x3", "[kalman_filter_test]") {
  using Space = Matrixd<3, 3>;
  constexpr auto P = Variable<SymMatVarMeta<Space>>();
  constexpr auto I = make_identity<double, 3>();

  SECTION("Explicit 3x3 feasible point satisfies current estimator LMI") {
    constexpr auto Q = I;
    constexpr auto R = I;
    constexpr auto A = -1.0 * I;
    constexpr auto B = I;
    constexpr auto C = I;
    constexpr auto ss =
        StateSpace<TimeDomainType::CONTINUOUS, Space, Space, Space>(A, B, C);

    constexpr auto kf = KalmanFilter<decltype(ss), decltype(Q), decltype(R)>();
    constexpr auto lmi = kf(ss, Q, R, P);
    constexpr auto P_geq =
        Inequality<InequalityType::GEQ, decltype(P), Space>(P, 1e-6 * I);
    constexpr auto prob_constr =
        make_problem(P).add_constraint(P_geq).in_set(lmi);

    STATIC_REQUIRE(prob_constr.validate());

    const auto P_feas = I;
    const auto [L_res, L_event] = kf.L(P_feas, ss, R);
    REQUIRE(L_event == linalg::NumericalEvent::NONE);
    REQUIRE(all_equal(L_res, I, 1e-6));

    const auto cert = (-1.0) * (A * P_feas + P_feas * transpose(A) + Q) +
                      P_feas * transpose(C) * C * P_feas;
    const auto cert_sym = SymmetricMatrix(cert, 1e-5);
    REQUIRE(cert_sym.is_valid());
    REQUIRE(cert_sym.is_psd());
  }
}
