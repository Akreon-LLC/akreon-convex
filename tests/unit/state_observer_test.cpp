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
static constexpr auto state_observer_solver_config() {
  auto cfg = config::PrimalDualIPMConfig<T>{};
  cfg.steps = 96;
  cfg.duality_gap_tol = T(1e-3);
  return cfg;
}

TEST_CASE("Luenberger Observer 2x2", "[state_observer_test]") {
  using Space = Matrixd<2, 2>;
  using PlantSS = StateSpace<TimeDomainType::CONTINUOUS, Space, Space, Space>;
  using Inputs = InputLayout<InputChannel<InputKind::CONTROL, 0, 2>>;
  using Outputs = OutputLayout<OutputChannel<OutputKind::MEASUREMENT, 0, 2>>;
  using Plant = PlantModel<PlantSS, Inputs, Outputs>;
  using QSpace = SymmetricMatrix<Space>;
  constexpr auto Q = Variable<SymMatVarMeta<Space>>();
  constexpr auto I = make_identity<double, 2>();
  constexpr auto Z = Matrixd<2, 2>{};
  constexpr auto MatBuilder = mat_traits<Space>();

  SECTION("Hand-checkable canonical LMI") {
    constexpr auto A = Z;
    constexpr auto B = I;
    constexpr auto C = I;
    constexpr auto plant = Plant(PlantSS(A, B, C));

    constexpr auto obs = LuenbergerObserver<decltype(plant)>();
    constexpr auto lmi = obs(plant, Q);

    constexpr auto builder = conic::create_empty_cblk_builder<QSpace>();
    constexpr auto cone_block =
        lmi.template canonicalize<AtomType::SET>(builder).cb;

    constexpr auto F_exp =
        Matrixd<6, 3>{Vectord<3>{1.0, 0.0, 0.0}, Vectord<3>{0.0, 1.0, 0.0},
                      Vectord<3>{0.0, 0.0, 1.0}, Vectord<3>{0.0, 0.0, 0.0},
                      Vectord<3>{0.0, 0.0, 0.0}, Vectord<3>{0.0, 0.0, 0.0}};
    STATIC_REQUIRE(all_equal(cone_block.map.linear_part(), F_exp));
    STATIC_REQUIRE(cone_block.map.translation() ==
                   Vectord<6>{-1e-6, 0.0, -1e-6, 1.0, 0.0, 1.0});

    constexpr auto cones = cone_block.cones;
    constexpr auto sz = std::tuple_size<decltype(cones)>::value;
    STATIC_REQUIRE(sz == 2);
    STATIC_REQUIRE(cone_block.validate());
  }

  SECTION("Solver finds feasible Q and stabilizing observer gain") {
    constexpr auto A = MatBuilder(0.2, 0.0, 0.0, 0.1);
    constexpr auto B = I;
    constexpr auto C = I;
    constexpr auto plant = Plant(PlantSS(A, B, C));

    constexpr auto obs = LuenbergerObserver<decltype(plant)>();
    constexpr auto prob = make_observer_problem(plant, Q);

    STATIC_REQUIRE(is_problem_set_v<decltype(prob)>);
    STATIC_REQUIRE(prob.validate());
    STATIC_REQUIRE(std::tuple_size<decltype(prob.constraint_cones())>::value ==
                   2);

    const PrimalDualIPM solver(prob);
    const auto& res = solver.res;
#ifdef AKREON_REPORT_OUTPUT
    const auto& trace = solver.trace;
    if (res.tag != SolverResult::OPTIMAL || res.stalled) log(res, trace);
#endif

    REQUIRE(res.tag == SolverResult::OPTIMAL);
    REQUIRE(!res.stalled);

    const auto Q_res = make_smat<2>(res.iterate.z);
    REQUIRE(all_equal(Q_res, transpose(Q_res), rel_tol));
    REQUIRE(Q_res(0)(0) > 1e-6);
    REQUIRE(Q_res(1)(1) > 1e-6);

    const auto [L_res, L_event] = obs.L(plant, Q_res);
    REQUIRE(L_event == linalg::NumericalEvent::NONE);
    REQUIRE(L_res(0)(0) > 0.0);
    REQUIRE(L_res(1)(1) > 0.0);
    REQUIRE(constexpr_abs(L_res(0)(1)) < 1e-3);
    REQUIRE(constexpr_abs(L_res(1)(0)) < 1e-3);

    const auto error_dynamics = A - L_res * C;
    REQUIRE(error_dynamics(0)(0) < 0.0);
    REQUIRE(error_dynamics(1)(1) < 0.0);

    const auto cert =
        (-1.0) * (transpose(A) * Q_res + Q_res * A) + transpose(C) * C;
    const auto cert_sym = linalg::SymmetricMatrix(cert);
    REQUIRE(cert_sym.is_valid());
    REQUIRE(cert_sym.is_psd());

    const auto lyap =
        (-1.0) * (transpose(error_dynamics) * Q_res + Q_res * error_dynamics);
    const auto lyap_sym = linalg::SymmetricMatrix(lyap);
    REQUIRE(lyap_sym.is_valid());
    REQUIRE(lyap_sym.is_psd());
  }

  SECTION("Explicit feasible point satisfies canonical constraints") {
    constexpr auto A = MatBuilder(0.2, 0.0, 0.0, 0.1);
    constexpr auto B = I;
    constexpr auto C = I;
    constexpr auto plant = Plant(PlantSS(A, B, C));

    constexpr auto prob = make_observer_problem(plant, Q);

    STATIC_REQUIRE(is_problem_set_v<decltype(prob)>);
    STATIC_REQUIRE(prob.validate());

    const auto Q_feas = I;
    const auto z_feas = get_svec(Q_feas);
    const auto x_feas = prob.constraint_map().linear_part() * z_feas +
                        prob.constraint_map().translation();

    const auto Q_slack =
        make_smat<2>(Vectord<3>{x_feas(0), x_feas(1), x_feas(2)});
    REQUIRE(Q_slack(0)(0) > 0.0);
    REQUIRE(Q_slack(1)(1) > 0.0);

    const auto cert_slack =
        make_smat<2>(Vectord<3>{x_feas(3), x_feas(4), x_feas(5)});
    const auto cert_sym = SymmetricMatrix(cert_slack, 1e-6);
    REQUIRE(cert_sym.is_valid());
    REQUIRE(cert_sym.is_psd());
  }
}

TEST_CASE("Luenberger Observer 3x3", "[state_observer_test]") {
  using Space = Matrixd<3, 3>;
  using PlantSS = StateSpace<TimeDomainType::CONTINUOUS, Space, Space, Space>;
  using Inputs = InputLayout<InputChannel<InputKind::CONTROL, 0, 3>>;
  using Outputs = OutputLayout<OutputChannel<OutputKind::MEASUREMENT, 0, 3>>;
  using Plant = PlantModel<PlantSS, Inputs, Outputs>;
  constexpr auto Q = Variable<SymMatVarMeta<Space>>();
  constexpr auto I = make_identity<double, 3>();
  constexpr auto MatBuilder = mat_traits<Space>();

  SECTION("Solver finds feasible Q on coupled observable plant") {
    constexpr auto A =
        MatBuilder(0.2, 0.05, 0.0, 0.0, 0.1, 0.04, 0.0, 0.0, 0.15);
    constexpr auto B = I;
    constexpr auto C = I;
    constexpr auto plant = Plant(PlantSS(A, B, C));

    constexpr auto obs = LuenbergerObserver<decltype(plant)>();
    constexpr auto prob = make_observer_problem(plant, Q);

    STATIC_REQUIRE(is_problem_set_v<decltype(prob)>);
    STATIC_REQUIRE(prob.validate());
    STATIC_REQUIRE(std::tuple_size<decltype(prob.constraint_cones())>::value ==
                   2);

    const PrimalDualIPM solver(prob);
    const auto& res = solver.res;
#ifdef AKREON_REPORT_OUTPUT
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

    const auto cert =
        (-1.0) * (transpose(A) * Q_res + Q_res * A) + transpose(C) * C;
    const auto cert_sym = linalg::SymmetricMatrix(cert);
    REQUIRE(cert_sym.is_valid());
    REQUIRE(cert_sym.is_psd());

    const auto [L_res, L_event] = obs.L(plant, Q_res);
    REQUIRE(L_event == linalg::NumericalEvent::NONE);

    const auto error_dynamics = A - L_res * C;
    const auto lyap =
        (-1.0) * (transpose(error_dynamics) * Q_res + Q_res * error_dynamics);
    const auto lyap_sym = linalg::SymmetricMatrix(lyap, 1e-6);
    REQUIRE(lyap_sym.is_valid());
    REQUIRE(lyap_sym.is_psd());
  }

  SECTION("Explicit feasible 3x3 point satisfies canonical constraints") {
    constexpr auto A =
        MatBuilder(0.2, 0.05, 0.0, 0.0, 0.1, 0.04, 0.0, 0.0, 0.15);
    constexpr auto B = I;
    constexpr auto C = I;
    constexpr auto plant = Plant(PlantSS(A, B, C));

    constexpr auto prob = make_observer_problem(plant, Q);

    STATIC_REQUIRE(is_problem_set_v<decltype(prob)>);
    STATIC_REQUIRE(prob.validate());

    const auto Q_feas = I;
    const auto z_feas = get_svec(Q_feas);
    const auto x_feas = prob.constraint_map().linear_part() * z_feas +
                        prob.constraint_map().translation();

    const auto Q_slack = make_smat<3>(Vectord<6>{
        x_feas(0), x_feas(1), x_feas(2), x_feas(3), x_feas(4), x_feas(5)});
    REQUIRE(Q_slack(0)(0) > 0.0);
    REQUIRE(Q_slack(1)(1) > 0.0);
    REQUIRE(Q_slack(2)(2) > 0.0);

    const auto cert_slack = make_smat<3>(Vectord<6>{
        x_feas(6), x_feas(7), x_feas(8), x_feas(9), x_feas(10), x_feas(11)});
    const auto cert_sym = SymmetricMatrix(cert_slack, 1e-6);
    REQUIRE(cert_sym.is_valid());
    REQUIRE(cert_sym.is_psd());
  }
}

TEST_CASE("H2 State Observer synthesis 2x2", "[state_observer_test]") {
  using StateSpaceMat = Matrixd<2, 2>;
  using InputMat = Matrixd<2, 2>;
  using OutputMat = Matrixd<2, 2>;
  using SS = StateSpace<TimeDomainType::CONTINUOUS, StateSpaceMat, InputMat,
                        OutputMat>;
  using Inputs = InputLayout<InputChannel<InputKind::DISTURBANCE, 0, 1>,
                             InputChannel<InputKind::CONTROL, 1, 1>>;
  using Outputs = OutputLayout<OutputChannel<OutputKind::MEASUREMENT, 0, 1>,
                               OutputChannel<OutputKind::PERFORMANCE, 1, 1>>;
  using Plant = PlantModel<SS, Inputs, Outputs>;

  constexpr auto MatBuilder = mat_traits<StateSpaceMat>();
  constexpr auto A = MatBuilder(-1.0, 0.0, 0.0, -2.0);
  constexpr auto B = InputMat{Vectord<2>{1.0, 0.0}, Vectord<2>{0.0, 1.0}};
  constexpr auto C = OutputMat{Vectord<2>{1.0, 0.0}, Vectord<2>{1.0, 0.0}};
  constexpr auto D = Matrixd<2, 2>{Vectord<2>{0.0, 0.0}, Vectord<2>{0.0, 0.0}};
  constexpr auto plant = Plant(SS(A, B, C, D));

  using H2 = H2StateObserver<decltype(plant)>;
  using DecVarSpace = RowVectord<H2::DecVarDim>;
  constexpr auto Z_hat = Variable<VarMeta<DecVarSpace>>();
  const auto prob = make_h2_state_observer_problem(plant, Z_hat);

  REQUIRE(is_problem_set_v<decltype(prob)>);
  REQUIRE(prob.validate());
  REQUIRE(std::tuple_size<decltype(prob.constraint_cones())>::value == 2);

  const auto c = prob.linear_objective();
  const auto c_exp = RowVectord<H2::DecVarDim>{
      Vectord<H2::DecVarDim>{0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
  REQUIRE(c == c_exp);

  using Solver = PrimalDualIPM<decltype(prob)>;
  const auto cfg = state_observer_solver_config<double>();
  const Solver solver(prob, cfg);
  const auto& res = solver.res;
#ifdef AKREON_REPORT_OUTPUT
  const auto& trace = solver.trace;
  if (res.tag != SolverResult::OPTIMAL || res.stalled) log(res, trace);
#endif

  REQUIRE(res.tag == SolverResult::OPTIMAL);
  REQUIRE(!res.stalled);

  const auto q_wz_split = linalg::splice<H2::QDim>(res.iterate.z);
  const auto Q_res = make_smat<2>(std::get<0>(q_wz_split));
  const auto w_z_split = linalg::splice<H2::WDim>(std::get<1>(q_wz_split));
  ColVectord<2> W_res{};
  for (size_t i = 0; i < H2::WDim; ++i) W_res(i)(0) = std::get<0>(w_z_split)(i);
  const auto Z_res = make_smat<1>(std::get<1>(w_z_split));

  REQUIRE(all_equal(Q_res, transpose(Q_res), rel_tol));
  REQUIRE(Q_res(0)(0) > 1e-6);
  REQUIRE(Q_res(1)(1) > 1e-6);
  REQUIRE(Z_res(0)(0) > 1e-6);

  const auto Q_sym = SymmetricMatrix(Q_res, 1e-6);
  REQUIRE(Q_sym.is_valid());
  REQUIRE(Q_sym.is_psd());

  constexpr auto h2 = H2();
  const auto [L_res, L_event] = h2.L(Q_res, W_res);
  REQUIRE(L_event == linalg::NumericalEvent::NONE);
  REQUIRE(all_equal(W_res, Q_res * L_res, 1e-4));

  constexpr auto B_w = ColVectord<2>{Vectord<1>{1.0}, Vectord<1>{0.0}};
  constexpr auto C_y = RowVectord<2>{Vectord<2>{1.0, 0.0}};
  constexpr auto C_e = RowVectord<2>{Vectord<2>{1.0, 0.0}};
  constexpr auto D_yw = RowVectord<1>{Vectord<1>{0.0}};
  const auto stability_cert = (-1.0) * (transpose(A) * Q_res + Q_res * A -
                                        transpose(C_y) * transpose(W_res) -
                                        W_res * C_y + transpose(C_e) * C_e);
  const auto stability_sym = SymmetricMatrix(stability_cert, 1e-6);
  REQUIRE(stability_sym.is_valid());
  REQUIRE(stability_sym.is_psd());

  const auto bottom_left = Q_res * B_w - W_res * D_yw;
  const auto epigraph = Matrixd<3, 3>{
      Vectord<3>{Z_res(0)(0), bottom_left(0)(0), bottom_left(1)(0)},
      Vectord<3>{bottom_left(0)(0), Q_res(0)(0), Q_res(0)(1)},
      Vectord<3>{bottom_left(1)(0), Q_res(1)(0), Q_res(1)(1)}};
  const auto epigraph_sym = SymmetricMatrix(epigraph, 1e-6);
  REQUIRE(epigraph_sym.is_valid());
  REQUIRE(epigraph_sym.is_psd());

  const auto error_dynamics = A - L_res * C_y;
  const auto lyap = (-1.0) * (transpose(error_dynamics) * Q_res +
                              Q_res * error_dynamics + transpose(C_e) * C_e);
  const auto lyap_sym = SymmetricMatrix(lyap, 1e-6);
  REQUIRE(lyap_sym.is_valid());
  REQUIRE(lyap_sym.is_psd());
}
