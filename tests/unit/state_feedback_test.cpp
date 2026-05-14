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
static constexpr auto state_feedback_solver_config() {
  auto cfg = config::PrimalDualIPMConfig<T>{};
  cfg.steps = 64;
  cfg.duality_gap_tol = T(1e-4);
  return cfg;
}

TEST_CASE("State Feedback 2x2", "[state_feedback_test]") {
  using Space = Matrixd<2, 2>;
  using InputSpace = Matrixd<2, 1>;
  using OutputSpace = RowVectord<2>;
  using FullPlantSS =
      StateSpace<TimeDomainType::CONTINUOUS, Space, Space, Space>;
  using FullInputs = InputLayout<InputChannel<InputKind::CONTROL, 0, 2>>;
  using FullOutputs =
      OutputLayout<OutputChannel<OutputKind::MEASUREMENT, 0, 2>>;
  using FullPlant = PlantModel<FullPlantSS, FullInputs, FullOutputs>;
  using SolverPlantSS =
      StateSpace<TimeDomainType::CONTINUOUS, Space, InputSpace, OutputSpace>;
  using SolverInputs = InputLayout<InputChannel<InputKind::CONTROL, 0, 1>>;
  using SolverOutputs =
      OutputLayout<OutputChannel<OutputKind::MEASUREMENT, 0, 1>>;
  using SolverPlant = PlantModel<SolverPlantSS, SolverInputs, SolverOutputs>;

  constexpr auto MatBuilder = mat_traits<Space>();
  constexpr auto BCol = ColVectord<2>{Vectord<1>{1.0}, Vectord<1>{0.5}};
  constexpr auto CRow = RowVectord<2>{Vectord<2>{1.0, 0.0}};

  static constexpr size_t PDim = 3;
  static constexpr size_t YDim = 2;
  static constexpr size_t ZDim = PDim + YDim;
  using ZSpace = RowVectord<ZDim>;
  constexpr auto Z = Variable<VarMeta<ZSpace>>();

  static constexpr size_t FullInputZDim = PDim + 4;
  using FullInputZSpace = RowVectord<FullInputZDim>;
  constexpr auto FullInputZ = Variable<VarMeta<FullInputZSpace>>();

  SECTION("Hand-checkable merged canonical LMI") {
    constexpr auto A = Matrixd<2, 2>{};
    constexpr auto B = make_identity<double, 2>();
    constexpr auto C = make_identity<double, 2>();
    constexpr auto plant = FullPlant(FullPlantSS(A, B, C));

    constexpr auto sf = StateFeedback<decltype(plant)>();
    const auto lmi = sf(plant, FullInputZ);

    constexpr auto builder =
        conic::create_empty_cblk_builder<FullInputZSpace>();
    const auto cone_block =
        lmi.template canonicalize<AtomType::SET>(builder).cb;

    constexpr auto s2 = constexpr_sqrt_safe(2.0);
    constexpr auto F_exp =
        Matrixd<6, 7>{Vectord<7>{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                      Vectord<7>{0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                      Vectord<7>{0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
                      Vectord<7>{0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0},
                      Vectord<7>{0.0, 0.0, 0.0, 0.0, -s2, -s2, 0.0},
                      Vectord<7>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0}};

    REQUIRE(all_equal(cone_block.map.linear_part(), F_exp));
    REQUIRE(cone_block.map.translation() ==
            Vectord<6>{-1e-6, 0.0, -1e-6, 0.0, 0.0, 0.0});
    REQUIRE(std::tuple_size<decltype(cone_block.cones)>::value == 2);
    REQUIRE(cone_block.validate());
  }

  SECTION("Solver finds feasible P/Y and stabilizing K") {
    constexpr auto A = MatBuilder(-1.0, 0.0, 0.0, -2.0);
    constexpr auto plant = SolverPlant(SolverPlantSS(A, BCol, CRow));

    constexpr auto sf = StateFeedback<decltype(plant)>();
    constexpr auto P = Matrix<double, ZDim, ZDim>{
        Vectord<ZDim>{1.0, 0.0, 0.0, 0.0, 0.0},
        Vectord<ZDim>{0.0, 1.0, 0.0, 0.0, 0.0},
        Vectord<ZDim>{0.0, 0.0, 1.0, 0.0, 0.0},
        Vectord<ZDim>{0.0, 0.0, 0.0, 1.0, 0.0},
        Vectord<ZDim>{0.0, 0.0, 0.0, 0.0, 1.0},
    };
    constexpr auto q = Vectord<ZDim>{};
    constexpr auto obj = make_quad_form(P, q, Z);
    const auto prob = make_state_feedback_problem(plant, Z, obj);

    REQUIRE(is_problem_set_v<decltype(prob)>);
    REQUIRE(prob.validate());
    REQUIRE(std::tuple_size<decltype(prob.constraint_cones())>::value == 4);

    using Solver = PrimalDualIPM<decltype(prob)>;
    const auto cfg = state_feedback_solver_config<double>();
    const Solver solver(prob, cfg);
    const auto& res = solver.res;
#ifdef AKREON_REPORT_OUTPUT
    const auto& trace = solver.trace;
    if (res.tag != SolverResult::OPTIMAL || res.stalled) log(res, trace);
#endif

    REQUIRE(res.tag == SolverResult::OPTIMAL);
    REQUIRE(!res.stalled);

    const auto z_split = linalg::splice<PDim>(res.iterate.z);
    const auto P_res = make_smat<2>(std::get<0>(z_split));
    RowVectord<2> Y_res{};
    for (size_t j = 0; j < YDim; ++j) Y_res(0)(j) = std::get<1>(z_split)(j);

    REQUIRE(all_equal(P_res, transpose(P_res), rel_tol));
    REQUIRE(P_res(0)(0) > 1e-6);
    REQUIRE(P_res(1)(1) > 1e-6);

    const auto P_sym = SymmetricMatrix(P_res, 1e-6);
    REQUIRE(P_sym.is_valid());
    REQUIRE(P_sym.is_psd());

    const auto [K_res, K_event] = sf.K(P_res, Y_res);
    REQUIRE(K_event == linalg::NumericalEvent::NONE);
    REQUIRE(all_equal(Y_res, K_res * P_res, 1e-4));

    const auto cert =
        (-1.0) * (A * P_res + P_res * transpose(A) + BCol * Y_res +
                  transpose(Y_res) * transpose(BCol));
    const auto cert_sym = SymmetricMatrix(cert, 1e-6);
    REQUIRE(cert_sym.is_valid());
    REQUIRE(cert_sym.is_psd());

    const auto cl = A + BCol * K_res;
    const auto lyap = (-1.0) * (cl * P_res + P_res * transpose(cl));
    const auto lyap_sym = SymmetricMatrix(lyap, 1e-6);
    REQUIRE(lyap_sym.is_valid());
    REQUIRE(lyap_sym.is_psd());
  }
}

TEST_CASE("H2 State Feedback synthesis 2x2", "[state_feedback_test]") {
  using StateSpaceMat = Matrixd<2, 2>;
  using InputMat = Matrixd<2, 2>;
  using OutputMat = RowVectord<2>;
  using SS = StateSpace<TimeDomainType::CONTINUOUS, StateSpaceMat, InputMat,
                        OutputMat>;
  using Inputs = InputLayout<InputChannel<InputKind::DISTURBANCE, 0, 1>,
                             InputChannel<InputKind::CONTROL, 1, 1>>;
  using Outputs = OutputLayout<OutputChannel<OutputKind::PERFORMANCE, 0, 1>>;
  using Plant = PlantModel<SS, Inputs, Outputs>;

  constexpr auto MatBuilder = mat_traits<StateSpaceMat>();
  constexpr auto A = MatBuilder(-1.0, 0.0, 0.0, -2.0);
  constexpr auto B = InputMat{Vectord<2>{1.0, 0.0}, Vectord<2>{0.0, 1.0}};
  constexpr auto C = OutputMat{Vectord<2>{1.0, 0.0}};
  constexpr auto D = RowVectord<2>{Vectord<2>{0.0, 0.0}};
  constexpr auto plant = Plant(SS(A, B, C, D));

  using H2 = H2StateFeedback<decltype(plant)>;
  using DecVarSpace = RowVectord<H2::DecVarDim>;
  constexpr auto Z_hat = Variable<VarMeta<DecVarSpace>>();
  const auto prob = make_h2_state_feedback_problem(plant, Z_hat);

  REQUIRE(is_problem_set_v<decltype(prob)>);
  REQUIRE(prob.validate());
  REQUIRE(std::tuple_size<decltype(prob.constraint_cones())>::value == 2);

  const auto c = prob.linear_objective();
  const auto c_exp = RowVectord<H2::DecVarDim>{
      Vectord<H2::DecVarDim>{0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
  REQUIRE(c == c_exp);

  using Solver = PrimalDualIPM<decltype(prob)>;
  auto cfg = state_feedback_solver_config<double>();
  cfg.steps = 96;
  cfg.duality_gap_tol = 1e-3;
  const Solver solver(prob, cfg);
  const auto& res = solver.res;
#ifdef AKREON_REPORT_OUTPUT
  const auto& trace = solver.trace;
  if (res.tag != SolverResult::OPTIMAL || res.stalled) log(res, trace);
#endif

  REQUIRE(res.tag == SolverResult::OPTIMAL);
  REQUIRE(!res.stalled);

  const auto p_yz_split = linalg::splice<H2::PDim>(res.iterate.z);
  const auto P_res = make_smat<2>(std::get<0>(p_yz_split));
  const auto y_z_split = linalg::splice<H2::YDim>(std::get<1>(p_yz_split));
  RowVectord<2> Y_res{};
  for (size_t j = 0; j < H2::YDim; ++j) Y_res(0)(j) = std::get<0>(y_z_split)(j);
  const auto Z_res = make_smat<1>(std::get<1>(y_z_split));

  REQUIRE(all_equal(P_res, transpose(P_res), rel_tol));
  REQUIRE(P_res(0)(0) > 1e-6);
  REQUIRE(P_res(1)(1) > 1e-6);
  REQUIRE(Z_res(0)(0) > 1e-6);

  const auto P_sym = SymmetricMatrix(P_res, 1e-6);
  REQUIRE(P_sym.is_valid());
  REQUIRE(P_sym.is_psd());

  constexpr auto h2 = H2();
  const auto [K_res, K_event] = h2.K(P_res, Y_res);
  REQUIRE(K_event == linalg::NumericalEvent::NONE);
  REQUIRE(all_equal(Y_res, K_res * P_res, 1e-4));

  constexpr auto B_w = ColVectord<2>{Vectord<1>{1.0}, Vectord<1>{0.0}};
  constexpr auto B_u = ColVectord<2>{Vectord<1>{0.0}, Vectord<1>{1.0}};
  constexpr auto C_z = RowVectord<2>{Vectord<2>{1.0, 0.0}};
  constexpr auto D_u = RowVectord<1>{Vectord<1>{0.0}};
  const auto stability_cert =
      (-1.0) * (A * P_res + P_res * transpose(A) + B_u * Y_res +
                transpose(Y_res) * transpose(B_u) + B_w * transpose(B_w));
  const auto stability_sym = SymmetricMatrix(stability_cert, 1e-6);
  REQUIRE(stability_sym.is_valid());
  REQUIRE(stability_sym.is_psd());

  const auto top_right = C_z * P_res + D_u * Y_res;
  const auto epigraph =
      Matrixd<3, 3>{Vectord<3>{Z_res(0)(0), top_right(0)(0), top_right(0)(1)},
                    Vectord<3>{top_right(0)(0), P_res(0)(0), P_res(0)(1)},
                    Vectord<3>{top_right(0)(1), P_res(1)(0), P_res(1)(1)}};
  const auto epigraph_sym = SymmetricMatrix(epigraph, 1e-6);
  REQUIRE(epigraph_sym.is_valid());
  REQUIRE(epigraph_sym.is_psd());

  const auto cl = A + B_u * K_res;
  const auto lyap =
      (-1.0) * (cl * P_res + P_res * transpose(cl) + B_w * transpose(B_w));
  const auto lyap_sym = SymmetricMatrix(lyap, 1e-6);
  REQUIRE(lyap_sym.is_valid());
  REQUIRE(lyap_sym.is_psd());
}
