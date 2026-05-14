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
static constexpr auto pldi_solver_config() {
  auto cfg = config::PrimalDualIPMConfig<T>{};
  cfg.steps = 64;
  cfg.duality_gap_tol = T(1e-4);
  return cfg;
}

TEST_CASE("Polytopic State Feedback 2x2", "[polytopic_state_feedback_test]") {
  using Space = Matrixd<2, 2>;
  using FullPlantSS =
      StateSpace<TimeDomainType::CONTINUOUS, Space, Space, Space>;
  using FullInputs = InputLayout<InputChannel<InputKind::CONTROL, 0, 2>>;
  using FullOutputs =
      OutputLayout<OutputChannel<OutputKind::MEASUREMENT, 0, 2>>;
  using FullPlant = PlantModel<FullPlantSS, FullInputs, FullOutputs>;
  constexpr auto I = make_identity<double, 2>();
  constexpr auto Z = Matrixd<2, 2>{};
  constexpr auto MatBuilder = mat_traits<Space>();

  static constexpr size_t PDim = 3;
  static constexpr size_t YDim = 4;
  static constexpr size_t ZDim = PDim + YDim;
  using ZSpace = RowVectord<ZDim>;
  constexpr auto dec = Variable<VarMeta<ZSpace>>();

  SECTION("Hand-checkable merged canonical LMI") {
    constexpr auto A0 = Z;
    constexpr auto A1 = MatBuilder(-1.0, 0.0, 0.0, -2.0);
    constexpr auto B0 = I;
    constexpr auto B1 = I;
    constexpr auto C = I;

    constexpr auto plant0 = FullPlant(FullPlantSS(A0, B0, C));
    constexpr auto plant1 = FullPlant(FullPlantSS(A1, B1, C));
    constexpr auto poly = Vector<decltype(plant0), 2>{plant0, plant1};

    constexpr auto psf = PolytopicStateFeedback<decltype(plant0), 2>();
    const auto lmis = psf(poly, dec);

    constexpr auto builder = conic::create_empty_cblk_builder<ZSpace>();
    const auto canon = lmis.template canonicalize<AtomType::SET>(builder);
    const auto cone_block = canon.cb;

    constexpr auto s2 = constexpr_sqrt_safe(2.0);
    constexpr auto F_exp = Matrixd<9, 7>{
        Vectord<7>{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        Vectord<7>{0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        Vectord<7>{0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
        Vectord<7>{0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0},
        Vectord<7>{0.0, 0.0, 0.0, 0.0, -s2, -s2, 0.0},
        Vectord<7>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0},
        Vectord<7>{2.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0},
        Vectord<7>{0.0, 3.0, 0.0, 0.0, -s2, -s2, 0.0},
        Vectord<7>{0.0, 0.0, 4.0, 0.0, 0.0, 0.0, -2.0},
    };

    REQUIRE(all_equal(cone_block.map.linear_part(), F_exp));
    REQUIRE(cone_block.map.translation() ==
            Vectord<9>{-1e-6, 0.0, -1e-6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
    REQUIRE(std::tuple_size<decltype(cone_block.cones)>::value == 3);
    REQUIRE(cone_block.validate());
  }

  SECTION("Solver finds feasible merged z with PSD P block") {
    using InputSpace = Matrixd<2, 1>;
    using OutputSpace = RowVectord<2>;
    using SolverPlantSS =
        StateSpace<TimeDomainType::CONTINUOUS, Space, InputSpace, OutputSpace>;
    using SolverInputs = InputLayout<InputChannel<InputKind::CONTROL, 0, 1>>;
    using SolverOutputs =
        OutputLayout<OutputChannel<OutputKind::MEASUREMENT, 0, 1>>;
    using SolverPlant = PlantModel<SolverPlantSS, SolverInputs, SolverOutputs>;
    constexpr auto BCol = ColVectord<2>{Vectord<1>{1.0}, Vectord<1>{0.5}};
    constexpr auto CRow = RowVectord<2>{Vectord<2>{1.0, 0.0}};
    static constexpr size_t SolverPDim = 3;
    static constexpr size_t SolverYDim = 2;
    static constexpr size_t SolverZDim = SolverPDim + SolverYDim;
    using SolverZSpace = RowVectord<SolverZDim>;
    constexpr auto solver_dec = Variable<VarMeta<SolverZSpace>>();

    constexpr auto A0 = MatBuilder(-1.0, 0.0, 0.0, -2.0);
    constexpr auto A1 = MatBuilder(-0.5, 0.0, 0.0, -1.0);
    constexpr auto B0 = BCol;
    constexpr auto B1 = BCol;
    constexpr auto C = CRow;

    constexpr auto plant0 = SolverPlant(SolverPlantSS(A0, B0, C));
    constexpr auto plant1 = SolverPlant(SolverPlantSS(A1, B1, C));
    constexpr auto poly = Vector<decltype(plant0), 2>{plant0, plant1};

    constexpr auto psf = PolytopicStateFeedback<decltype(plant0), 2>();
    constexpr auto p_w = Vectord<SolverPDim>{1.0, 0.0, 1.0};
    constexpr auto y_w = Vectord<SolverYDim>{0.0, 0.0};
    constexpr auto obj =
        make_polytopic_state_feedback_linear_objective<decltype(plant0), 2>(
            p_w, y_w, solver_dec);
    const auto prob =
        make_polytopic_state_feedback_problem<decltype(plant0), 2>(
            poly, solver_dec, obj);

    REQUIRE(is_problem_set_v<decltype(prob)>);
    REQUIRE(prob.validate());
    REQUIRE(std::tuple_size<decltype(prob.constraint_cones())>::value == 3);

    using Solver = PrimalDualIPM<decltype(prob)>;
    const auto cfg = pldi_solver_config<double>();
    const Solver solver(prob, cfg);
    const auto& res = solver.res;
#ifdef AKREON_REPORT_OUTPUT
    const auto& trace = solver.trace;
    if (res.tag != SolverResult::OPTIMAL || res.stalled) log(res, trace);
#endif

    REQUIRE(res.tag == SolverResult::OPTIMAL);
    REQUIRE(!res.stalled);

    Vectord<SolverPDim> q_vec{};
    for (size_t i = 0; i < SolverPDim; ++i) q_vec(i) = res.iterate.z(i);
    const auto P_res = make_smat<2>(q_vec);

    RowVectord<2> Y_res{};
    for (size_t j = 0; j < 2; ++j) Y_res(0)(j) = res.iterate.z(SolverPDim + j);

    REQUIRE(all_equal(P_res, transpose(P_res), rel_tol));
    REQUIRE(P_res(0)(0) > 1e-6);
    REQUIRE(P_res(1)(1) > 1e-6);

    const auto P_sym = SymmetricMatrix(P_res, 1e-6);
    REQUIRE(P_sym.is_valid());
    REQUIRE(P_sym.is_psd());

    const auto [K_res, K_event] = psf.K(P_res, Y_res);
    REQUIRE(K_event == linalg::NumericalEvent::NONE);
    REQUIRE(all_equal(Y_res, K_res * P_res, 1e-4));

    const auto cert0 = (-1.0) * (A0 * P_res + P_res * transpose(A0) +
                                 B0 * Y_res + transpose(Y_res) * transpose(B0));
    const auto cert1 = (-1.0) * (A1 * P_res + P_res * transpose(A1) +
                                 B1 * Y_res + transpose(Y_res) * transpose(B1));

    const auto cert0_sym = SymmetricMatrix(cert0, 1e-6);
    const auto cert1_sym = SymmetricMatrix(cert1, 1e-6);
    REQUIRE(cert0_sym.is_valid());
    REQUIRE(cert1_sym.is_valid());
    REQUIRE(cert0_sym.is_psd());
    REQUIRE(cert1_sym.is_psd());

    const auto cl0 = A0 + B0 * K_res;
    const auto cl1 = A1 + B1 * K_res;
    const auto lyap0 = (-1.0) * (cl0 * P_res + P_res * transpose(cl0));
    const auto lyap1 = (-1.0) * (cl1 * P_res + P_res * transpose(cl1));
    const auto lyap0_sym = SymmetricMatrix(lyap0, 1e-6);
    const auto lyap1_sym = SymmetricMatrix(lyap1, 1e-6);
    REQUIRE(lyap0_sym.is_valid());
    REQUIRE(lyap1_sym.is_valid());
    REQUIRE(lyap0_sym.is_psd());
    REQUIRE(lyap1_sym.is_psd());
  }

  SECTION("Quadratic objective helper solves lifted PLDI problem") {
    using InputSpace = Matrixd<2, 1>;
    using OutputSpace = RowVectord<2>;
    using SolverPlantSS =
        StateSpace<TimeDomainType::CONTINUOUS, Space, InputSpace, OutputSpace>;
    using SolverInputs = InputLayout<InputChannel<InputKind::CONTROL, 0, 1>>;
    using SolverOutputs =
        OutputLayout<OutputChannel<OutputKind::MEASUREMENT, 0, 1>>;
    using SolverPlant = PlantModel<SolverPlantSS, SolverInputs, SolverOutputs>;
    constexpr auto BCol = ColVectord<2>{Vectord<1>{1.0}, Vectord<1>{0.5}};
    constexpr auto CRow = RowVectord<2>{Vectord<2>{1.0, 0.0}};
    static constexpr size_t SolverPDim = 3;
    static constexpr size_t SolverYDim = 2;
    static constexpr size_t SolverZDim = SolverPDim + SolverYDim;
    using SolverZSpace = RowVectord<SolverZDim>;
    constexpr auto solver_dec = Variable<VarMeta<SolverZSpace>>();

    constexpr auto A0 = MatBuilder(-1.0, 0.0, 0.0, -2.0);
    constexpr auto A1 = MatBuilder(-0.5, 0.0, 0.0, -1.0);
    constexpr auto B0 = BCol;
    constexpr auto B1 = BCol;
    constexpr auto C = CRow;

    constexpr auto plant0 = SolverPlant(SolverPlantSS(A0, B0, C));
    constexpr auto plant1 = SolverPlant(SolverPlantSS(A1, B1, C));
    constexpr auto poly = Vector<decltype(plant0), 2>{plant0, plant1};

    constexpr auto psf = PolytopicStateFeedback<decltype(plant0), 2>();
    constexpr auto P = Matrixd<SolverZDim, SolverZDim>{
        Vectord<SolverZDim>{1.0, 0.0, 0.0, 0.0, 0.0},
        Vectord<SolverZDim>{0.0, 1.0, 0.0, 0.0, 0.0},
        Vectord<SolverZDim>{0.0, 0.0, 1.0, 0.0, 0.0},
        Vectord<SolverZDim>{0.0, 0.0, 0.0, 1.0, 0.0},
        Vectord<SolverZDim>{0.0, 0.0, 0.0, 0.0, 1.0},
    };
    constexpr auto q = Vectord<SolverZDim>{};
    constexpr auto obj =
        make_polytopic_state_feedback_quadratic_objective<decltype(plant0), 2>(
            P, q, solver_dec);
    const auto prob =
        make_polytopic_state_feedback_problem<decltype(plant0), 2>(
            poly, solver_dec, obj);

    REQUIRE(prob.validate());
    REQUIRE(std::tuple_size<decltype(prob.constraint_cones())>::value > 3);

    const auto cfg = pldi_solver_config<double>();
    const PrimalDualIPM solver(prob, cfg);
    const auto& res = solver.res;
#ifdef AKREON_REPORT_OUTPUT
    const auto& trace = solver.trace;
    if (res.tag != SolverResult::OPTIMAL || res.stalled) log(res, trace);
#endif

    REQUIRE(res.tag == SolverResult::OPTIMAL);
    REQUIRE(!res.stalled);

    Vectord<SolverPDim> q_vec{};
    for (size_t i = 0; i < SolverPDim; ++i) q_vec(i) = res.iterate.z(i);
    const auto P_res = make_smat<2>(q_vec);

    RowVectord<2> Y_res{};
    for (size_t j = 0; j < 2; ++j) Y_res(0)(j) = res.iterate.z(SolverPDim + j);

    const auto [K_res, K_event] = psf.K(P_res, Y_res);
    REQUIRE(K_event == linalg::NumericalEvent::NONE);
    REQUIRE(all_equal(Y_res, K_res * P_res, 1e-4));

    const auto cert0 = (-1.0) * (A0 * P_res + P_res * transpose(A0) +
                                 B0 * Y_res + transpose(Y_res) * transpose(B0));
    const auto cert1 = (-1.0) * (A1 * P_res + P_res * transpose(A1) +
                                 B1 * Y_res + transpose(Y_res) * transpose(B1));
    const auto cert0_sym = SymmetricMatrix(cert0, 1e-6);
    const auto cert1_sym = SymmetricMatrix(cert1, 1e-6);
    REQUIRE(cert0_sym.is_valid());
    REQUIRE(cert1_sym.is_valid());
    REQUIRE(cert0_sym.is_psd());
    REQUIRE(cert1_sym.is_psd());
  }
}
