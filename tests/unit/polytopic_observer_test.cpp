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

TEST_CASE("Polytopic Observer 2x2", "[polytopic_observer_test]") {
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

  static constexpr size_t QDim = 3;
  static constexpr size_t WDim = 4;
  static constexpr size_t ZDim = QDim + WDim;
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

    constexpr auto po = PolytopicObserver<decltype(plant0), 2>();
    const auto lmis = po(poly, dec);

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

  SECTION("Solver finds feasible merged z with PSD Q block") {
    using InputSpace = ColVectord<2>;
    using OutputSpace = RowVectord<2>;
    using SolverPlantSS =
        StateSpace<TimeDomainType::CONTINUOUS, Space, InputSpace, OutputSpace>;
    using SolverInputs = InputLayout<InputChannel<InputKind::CONTROL, 0, 1>>;
    using SolverOutputs =
        OutputLayout<OutputChannel<OutputKind::MEASUREMENT, 0, 1>>;
    using SolverPlant = PlantModel<SolverPlantSS, SolverInputs, SolverOutputs>;
    constexpr auto BCol = ColVectord<2>{Vectord<1>{1.0}, Vectord<1>{0.5}};
    constexpr auto CRow = RowVectord<2>{Vectord<2>{1.0, 0.0}};
    static constexpr size_t SolverQDim = 3;
    static constexpr size_t SolverWDim = 2;
    static constexpr size_t SolverZDim = SolverQDim + SolverWDim;
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

    constexpr auto po = PolytopicObserver<decltype(plant0), 2>();
    const auto prob =
        make_polytopic_observer_problem<decltype(plant0), 2>(poly, solver_dec);

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

    Vectord<SolverQDim> q_vec{};
    for (size_t i = 0; i < SolverQDim; ++i) q_vec(i) = res.iterate.z(i);
    const auto Q_res = make_smat<2>(q_vec);

    ColVectord<2> W_res{};
    for (size_t j = 0; j < SolverWDim; ++j)
      W_res(j)(0) = res.iterate.z(SolverQDim + j);

    REQUIRE(all_equal(Q_res, transpose(Q_res), rel_tol));
    REQUIRE(Q_res(0)(0) > 1e-6);
    REQUIRE(Q_res(1)(1) > 1e-6);

    const auto Q_sym = SymmetricMatrix(Q_res, 1e-6);
    REQUIRE(Q_sym.is_valid());
    REQUIRE(Q_sym.is_psd());

    const auto [L_res, L_event] = po.L(Q_res, W_res);
    REQUIRE(L_event == linalg::NumericalEvent::NONE);
    REQUIRE(all_equal(W_res, Q_res * L_res, 1e-4));

    const auto cert0 = (-1.0) * (transpose(A0) * Q_res + Q_res * A0 +
                                 W_res * C + transpose(C) * transpose(W_res));
    const auto cert1 = (-1.0) * (transpose(A1) * Q_res + Q_res * A1 +
                                 W_res * C + transpose(C) * transpose(W_res));

    const auto cert0_sym = SymmetricMatrix(cert0, 1e-6);
    const auto cert1_sym = SymmetricMatrix(cert1, 1e-6);
    REQUIRE(cert0_sym.is_valid());
    REQUIRE(cert1_sym.is_valid());
    REQUIRE(cert0_sym.is_psd());
    REQUIRE(cert1_sym.is_psd());

    const auto err0 = A0 - L_res * C;
    const auto err1 = A1 - L_res * C;
    const auto lyap0 = (-1.0) * (transpose(err0) * Q_res + Q_res * err0);
    const auto lyap1 = (-1.0) * (transpose(err1) * Q_res + Q_res * err1);
    const auto lyap0_sym = SymmetricMatrix(lyap0, 1e-6);
    const auto lyap1_sym = SymmetricMatrix(lyap1, 1e-6);
    REQUIRE(lyap0_sym.is_valid());
    REQUIRE(lyap1_sym.is_valid());
    REQUIRE(lyap0_sym.is_psd());
    REQUIRE(lyap1_sym.is_psd());
  }

  SECTION("Quadratic objective helper solves lifted observer problem") {
    using InputSpace = ColVectord<2>;
    using OutputSpace = RowVectord<2>;
    using SolverPlantSS =
        StateSpace<TimeDomainType::CONTINUOUS, Space, InputSpace, OutputSpace>;
    using SolverInputs = InputLayout<InputChannel<InputKind::CONTROL, 0, 1>>;
    using SolverOutputs =
        OutputLayout<OutputChannel<OutputKind::MEASUREMENT, 0, 1>>;
    using SolverPlant = PlantModel<SolverPlantSS, SolverInputs, SolverOutputs>;
    constexpr auto BCol = ColVectord<2>{Vectord<1>{1.0}, Vectord<1>{0.5}};
    constexpr auto CRow = RowVectord<2>{Vectord<2>{1.0, 0.0}};
    static constexpr size_t SolverQDim = 3;
    static constexpr size_t SolverWDim = 2;
    static constexpr size_t SolverZDim = SolverQDim + SolverWDim;
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

    constexpr auto po = PolytopicObserver<decltype(plant0), 2>();
    constexpr auto Pmat = Matrixd<SolverZDim, SolverZDim>{
        Vectord<SolverZDim>{1.0, 0.0, 0.0, 0.0, 0.0},
        Vectord<SolverZDim>{0.0, 1.0, 0.0, 0.0, 0.0},
        Vectord<SolverZDim>{0.0, 0.0, 1.0, 0.0, 0.0},
        Vectord<SolverZDim>{0.0, 0.0, 0.0, 1.0, 0.0},
        Vectord<SolverZDim>{0.0, 0.0, 0.0, 0.0, 1.0},
    };
    constexpr auto q = Vectord<SolverZDim>{};
    constexpr auto obj =
        make_polytopic_observer_quadratic_objective<decltype(plant0), 2>(
            Pmat, q, solver_dec);
    const auto prob = make_polytopic_observer_problem<decltype(plant0), 2>(
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

    Vectord<SolverQDim> q_vec{};
    for (size_t i = 0; i < SolverQDim; ++i) q_vec(i) = res.iterate.z(i);
    const auto Q_res = make_smat<2>(q_vec);

    ColVectord<2> W_res{};
    for (size_t j = 0; j < SolverWDim; ++j)
      W_res(j)(0) = res.iterate.z(SolverQDim + j);

    const auto [L_res, L_event] = po.L(Q_res, W_res);
    REQUIRE(L_event == linalg::NumericalEvent::NONE);
    REQUIRE(all_equal(W_res, Q_res * L_res, 1e-4));

    const auto cert0 = (-1.0) * (transpose(A0) * Q_res + Q_res * A0 +
                                 W_res * C + transpose(C) * transpose(W_res));
    const auto cert1 = (-1.0) * (transpose(A1) * Q_res + Q_res * A1 +
                                 W_res * C + transpose(C) * transpose(W_res));
    const auto cert0_sym = SymmetricMatrix(cert0, 1e-6);
    const auto cert1_sym = SymmetricMatrix(cert1, 1e-6);
    REQUIRE(cert0_sym.is_valid());
    REQUIRE(cert1_sym.is_valid());
    REQUIRE(cert0_sym.is_psd());
    REQUIRE(cert1_sym.is_psd());

    const auto err0 = A0 - L_res * C;
    const auto err1 = A1 - L_res * C;
    const auto lyap0 = (-1.0) * (transpose(err0) * Q_res + Q_res * err0);
    const auto lyap1 = (-1.0) * (transpose(err1) * Q_res + Q_res * err1);
    const auto lyap0_sym = SymmetricMatrix(lyap0, 1e-6);
    const auto lyap1_sym = SymmetricMatrix(lyap1, 1e-6);
    REQUIRE(lyap0_sym.is_valid());
    REQUIRE(lyap1_sym.is_valid());
    REQUIRE(lyap0_sym.is_psd());
    REQUIRE(lyap1_sym.is_psd());
  }
}
