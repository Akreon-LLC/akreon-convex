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
  constexpr auto I = make_identity<double, 2>();
  constexpr auto Z = Matrixd<2, 2>{};
  constexpr auto MatBuilder = mat_traits<Space>();

  static constexpr size_t QDim = 3;
  static constexpr size_t YDim = 4;
  static constexpr size_t ZDim = QDim + YDim;
  using ZSpace = RowVectord<ZDim>;
  constexpr auto dec = Variable<VarMeta<ZSpace>>();

  SECTION("Hand-checkable merged canonical LMI") {
    constexpr auto A0 = Z;
    constexpr auto A1 = MatBuilder(-1.0, 0.0, 0.0, -2.0);
    constexpr auto B0 = I;
    constexpr auto B1 = I;
    constexpr auto C = I;

    constexpr auto ss0 =
        StateSpace<Space, Space, Space>(A0, B0, C);
    constexpr auto ss1 =
        StateSpace<Space, Space, Space>(A1, B1, C);
    constexpr auto poly = Vector<decltype(ss0), 2>{ss0, ss1};

    constexpr auto psf = PolytopicStateFeedback<decltype(ss0), 2>();
    const auto lmis = psf(poly, dec);

    constexpr auto builder = conic::create_empty_cblk_builder<ZSpace>();
    const auto canon = lmis.template canonicalize<AtomType::SET>(builder);
    const auto cone_block = canon.cb;

    constexpr auto s2 = constexpr_sqrt_safe(2.0);
    constexpr auto F_exp = Matrixd<6, 7>{
        Vectord<7>{0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0},
        Vectord<7>{0.0, 0.0, 0.0, 0.0, -s2, -s2, 0.0},
        Vectord<7>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0},
        Vectord<7>{2.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0},
        Vectord<7>{0.0, 3.0, 0.0, 0.0, -s2, -s2, 0.0},
        Vectord<7>{0.0, 0.0, 4.0, 0.0, 0.0, 0.0, -2.0},
    };

    REQUIRE(all_equal(cone_block.map.linear_part(), F_exp));
    REQUIRE(cone_block.map.translation() == Vectord<6>{});
    REQUIRE(std::tuple_size<decltype(cone_block.cones)>::value == 2);
    REQUIRE(cone_block.validate());
  }

  SECTION("Solver finds feasible merged z with PSD Q block") {
    using InputSpace = Matrixd<2, 1>;
    using OutputSpace = RowVectord<2>;
    constexpr auto BCol = ColVectord<2>{Vectord<1>{1.0}, Vectord<1>{0.5}};
    constexpr auto CRow = RowVectord<2>{Vectord<2>{1.0, 0.0}};
    static constexpr size_t SolverQDim = 3;
    static constexpr size_t SolverYDim = 2;
    static constexpr size_t SolverZDim = SolverQDim + SolverYDim;
    using SolverZSpace = RowVectord<SolverZDim>;
    constexpr auto solver_dec = Variable<VarMeta<SolverZSpace>>();

    constexpr auto A0 = MatBuilder(-1.0, 0.0, 0.0, -2.0);
    constexpr auto A1 = MatBuilder(-0.5, 0.0, 0.0, -1.0);
    constexpr auto B0 = BCol;
    constexpr auto B1 = BCol;
    constexpr auto C = CRow;

    constexpr auto ss0 =
        StateSpace<Space, InputSpace, OutputSpace>(
            A0, B0, C);
    constexpr auto ss1 =
        StateSpace<Space, InputSpace, OutputSpace>(
            A1, B1, C);
    constexpr auto poly = Vector<decltype(ss0), 2>{ss0, ss1};

    constexpr auto psf = PolytopicStateFeedback<decltype(ss0), 2>();
    constexpr auto q_w = Vectord<SolverQDim>{1.0, 0.0, 1.0};
    constexpr auto y_w = Vectord<SolverYDim>{0.0, 0.0};
    constexpr auto obj =
        make_polytopic_state_feedback_linear_objective<decltype(ss0), 2>(
            q_w, y_w, solver_dec);
    const auto prob =
        make_polytopic_state_feedback_problem<decltype(ss0), 2>(poly,
                                                                solver_dec, obj);

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

    RowVectord<2> Y_res{};
    for (size_t j = 0; j < 2; ++j) Y_res(0)(j) = res.iterate.z(SolverQDim + j);

    REQUIRE(all_equal(Q_res, transpose(Q_res), rel_tol));
    REQUIRE(Q_res(0)(0) > 1e-6);
    REQUIRE(Q_res(1)(1) > 1e-6);

    const auto Q_sym = SymmetricMatrix(Q_res, 1e-6);
    REQUIRE(Q_sym.is_valid());
    REQUIRE(Q_sym.is_psd());

    const auto [K_res, K_event] = psf.K(Q_res, Y_res);
    REQUIRE(K_event == linalg::NumericalEvent::NONE);
    REQUIRE(all_equal(Y_res, K_res * Q_res, 1e-4));

    const auto cert0 = (-1.0) * (A0 * Q_res + Q_res * transpose(A0) +
                                 B0 * Y_res + transpose(Y_res) * transpose(B0));
    const auto cert1 = (-1.0) * (A1 * Q_res + Q_res * transpose(A1) +
                                 B1 * Y_res + transpose(Y_res) * transpose(B1));

    const auto cert0_sym = SymmetricMatrix(cert0, 1e-6);
    const auto cert1_sym = SymmetricMatrix(cert1, 1e-6);
    REQUIRE(cert0_sym.is_valid());
    REQUIRE(cert1_sym.is_valid());
    REQUIRE(cert0_sym.is_psd());
    REQUIRE(cert1_sym.is_psd());

    const auto cl0 = A0 + B0 * K_res;
    const auto cl1 = A1 + B1 * K_res;
    const auto lyap0 = (-1.0) * (cl0 * Q_res + Q_res * transpose(cl0));
    const auto lyap1 = (-1.0) * (cl1 * Q_res + Q_res * transpose(cl1));
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
    constexpr auto BCol = ColVectord<2>{Vectord<1>{1.0}, Vectord<1>{0.5}};
    constexpr auto CRow = RowVectord<2>{Vectord<2>{1.0, 0.0}};
    static constexpr size_t SolverQDim = 3;
    static constexpr size_t SolverYDim = 2;
    static constexpr size_t SolverZDim = SolverQDim + SolverYDim;
    using SolverZSpace = RowVectord<SolverZDim>;
    constexpr auto solver_dec = Variable<VarMeta<SolverZSpace>>();

    constexpr auto A0 = MatBuilder(-1.0, 0.0, 0.0, -2.0);
    constexpr auto A1 = MatBuilder(-0.5, 0.0, 0.0, -1.0);
    constexpr auto B0 = BCol;
    constexpr auto B1 = BCol;
    constexpr auto C = CRow;

    constexpr auto ss0 =
        StateSpace<Space, InputSpace, OutputSpace>(
            A0, B0, C);
    constexpr auto ss1 =
        StateSpace<Space, InputSpace, OutputSpace>(
            A1, B1, C);
    constexpr auto poly = Vector<decltype(ss0), 2>{ss0, ss1};

    constexpr auto psf = PolytopicStateFeedback<decltype(ss0), 2>();
    constexpr auto P = Matrixd<SolverZDim, SolverZDim>{
        Vectord<SolverZDim>{1.0, 0.0, 0.0, 0.0, 0.0},
        Vectord<SolverZDim>{0.0, 1.0, 0.0, 0.0, 0.0},
        Vectord<SolverZDim>{0.0, 0.0, 1.0, 0.0, 0.0},
        Vectord<SolverZDim>{0.0, 0.0, 0.0, 1.0, 0.0},
        Vectord<SolverZDim>{0.0, 0.0, 0.0, 0.0, 1.0},
    };
    constexpr auto q = Vectord<SolverZDim>{};
    constexpr auto obj =
        make_polytopic_state_feedback_quadratic_objective<decltype(ss0), 2>(
            P, q, solver_dec);
    const auto prob =
        make_polytopic_state_feedback_problem<decltype(ss0), 2>(poly,
                                                                solver_dec, obj);

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

    RowVectord<2> Y_res{};
    for (size_t j = 0; j < 2; ++j) Y_res(0)(j) = res.iterate.z(SolverQDim + j);

    const auto [K_res, K_event] = psf.K(Q_res, Y_res);
    REQUIRE(K_event == linalg::NumericalEvent::NONE);
    REQUIRE(all_equal(Y_res, K_res * Q_res, 1e-4));

    const auto cert0 = (-1.0) * (A0 * Q_res + Q_res * transpose(A0) +
                                 B0 * Y_res + transpose(Y_res) * transpose(B0));
    const auto cert1 = (-1.0) * (A1 * Q_res + Q_res * transpose(A1) +
                                 B1 * Y_res + transpose(Y_res) * transpose(B1));
    const auto cert0_sym = SymmetricMatrix(cert0, 1e-6);
    const auto cert1_sym = SymmetricMatrix(cert1, 1e-6);
    REQUIRE(cert0_sym.is_valid());
    REQUIRE(cert1_sym.is_valid());
    REQUIRE(cert0_sym.is_psd());
    REQUIRE(cert1_sym.is_psd());
  }
}
