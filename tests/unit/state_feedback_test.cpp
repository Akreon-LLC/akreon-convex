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

  constexpr auto MatBuilder = mat_traits<Space>();
  constexpr auto BCol = ColVectord<2>{Vectord<1>{1.0}, Vectord<1>{0.5}};
  constexpr auto CRow = RowVectord<2>{Vectord<2>{1.0, 0.0}};

  static constexpr size_t QDim = 3;
  static constexpr size_t YDim = 2;
  static constexpr size_t ZDim = QDim + YDim;
  using ZSpace = RowVectord<ZDim>;
  constexpr auto Z = Variable<VarMeta<ZSpace>>();

  static constexpr size_t FullInputZDim = QDim + 4;
  using FullInputZSpace = RowVectord<FullInputZDim>;
  constexpr auto FullInputZ = Variable<VarMeta<FullInputZSpace>>();

  SECTION("Hand-checkable merged canonical LMI") {
    constexpr auto A = Matrixd<2, 2>{};
    constexpr auto B = make_identity<double, 2>();
    constexpr auto C = make_identity<double, 2>();
    constexpr auto ss =
        StateSpace<TimeDomainType::CONTINUOUS, Space, Space, Space>(A, B, C);

    constexpr auto sf = StateFeedback<decltype(ss)>();
    const auto lmi = sf(ss, FullInputZ);

    constexpr auto builder =
        conic::create_empty_cblk_builder<FullInputZSpace>();
    const auto cone_block =
        lmi.template canonicalize<AtomType::SET>(builder).cb;

    constexpr auto s2 = constexpr_sqrt_safe(2.0);
    constexpr auto F_exp =
        Matrixd<3, 7>{Vectord<7>{0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0},
                      Vectord<7>{0.0, 0.0, 0.0, 0.0, -s2, -s2, 0.0},
                      Vectord<7>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0}};

    REQUIRE(all_equal(cone_block.map.linear_part(), F_exp));
    REQUIRE(cone_block.map.translation() == Vectord<3>{});
    REQUIRE(std::tuple_size<decltype(cone_block.cones)>::value == 1);
    REQUIRE(cone_block.validate());
  }

  SECTION("Solver finds feasible Q/Y and stabilizing K") {
    constexpr auto A = MatBuilder(-1.0, 0.0, 0.0, -2.0);
    constexpr auto ss =
        StateSpace<TimeDomainType::CONTINUOUS, Space, InputSpace, OutputSpace>(
            A, BCol, CRow);

    constexpr auto sf = StateFeedback<decltype(ss)>();
    constexpr auto P = Matrix<double, ZDim, ZDim>{
        Vectord<ZDim>{1.0, 0.0, 0.0, 0.0, 0.0},
        Vectord<ZDim>{0.0, 1.0, 0.0, 0.0, 0.0},
        Vectord<ZDim>{0.0, 0.0, 1.0, 0.0, 0.0},
        Vectord<ZDim>{0.0, 0.0, 0.0, 1.0, 0.0},
        Vectord<ZDim>{0.0, 0.0, 0.0, 0.0, 1.0},
    };
    constexpr auto q = Vectord<ZDim>{};
    constexpr auto obj = make_quad_form(P, q, Z);
    const auto prob = make_state_feedback_problem(ss, Z, obj);

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

    const auto z_split = linalg::splice<QDim>(res.iterate.z);
    const auto Q_res = make_smat<2>(std::get<0>(z_split));
    RowVectord<2> Y_res{};
    for (size_t j = 0; j < YDim; ++j) Y_res(0)(j) = std::get<1>(z_split)(j);

    REQUIRE(all_equal(Q_res, transpose(Q_res), rel_tol));
    REQUIRE(Q_res(0)(0) > 1e-6);
    REQUIRE(Q_res(1)(1) > 1e-6);

    const auto Q_sym = SymmetricMatrix(Q_res, 1e-6);
    REQUIRE(Q_sym.is_valid());
    REQUIRE(Q_sym.is_psd());

    const auto [K_res, K_event] = sf.K(Q_res, Y_res);
    REQUIRE(K_event == linalg::NumericalEvent::NONE);
    REQUIRE(all_equal(Y_res, K_res * Q_res, 1e-4));

    const auto cert =
        (-1.0) * (A * Q_res + Q_res * transpose(A) + BCol * Y_res +
                  transpose(Y_res) * transpose(BCol));
    const auto cert_sym = SymmetricMatrix(cert, 1e-6);
    REQUIRE(cert_sym.is_valid());
    REQUIRE(cert_sym.is_psd());

    const auto cl = A + BCol * K_res;
    const auto lyap = (-1.0) * (cl * Q_res + Q_res * transpose(cl));
    const auto lyap_sym = SymmetricMatrix(lyap, 1e-6);
    REQUIRE(lyap_sym.is_valid());
    REQUIRE(lyap_sym.is_psd());
  }
}
