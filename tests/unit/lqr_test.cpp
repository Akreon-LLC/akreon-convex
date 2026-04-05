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

TEST_CASE("Small LQR 2x2", "[lqr_test]") {
  using Space = Matrixd<2, 2>;
  using PSpace = SymmetricMatrix<Space>;
  constexpr auto P = Variable<PsdVarMeta<Space>>();
  constexpr auto MatBuilder = mat_traits<Space>();

  SECTION("LQR case 1") {
    SECTION("Hand-checkable canonical LMI") {
      constexpr auto I = make_identity<double, 2>();
      constexpr auto Z = Matrixd<2, 2>{};
      constexpr auto Q = MatBuilder(1.0, 0.0, 0.0, 1.0);
      constexpr auto R = MatBuilder(1.0, 0.0, 0.0, 1.0);
      constexpr auto A = Z;
      constexpr auto B = I;
      constexpr auto C = I;

      constexpr auto ss =
          StateSpace<TimeDomainType::CONTINUOUS, Space, Space, Space>(A, B, C);

      constexpr auto lqr = LQR<decltype(ss), decltype(Q), decltype(R)>();
      constexpr auto lmi = lqr(ss, Q, R, P);

      STATIC_REQUIRE(lmi.is_well_formed());

      constexpr auto builder = conic::create_empty_cblk_builder<PSpace>();
      constexpr auto cone_block =
          lmi.template canonicalize<AtomType::SET>(builder).cb;

      constexpr auto sqrt2 = constexpr_sqrt_safe(2.0);
      constexpr auto F_exp = Matrixd<10, 3>{
          Vectord<3>{0.0, 0.0, 0.0},    // (0,0)
          Vectord<3>{0.0, 0.0, 0.0},    // (1,0)
          Vectord<3>{0.0, 0.0, 0.0},    // (1,1)
          Vectord<3>{sqrt2, 0.0, 0.0},  // (2,0)
          Vectord<3>{0.0, 1.0, 0.0},    // (2,1)
          Vectord<3>{0.0, 0.0, 0.0},    // (2,2)
          Vectord<3>{0.0, 1.0, 0.0},    // (3,0)
          Vectord<3>{0.0, 0.0, sqrt2},  // (3,1)
          Vectord<3>{0.0, 0.0, 0.0},    // (3,2)
          Vectord<3>{0.0, 0.0, 0.0},    // (3,3)
      };

      STATIC_REQUIRE(all_equal(cone_block.map.linear_part(), F_exp));
      STATIC_REQUIRE(
          cone_block.map.translation() ==
          Vectord<10>{1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0});

      constexpr auto cones = cone_block.cones;
      constexpr auto sz = std::tuple_size<decltype(cones)>::value;
      STATIC_REQUIRE(sz == 1);
      STATIC_REQUIRE(cone_block.validate());
    }

    SECTION("Hand-checkable solver path") {
      constexpr auto I = make_identity<double, 2>();
      constexpr auto Z = Matrixd<2, 2>{};
      constexpr auto Q = MatBuilder(1.0, 0.0, 0.0, 1.0);
      constexpr auto R = MatBuilder(1.0, 0.0, 0.0, 1.0);
      constexpr auto A = Z;
      constexpr auto B = I;
      constexpr auto C = I;
      constexpr auto ss =
          StateSpace<TimeDomainType::CONTINUOUS, Space, Space, Space>(A, B, C);

      constexpr auto P_geq =
          Inequality<InequalityType::GEQ, decltype(P), Space>(P, 1e-6 * I);

      constexpr auto lqr = LQR<decltype(ss), decltype(Q), decltype(R)>();
      constexpr auto lmi = lqr(ss, Q, R, P);
      constexpr auto prob_constr = make_problem(P).add_constraint(P_geq).in_set(lmi);
      constexpr auto c = Vectord<3>{-1.0, 0.0, -1.0};
      constexpr auto obj = WeightedSum<PSpace>()(c, P);
      constexpr auto prob = prob_constr.minimize(obj);

      STATIC_REQUIRE(is_problem_set_v<decltype(prob)>);
      STATIC_REQUIRE(prob.validate());
      STATIC_REQUIRE(
          std::tuple_size<decltype(prob.constraint_cones())>::value == 2);
      STATIC_REQUIRE(prob.linear_objective() == Matrixd<1, 3>{c});

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
      REQUIRE(constexpr_abs(P_res(0)(0) - 1.0) < 1e-3);
      REQUIRE(constexpr_abs(P_res(1)(1) - 1.0) < 1e-3);
      REQUIRE(constexpr_abs(P_res(0)(1)) < 1e-3);
    }

    SECTION("Stable diagonal plant solver path") {
      constexpr auto I = make_identity<double, 2>();
      constexpr auto Q = MatBuilder(1.0, 0.0, 0.0, 1.0);
      constexpr auto R = MatBuilder(1.0, 0.0, 0.0, 1.0);
      constexpr auto A = MatBuilder(-1.0, 0.0, 0.0, -2.0);
      constexpr auto B = I;
      constexpr auto C = I;
      constexpr auto ss =
          StateSpace<TimeDomainType::CONTINUOUS, Space, Space, Space>(A, B, C);

      constexpr auto P_geq =
          Inequality<InequalityType::GEQ, decltype(P), Space>(P, 1e-6 * I);

      constexpr auto lqr = LQR<decltype(ss), decltype(Q), decltype(R)>();
      constexpr auto lmi = lqr(ss, Q, R, P);
      constexpr auto prob_constr =
          make_problem(P).add_constraint(P_geq).in_set(lmi);
      constexpr auto c = Vectord<3>{1.0, 0.0, 1.0};
      constexpr auto obj = WeightedSum<PSpace>()(c, P);
      constexpr auto prob = prob_constr.minimize(obj);

      STATIC_REQUIRE(is_problem_set_v<decltype(prob)>);
      STATIC_REQUIRE(prob.validate());
      STATIC_REQUIRE(
          std::tuple_size<decltype(prob.constraint_cones())>::value == 2);
      STATIC_REQUIRE(prob.linear_objective() == Matrixd<1, 3>{c});

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
      REQUIRE(constexpr_abs(P_res(0)(1)) < 1e-2);

      const auto lmi_eval = Matrixd<4, 4>{
          Vectord<4>{Q(0)(0) + 2.0 * A(0)(0) * P_res(0)(0), 0.0,
                     P_res(0)(0), P_res(0)(1)},
          Vectord<4>{0.0, Q(1)(1) + 2.0 * A(1)(1) * P_res(1)(1),
                     P_res(0)(1), P_res(1)(1)},
          Vectord<4>{P_res(0)(0), P_res(0)(1), R(0)(0), 0.0},
          Vectord<4>{P_res(0)(1), P_res(1)(1), 0.0, R(1)(1)},
      };
      const auto chol = Cholesky(lmi_eval);
      REQUIRE(chol.is_valid());
      REQUIRE(chol.is_pd());
    }
  }
}
