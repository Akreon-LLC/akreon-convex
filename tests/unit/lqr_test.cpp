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
static constexpr auto lqr_solver_config() {
  auto cfg = config::PrimalDualIPMConfig<T>{};
  cfg.steps = 64;
  cfg.duality_gap_tol = T(2e-4);
  return cfg;
}

TEST_CASE("Small LQR 2x2", "[lqr_test]") {
  using Space = Matrixd<2, 2>;
  using PSpace = SymmetricMatrix<Space>;
  constexpr auto P = Variable<PsdVarMeta<Space>>();
  constexpr auto MatBuilder = mat_traits<Space>();

  SECTION("Canonical LMI matches Lyapunov form") {
    constexpr auto I = make_identity<double, 2>();
    constexpr auto Z = Matrixd<2, 2>{};
    constexpr auto Q = MatBuilder(1.0, 0.0, 0.0, 1.0);
    constexpr auto R = MatBuilder(1.0, 0.0, 0.0, 1.0);
    constexpr auto A = Z;
    constexpr auto B = I;
    constexpr auto C = I;
    constexpr auto ss =
        StateSpace<TimeDomainType::CONTINUOUS, Space, Space, Space>(A, B, C);

    constexpr auto lqr = LyapunovCertificate<decltype(ss), decltype(Q), decltype(R)>();
    constexpr auto lmi = lqr(ss, Q, R, P);

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

  SECTION("Boundary witness for stable full-actuation plant") {
    constexpr auto I = make_identity<double, 2>();
    constexpr auto Q = MatBuilder(1.0, 0.0, 0.0, 1.0);
    constexpr auto R = MatBuilder(1.0, 0.0, 0.0, 1.0);
    constexpr auto A = -1.0 * I;
    constexpr auto B = I;
    constexpr auto C = I;
    constexpr auto ss =
        StateSpace<TimeDomainType::CONTINUOUS, Space, Space, Space>(A, B, C);
    constexpr auto lqr = LyapunovCertificate<decltype(ss), decltype(Q), decltype(R)>();

    const auto P_star = I;
    const auto [K_res, K_event] = lqr.K(P_star, ss, R);
    REQUIRE(K_event == linalg::NumericalEvent::NONE);
    REQUIRE(all_equal(K_res, I, 1e-6));

    Matrixd<4, 4> lmi_block{};
    lmi_block = set_block<0, 2, 0, 2>(
        lmi_block, (-1.0) * (transpose(A) * P_star + P_star * A + Q));
    lmi_block = set_block<0, 2, 2, 4>(lmi_block, (-1.0) * (P_star * B));
    lmi_block =
        set_block<2, 4, 0, 2>(lmi_block, (-1.0) * (transpose(B) * P_star));
    lmi_block = set_block<2, 4, 2, 4>(lmi_block, R);

    const auto lmi_sym = SymmetricMatrix(lmi_block, 1e-6);
    REQUIRE(lmi_sym.is_valid());
    REQUIRE(lmi_sym.is_psd());

    const auto care_res = (-1.0) * (transpose(A) * P_star + P_star * A + Q) -
                          transpose(P_star) * P_star;
    REQUIRE(all_equal(care_res, Matrixd<2, 2>{}, 1e-6));
  }

  SECTION("Solver converges on stable 2x2 full-actuation plant") {
    constexpr auto I = make_identity<double, 2>();
    constexpr auto Q = MatBuilder(1.0, 0.0, 0.0, 1.0);
    constexpr auto R = MatBuilder(1.0, 0.0, 0.0, 1.0);
    constexpr auto A = -1.0 * I;
    constexpr auto B = I;
    constexpr auto C = I;
    constexpr auto ss =
        StateSpace<TimeDomainType::CONTINUOUS, Space, Space, Space>(A, B, C);

    constexpr auto P_geq =
        Inequality<InequalityType::GEQ, decltype(P), Space>(P, 1e-6 * I);
    constexpr auto lqr = LyapunovCertificate<decltype(ss), decltype(Q), decltype(R)>();
    constexpr auto lmi = lqr(ss, Q, R, P);
    constexpr auto prob_constr =
        make_problem(P).add_constraint(P_geq).in_set(lmi);
    constexpr auto c = Vectord<3>{1.0, 0.0, 1.0};
    constexpr auto obj = WeightedSum<PSpace>()(c, P);
    constexpr auto prob = prob_constr.minimize(obj);

    STATIC_REQUIRE(is_problem_set_v<decltype(prob)>);
    STATIC_REQUIRE(prob.validate());

    using Solver = PrimalDualIPM<decltype(prob)>;
    const auto cfg = lqr_solver_config<double>();

    const Solver cold_solver(prob, cfg);
    const auto& cold_res = cold_solver.res;
#ifdef AKREON_DEBUG
    const auto& cold_trace = cold_solver.trace;
    if (cold_res.tag != SolverResult::OPTIMAL || cold_res.stalled)
      log(cold_res, cold_trace);
#endif

    REQUIRE(cold_res.tag == SolverResult::OPTIMAL);
    REQUIRE(!cold_res.stalled);

    PrimalDualIterate<typename Solver::Var, typename Solver::ConeVar,
                      typename Solver::NuVar>
        warm{};
    warm.z = get_svec(I + 1e-3 * I);
    warm.x = prob.constraint_map().linear_part() * warm.z +
             prob.constraint_map().translation();
    warm.y = warm.x;

    const Solver warm_solver(prob, warm, cfg);
    const auto& warm_res = warm_solver.res;
#ifdef AKREON_DEBUG
    const auto& warm_trace = warm_solver.trace;
    if (warm_res.tag != SolverResult::OPTIMAL || warm_res.stalled)
      log(warm_res, warm_trace);
#endif

    REQUIRE(warm_res.tag == SolverResult::OPTIMAL);
    REQUIRE(!warm_res.stalled);

    const auto P_cold = make_smat<2>(cold_res.iterate.z);
    const auto P_warm = make_smat<2>(warm_res.iterate.z);
    REQUIRE(all_equal(P_cold, transpose(P_cold), rel_tol));
    REQUIRE(all_equal(P_warm, transpose(P_warm), rel_tol));
    REQUIRE(P_cold(0)(0) > 1e-6);
    REQUIRE(P_cold(1)(1) > 1e-6);
    REQUIRE(P_warm(0)(0) > 1e-6);
    REQUIRE(P_warm(1)(1) > 1e-6);

    const auto [K_cold, K_cold_event] = lqr.K(P_cold, ss, R);
    const auto [K_warm, K_warm_event] = lqr.K(P_warm, ss, R);
    REQUIRE(K_cold_event == linalg::NumericalEvent::NONE);
    REQUIRE(K_warm_event == linalg::NumericalEvent::NONE);
    REQUIRE(all_equal(K_cold, P_cold, 1e-4));
    REQUIRE(all_equal(K_warm, P_warm, 1e-4));
  }

  SECTION(
      "Unstable controllable plant is infeasible under current Lyapunov-form "
      "P-only LQR LMI") {
    constexpr auto I = make_identity<double, 2>();
    constexpr auto Q = MatBuilder(1.0, 0.0, 0.0, 1.0);
    constexpr auto R = MatBuilder(1.0, 0.0, 0.0, 1.0);
    constexpr auto A = 0.2 * I;
    constexpr auto B = I;
    constexpr auto C = I;
    constexpr auto ss =
        StateSpace<TimeDomainType::CONTINUOUS, Space, Space, Space>(A, B, C);

    constexpr auto P_geq =
        Inequality<InequalityType::GEQ, decltype(P), Space>(P, 1e-6 * I);
    constexpr auto lqr = LyapunovCertificate<decltype(ss), decltype(Q), decltype(R)>();
    constexpr auto lmi = lqr(ss, Q, R, P);
    constexpr auto prob_constr =
        make_problem(P).add_constraint(P_geq).in_set(lmi);
    constexpr auto c = Vectord<3>{1.0, 0.0, 1.0};
    constexpr auto obj = WeightedSum<PSpace>()(c, P);
    constexpr auto prob = prob_constr.minimize(obj);

    STATIC_REQUIRE(is_problem_set_v<decltype(prob)>);
    STATIC_REQUIRE(prob.validate());

    using Solver = PrimalDualIPM<decltype(prob)>;
    const auto cfg = lqr_solver_config<double>();

    // With A = a I, a > 0, and B = R = Q = I, the Schur complement becomes
    // -(2 a P + I) - P^2, which is strictly negative for every P \succeq 0.
    const Solver cold_solver(prob, cfg);
    const auto& cold_res = cold_solver.res;
#ifdef AKREON_DEBUG
    const auto& cold_trace = cold_solver.trace;
    if (cold_res.tag == SolverResult::OPTIMAL && !cold_res.stalled)
      log(cold_res, cold_trace);
#endif

    REQUIRE(cold_res.tag != SolverResult::OPTIMAL);
  }
}

TEST_CASE("Small LQR 3x3", "[lqr_test]") {
  using Space = Matrixd<3, 3>;
  using PSpace = SymmetricMatrix<Space>;
  constexpr auto P = Variable<PsdVarMeta<Space>>();
  constexpr auto I = make_identity<double, 3>();
  constexpr auto MatBuilder = mat_traits<Space>();

  SECTION("Diagonal stable plant has solid boundary and interior witnesses") {
    constexpr auto Q = I;
    constexpr auto R = I;
    constexpr auto A =
        MatBuilder(-2.0, 0.0, 0.0, 0.0, -2.5, 0.0, 0.0, 0.0, -3.0);
    constexpr auto B = I;
    constexpr auto C = I;
    constexpr auto ss =
        StateSpace<TimeDomainType::CONTINUOUS, Space, Space, Space>(A, B, C);
    constexpr auto lqr = LyapunovCertificate<decltype(ss), decltype(Q), decltype(R)>();

    const auto p1 = 2.0 - constexpr_sqrt_safe(3.0);
    const auto p2 = 2.5 - constexpr_sqrt_safe(5.25);
    const auto p3 = 3.0 - constexpr_sqrt_safe(8.0);
    const auto P_star = MatBuilder(p1, 0.0, 0.0, 0.0, p2, 0.0, 0.0, 0.0, p3);
    const auto P_inner = P_star + 1e-3 * I;

    const auto [K_res, K_event] = lqr.K(P_star, ss, R);
    REQUIRE(K_event == linalg::NumericalEvent::NONE);
    REQUIRE(all_equal(K_res, P_star, 1e-6));

    const auto make_lmi_block = [&](const auto& P_eval) {
      Matrixd<6, 6> block{};
      block = set_block<0, 3, 0, 3>(
          block, (-1.0) * (transpose(A) * P_eval + P_eval * A + Q));
      block = set_block<0, 3, 3, 6>(block, (-1.0) * (P_eval * B));
      block = set_block<3, 6, 0, 3>(block, (-1.0) * (transpose(B) * P_eval));
      block = set_block<3, 6, 3, 6>(block, R);
      return block;
    };

    const auto boundary_block = make_lmi_block(P_star);
    const auto interior_block = make_lmi_block(P_inner);

    const auto boundary_sym = SymmetricMatrix(boundary_block, 1e-6);
    REQUIRE(boundary_sym.is_valid());
    REQUIRE(boundary_sym.is_psd());

    const auto interior_chol = Cholesky(interior_block);
    REQUIRE(interior_chol.is_valid());
    REQUIRE(interior_chol.is_pd());

    const auto boundary_res =
        (-1.0) * (transpose(A) * P_star + P_star * A + Q) -
        transpose(P_star) * P_star;
    const auto interior_res =
        (-1.0) * (transpose(A) * P_inner + P_inner * A + Q) -
        transpose(P_inner) * P_inner;

    REQUIRE(all_equal(boundary_res, Matrixd<3, 3>{}, 1e-5));
    REQUIRE(interior_res(0)(0) > 0.0);
    REQUIRE(interior_res(1)(1) > 0.0);
    REQUIRE(interior_res(2)(2) > 0.0);
  }

  SECTION("Coupled stable full-actuation plant admits an interior witness") {
    constexpr auto Q = I;
    constexpr auto R = I;
    constexpr auto A =
        MatBuilder(-2.0, 0.2, 0.0, 0.0, -1.5, 0.3, 0.0, 0.0, -2.5);
    constexpr auto B = I;
    constexpr auto C = I;
    constexpr auto ss =
        StateSpace<TimeDomainType::CONTINUOUS, Space, Space, Space>(A, B, C);
    constexpr auto lqr = LyapunovCertificate<decltype(ss), decltype(Q), decltype(R)>();

    const auto p1 = 1.2 * (2.0 - constexpr_sqrt_safe(3.0));
    const auto p2 = 1.2 * (1.5 - constexpr_sqrt_safe(1.25));
    const auto p3 = 1.2 * (2.5 - constexpr_sqrt_safe(5.25));
    const auto P_feas = MatBuilder(p1, 0.0, 0.0, 0.0, p2, 0.0, 0.0, 0.0, p3);

    const auto [K_res, K_event] = lqr.K(P_feas, ss, R);
    REQUIRE(K_event == linalg::NumericalEvent::NONE);
    REQUIRE(K_res(0)(0) > 0.0);
    REQUIRE(K_res(1)(1) > 0.0);
    REQUIRE(K_res(2)(2) > 0.0);

    Matrixd<6, 6> lmi_block{};
    lmi_block = set_block<0, 3, 0, 3>(
        lmi_block, (-1.0) * (transpose(A) * P_feas + P_feas * A + Q));
    lmi_block = set_block<0, 3, 3, 6>(lmi_block, (-1.0) * (P_feas * B));
    lmi_block =
        set_block<3, 6, 0, 3>(lmi_block, (-1.0) * (transpose(B) * P_feas));
    lmi_block = set_block<3, 6, 3, 6>(lmi_block, R);

    const auto chol = Cholesky(lmi_block);
    REQUIRE(chol.is_valid());
    REQUIRE(chol.is_pd());
  }

  SECTION("Solver converges on coupled stable full-actuation plant") {
    constexpr auto Q = I;
    constexpr auto R = I;
    constexpr auto A =
        MatBuilder(-2.0, 0.2, 0.0, 0.0, -1.5, 0.3, 0.0, 0.0, -2.5);
    constexpr auto B = I;
    constexpr auto C = I;
    constexpr auto ss =
        StateSpace<TimeDomainType::CONTINUOUS, Space, Space, Space>(A, B, C);

    constexpr auto P_geq =
        Inequality<InequalityType::GEQ, decltype(P), Space>(P, 1e-6 * I);
    constexpr auto lqr = LyapunovCertificate<decltype(ss), decltype(Q), decltype(R)>();
    constexpr auto lmi = lqr(ss, Q, R, P);
    constexpr auto prob_constr =
        make_problem(P).add_constraint(P_geq).in_set(lmi);
    constexpr auto c = Vectord<6>{1.0, 0.0, 1.0, 0.0, 0.0, 1.0};
    constexpr auto obj = WeightedSum<PSpace>()(c, P);
    constexpr auto prob = prob_constr.minimize(obj);

    STATIC_REQUIRE(is_problem_set_v<decltype(prob)>);
    STATIC_REQUIRE(prob.validate());

    using Solver = PrimalDualIPM<decltype(prob)>;
    const auto cfg = lqr_solver_config<double>();

    const Solver cold_solver(prob, cfg);
    const auto& cold_res = cold_solver.res;
#ifdef AKREON_DEBUG
    const auto& cold_trace = cold_solver.trace;
    if (cold_res.tag != SolverResult::OPTIMAL || cold_res.stalled)
      log(cold_res, cold_trace);
#endif

    REQUIRE(cold_res.tag == SolverResult::OPTIMAL);
    REQUIRE(!cold_res.stalled);

    const auto P_cold = make_smat<3>(cold_res.iterate.z);
    REQUIRE(all_equal(P_cold, transpose(P_cold), rel_tol));
    REQUIRE(P_cold(0)(0) > 1e-6);
    REQUIRE(P_cold(1)(1) > 1e-6);
    REQUIRE(P_cold(2)(2) > 1e-6);

    const auto [K_cold, K_cold_event] = lqr.K(P_cold, ss, R);
    REQUIRE(K_cold_event == linalg::NumericalEvent::NONE);
    REQUIRE(K_cold(0)(0) > 0.0);
    REQUIRE(K_cold(1)(1) > 0.0);
    REQUIRE(K_cold(2)(2) > 0.0);
  }

  SECTION("Solver converges on a feasible diagonal Lyapunov-form problem") {
    constexpr auto Q = I;
    constexpr auto R = I;
    constexpr auto A =
        MatBuilder(-2.0, 0.0, 0.0, 0.0, -2.5, 0.0, 0.0, 0.0, -3.0);
    constexpr auto B = I;
    constexpr auto C = I;
    constexpr auto ss =
        StateSpace<TimeDomainType::CONTINUOUS, Space, Space, Space>(A, B, C);

    constexpr auto P_geq =
        Inequality<InequalityType::GEQ, decltype(P), Space>(P, 1e-6 * I);
    constexpr auto lqr = LyapunovCertificate<decltype(ss), decltype(Q), decltype(R)>();
    constexpr auto lmi = lqr(ss, Q, R, P);
    constexpr auto prob_constr =
        make_problem(P).add_constraint(P_geq).in_set(lmi);
    constexpr auto c = Vectord<6>{1.0, 0.0, 1.0, 0.0, 0.0, 1.0};
    constexpr auto obj = WeightedSum<PSpace>()(c, P);
    constexpr auto prob = prob_constr.minimize(obj);

    STATIC_REQUIRE(is_problem_set_v<decltype(prob)>);
    STATIC_REQUIRE(prob.validate());

    using Solver = PrimalDualIPM<decltype(prob)>;
    const auto cfg = lqr_solver_config<double>();

    const auto p1 = 2.0 - constexpr_sqrt_safe(3.0);
    const auto p2 = 2.5 - constexpr_sqrt_safe(5.25);
    const auto p3 = 3.0 - constexpr_sqrt_safe(8.0);
    const auto P_star = MatBuilder(p1, 0.0, 0.0, 0.0, p2, 0.0, 0.0, 0.0, p3);
    const auto P_inner = P_star + 1e-3 * I;
    const auto P_near = P_star + 1e-5 * I;

    const Solver cold_solver(prob, cfg);
    const auto& cold_res = cold_solver.res;
#ifdef AKREON_DEBUG
    const auto& cold_trace = cold_solver.trace;
    if (cold_res.tag != SolverResult::OPTIMAL || cold_res.stalled)
      log(cold_res, cold_trace);
#endif

    REQUIRE(cold_res.tag == SolverResult::OPTIMAL);
    REQUIRE(!cold_res.stalled);

    PrimalDualIterate<typename Solver::Var, typename Solver::ConeVar,
                      typename Solver::NuVar>
        warm{};
    warm.z = get_svec(P_inner);
    warm.x = prob.constraint_map().linear_part() * warm.z +
             prob.constraint_map().translation();
    warm.y = warm.x;

    const Solver warm_solver(prob, warm, cfg);
    const auto& warm_res = warm_solver.res;
#ifdef AKREON_DEBUG
    const auto& warm_trace = warm_solver.trace;
    if (warm_res.tag != SolverResult::OPTIMAL || warm_res.stalled)
      log(warm_res, warm_trace);
#endif

    REQUIRE(warm_res.tag == SolverResult::OPTIMAL);
    REQUIRE(!warm_res.stalled);

    PrimalDualIterate<typename Solver::Var, typename Solver::ConeVar,
                      typename Solver::NuVar>
        near{};
    near.z = get_svec(P_near);
    near.x = prob.constraint_map().linear_part() * near.z +
             prob.constraint_map().translation();
    near.y = near.x;

    const Solver near_solver(prob, near, cfg);
    const auto& near_res = near_solver.res;
#ifdef AKREON_DEBUG
    const auto& near_trace = near_solver.trace;
    if (near_res.tag != SolverResult::OPTIMAL || near_res.stalled)
      log(near_res, near_trace);
#endif

    REQUIRE(near_res.tag == SolverResult::OPTIMAL);
    REQUIRE(!near_res.stalled);

    const auto P_cold = make_smat<3>(cold_res.iterate.z);
    const auto P_warm = make_smat<3>(warm_res.iterate.z);
    const auto P_near_res = make_smat<3>(near_res.iterate.z);
    REQUIRE(all_equal(P_cold, transpose(P_cold), rel_tol));
    REQUIRE(all_equal(P_warm, transpose(P_warm), rel_tol));
    REQUIRE(all_equal(P_near_res, transpose(P_near_res), rel_tol));
    REQUIRE(P_cold(0)(0) > 1e-6);
    REQUIRE(P_cold(1)(1) > 1e-6);
    REQUIRE(P_cold(2)(2) > 1e-6);
    REQUIRE(P_warm(0)(0) > 1e-6);
    REQUIRE(P_warm(1)(1) > 1e-6);
    REQUIRE(P_warm(2)(2) > 1e-6);
    REQUIRE(P_near_res(0)(0) > 1e-6);
    REQUIRE(P_near_res(1)(1) > 1e-6);
    REQUIRE(P_near_res(2)(2) > 1e-6);

    const auto [K_cold, K_cold_event] = lqr.K(P_cold, ss, R);
    const auto [K_warm, K_warm_event] = lqr.K(P_warm, ss, R);
    const auto [K_near, K_near_event] = lqr.K(P_near_res, ss, R);
    REQUIRE(K_cold_event == linalg::NumericalEvent::NONE);
    REQUIRE(K_warm_event == linalg::NumericalEvent::NONE);
    REQUIRE(K_near_event == linalg::NumericalEvent::NONE);
    REQUIRE(all_equal(K_cold, P_cold, 1e-4));
    REQUIRE(all_equal(K_warm, P_warm, 1e-4));
    REQUIRE(all_equal(K_near, P_near_res, 1e-4));
  }
}
