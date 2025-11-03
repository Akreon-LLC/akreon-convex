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


#include <functional>
#include <tuple>

#include "dynamical_system.hpp"

#pragma once

namespace dynamics {

template <typename T_fn, typename V_fn, typename ScalarT, size_t DoF>
  requires(linalg::FieldLike<ScalarT>)
constexpr auto Ld(const linalg::Vector<ScalarT, DoF>& q0,
                  const linalg::Vector<ScalarT, DoF>& q1, const T_fn& T,
                  const V_fn& V, const double& h) {
  const auto kinetic = T(q0, q1) / (h * h);
  const auto conservative = 0.5 * (V(q0) + V(q1));
  return h * (kinetic - conservative);
}

template <size_t k, typename T_fn, typename V_fn, typename ScalarT, size_t DoF>
  requires(linalg::FieldLike<ScalarT> && k < 2)
constexpr auto DLd(linalg::Matrix<ScalarT, DoF, 3> Q, const T_fn& T,
                   const V_fn& V, const ScalarT& h, ScalarT eps = 1e-8) {
  linalg::Vector<ScalarT, DoF> grad;
  for (size_t i = 0; i < DoF; ++i) {
    Q[i][1] += eps;
    const auto pos = Ld(get_col(Q, k), get_col(Q, k + 1), T, V, h);
    Q[i][1] -= (2 * eps);
    const auto neg = Ld(get_col(Q, k), get_col(Q, k + 1), T, V, h);
    Q[i][1] += eps;
    grad[i] = (pos - neg) / (2 * eps);
  }
  return grad;
}

template <size_t k, typename Qnc_fn, typename ScalarT, size_t DoF, size_t Nu>
  requires(linalg::FieldLike<ScalarT> && k < 2)
constexpr auto Fd(const linalg::Matrix<ScalarT, DoF, 3>& Q,
                  const linalg::Vector<ScalarT, Nu>& U, const Qnc_fn& Qnc,
                  const ScalarT& h) {
  // (dakre) apply trapezoidal midpoint for each half Fd^- and Fd^+ of discrete
  // forces Qnc
  const auto& qk = get_col(Q, k);
  const auto& qkp1 = get_col(Q, k + 1);

  // (dakre) TODO look into the below vs. what is expected in marsden text
  // (current test accuracy fails the below)
#if 0
  linalg::Vector<ScalarT, DoF> v_mid;
  for (size_t i = 0; i < DoF; ++i) v_mid[i] = (qkp1(i) - qk(i)) / h;

  if constexpr (k == 0)
    return 0.5 * h * Qnc(qk, v_mid, U, h);
  else
    return 0.5 * h * Qnc(qkp1, v_mid, U, h);
#endif
  return 0.5 * h * Qnc(qk, qkp1, U, h);
}

// (dakre) TODO this is the first stab at the DEL VI solving, it should be
// fleshed out more in the coming months with auto diff, better integration,
// etc... likewise the RHS solving for A & B to be constexpr template? User may
// just want to simulate and not care about forming A and B... likewise
// underlying qr solve could be better leveraged with LDL or something else...
// so need to test out and improve accuracy, speed, and memory footprint
template <typename ScalarT, size_t DoF, size_t Nu, typename DerivedSystem>
constexpr auto Solve(
    const DynamicalSystem<ScalarT, DoF, Nu, DerivedSystem>& system,
    const ScalarT& h, ScalarT eps = 1e-8, ScalarT tol = 1e-4) {
  auto T = std::bind(&DynamicalSystem<ScalarT, DoF, Nu, DerivedSystem>::T,
                     &system, std::placeholders::_1, std::placeholders::_2);
  auto V = std::bind(&DynamicalSystem<ScalarT, DoF, Nu, DerivedSystem>::V,
                     &system, std::placeholders::_1);
  auto Qnc = std::bind(&DynamicalSystem<ScalarT, DoF, Nu, DerivedSystem>::Qnc,
                       &system, std::placeholders::_1, std::placeholders::_2,
                       std::placeholders::_3, std::placeholders::_4);

  // (dakre) solve DEL via root finding (newton's method)
  auto DEL = [&](const auto& Q, const auto& U) -> linalg::Vector<ScalarT, DoF> {
    return DLd<0>(Q, T, V, h, eps) + DLd<1>(Q, T, V, h, eps) +
           Fd<0>(Q, U, Qnc, h) + Fd<1>(Q, U, Qnc, h);
  };

  auto calc_J = [&eps, &DEL](auto Q, const auto& U,
                             size_t idx) -> linalg::Matrix<ScalarT, DoF, DoF> {
    linalg::Matrix<ScalarT, DoF, DoF> J;
    for (size_t j = 0; j < DoF; ++j) {
      Q[j][idx] += eps;
      const auto pos = DEL(Q, U);

      Q[j][idx] -= (2 * eps);
      const auto neg = DEL(Q, U);

      const auto grad = (1 / (2 * eps)) * (pos - neg);
      for (size_t k = 0; k < DoF; ++k) J[k][j] = grad(k);
    }
    return J;
  };

  auto calc_J_wrt_U = [&eps, &DEL](const auto& Q,
                                   auto U) -> linalg::Matrix<ScalarT, DoF, Nu> {
    linalg::Matrix<ScalarT, DoF, Nu> J;
    for (size_t j = 0; j < Nu; ++j) {
      U[j] += eps;
      const auto pos = DEL(Q, U);
      U[j] -= eps;
      const auto neg = DEL(Q, U);

      const auto grad = (1 / (2 * eps)) * (pos - neg);
      for (size_t k = 0; k < DoF; ++k) J[k][j] = grad(k);
    }
    return J;
  };

  auto solve_ss = [&](const auto& Q, const auto& U, const auto& J_qr)
      -> std::tuple<linalg::Matrix<ScalarT, 2 * DoF, 2 * DoF>,
                    linalg::Matrix<ScalarT, 2 * DoF, Nu>> {
    const auto R_qkm1 = calc_J(Q, U, 0);
    const auto R_qk = calc_J(Q, U, 1);

    constexpr size_t RHS_COLS = (Nu > 0) ? 2 * DoF + Nu : 2 * DoF;

    linalg::Matrix<ScalarT, DoF, RHS_COLS> RHS;
    RHS = linalg::set_block<0, DoF, 0, DoF>(RHS, ScalarT(-1) * R_qk);
    RHS = linalg::set_block<0, DoF, DoF, 2 * DoF>(RHS, ScalarT(-1) * R_qkm1);
    if constexpr (0 < Nu) {
      const auto R_u = calc_J_wrt_U(Q, U);
      RHS =
          linalg::set_block<0, DoF, 2 * DoF, RHS_COLS>(RHS, ScalarT(-1) * R_u);
    }

    linalg::Matrix<ScalarT, DoF, RHS_COLS> S;
    for (size_t i = 0; i < RHS_COLS; ++i) {
      const auto x_i = J_qr.solve(get_col(RHS, i));
      for (size_t j = 0; j < DoF; ++j) S[j][i] = x_i(j);
    }

    linalg::Matrix<ScalarT, 2 * DoF, 2 * DoF> A;
    A = linalg::set_block<0, DoF, 0, DoF>(A,
                                          linalg::get_block<0, DoF, 0, DoF>(S));
    A = linalg::set_block<0, DoF, DoF, 2 * DoF>(
        A, linalg::get_block<0, DoF, DoF, 2 * DoF>(S));
    A = linalg::set_block<DoF, 2 * DoF, 0, DoF>(
        A, linalg::make_identity<ScalarT, DoF>());

    linalg::Matrix<ScalarT, 2 * DoF, Nu> B;
    if constexpr (Nu > 0) {
      B = linalg::set_block<0, DoF, 0, Nu>(
          B, linalg::get_block<0, DoF, 2 * DoF, RHS_COLS>(S));
    }

    return {A, B};
  };

  // (dakre) TODO A & B should go back into an appropriate linear SS model that
  // is created in the controls suite... or somehow meta template this func to
  // take it in
  linalg::Matrix<ScalarT, 2 * DoF, 2 * DoF> A;
  linalg::Matrix<ScalarT, 2 * DoF, Nu> B;
  linalg::Matrix<ScalarT, DoF, 3> Q;

  for (size_t i = 0; i < MAX_STEP_EVAL; ++i) {
    Q = system.get_Q();
    const auto U = system.get_U();

    const auto R_qkp1 = calc_J(Q, U, 2);

    // Qkp1 = J^{-1} R where -R = J * (Qkp1 - Qk)... Q_delta = Qkp1 - Qk
    const auto J_qr = linalg::QR(R_qkp1);
    const auto R = DEL(Q, U);
    const auto Q_delta = J_qr.solve(ScalarT(-1) * R);

    const auto ss = solve_ss(Q, U, J_qr);
    A = std::get<0>(ss);
    B = std::get<1>(ss);

    // Update Qkp1 column for next timestep
    for (size_t j = 0; j < DoF; ++j) Q[j][2] += Q_delta(j);

    if (linalg::L2<linalg::Vector<ScalarT, DoF>>()(Q_delta) < tol) break;
  }

  for (size_t i = 0; i < DoF; ++i) {
    Q[i][0] = Q(i)(1);
    Q[i][1] = Q(i)(2);
  }

  return std::tuple{Q, A, B};
}

}  // namespace dynamics
