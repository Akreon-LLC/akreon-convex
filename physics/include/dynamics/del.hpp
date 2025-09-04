/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#include <functional>
#include <tuple>

#include "dynamical_system.hpp"

#pragma once

namespace dynamics {

template <typename T_fn, typename V_fn, typename ScalarT, size_t DoF>
  requires(algebra::FieldLike<ScalarT>)
constexpr auto Ld(const algebra::Vector<ScalarT, DoF>& q0,
                  const algebra::Vector<ScalarT, DoF>& q1, const T_fn& T,
                  const V_fn& V, const double& h) {
  const auto kinetic = T(q0, q1) / (h * h);
  const auto conservative = 0.5 * (V(q0) + V(q1));
  return h * (kinetic - conservative);
}

template <typename T_fn, typename V_fn, typename ScalarT, size_t DoF>
  requires(algebra::FieldLike<ScalarT>)
constexpr auto D2Ld(algebra::Matrix<ScalarT, DoF, 3> Q, const T_fn& T,
                    const V_fn& V, const ScalarT& h, ScalarT eps = 1e-8) {
  algebra::Vector<ScalarT, DoF> grad;
  for (size_t i = 0; i < DoF; ++i) {
    Q[i][1] += eps;
    const auto pos = Ld(get_col(Q, 0), get_col(Q, 1), T, V, h);
    Q[i][1] -= (2 * eps);
    const auto neg = Ld(get_col(Q, 0), get_col(Q, 1), T, V, h);
    Q[i][1] += eps;
    grad[i] = (pos - neg) / (2 * eps);
  }
  return grad;
}

template <typename T_fn, typename V_fn, typename ScalarT, size_t DoF>
  requires(algebra::FieldLike<ScalarT>)
constexpr auto D1Ld(algebra::Matrix<ScalarT, DoF, 3> Q, const T_fn& T,
                    const V_fn& V, const ScalarT& h, ScalarT eps = 1e-8) {
  algebra::Vector<ScalarT, DoF> grad;
  for (size_t i = 0; i < DoF; ++i) {
    Q[i][1] += eps;
    const auto pos = Ld(get_col(Q, 1), get_col(Q, 2), T, V, h);
    Q[i][1] -= (2 * eps);
    const auto neg = Ld(get_col(Q, 1), get_col(Q, 2), T, V, h);
    Q[i][1] += eps;
    grad[i] = (pos - neg) / (2 * eps);
  }
  return grad;
}

template <typename ScalarT, size_t DoF, typename DerivedSystem>
constexpr auto Solve(const DynamicalSystem<ScalarT, DoF, DerivedSystem>& system,
                     const ScalarT& h, ScalarT eps = 1e-8, ScalarT tol = 1e-3) {
  auto T = std::bind(&DynamicalSystem<ScalarT, DoF, DerivedSystem>::T, &system,
                     std::placeholders::_1, std::placeholders::_2);
  auto V = std::bind(&DynamicalSystem<ScalarT, DoF, DerivedSystem>::V, &system,
                     std::placeholders::_1);
  auto Qnc = std::bind(&DynamicalSystem<ScalarT, DoF, DerivedSystem>::Qnc,
                       &system, std::placeholders::_1, std::placeholders::_2,
                       std::placeholders::_3);

  // (dakre) solve DEL via root finding (newton's method)
  auto DEL = [&](const auto& Q) -> algebra::Vector<ScalarT, DoF> {
    return D2Ld(Q, T, V, h, eps) + D1Ld(Q, T, V, h, eps) +
           h * Qnc(get_col(Q, 0), get_col(Q, 2), h);
  };

  algebra::Matrix<ScalarT, DoF, DoF> J;
  algebra::Matrix<ScalarT, DoF, 3> Q;

  for (size_t i = 0; i < MAX_STEP_EVAL; ++i) {
    J = algebra::make_zeros<ScalarT, DoF, DoF>();
    Q = system.get_Q();

    const auto R = DEL(Q);
    for (size_t j = 0; j < DoF; ++j) {
      Q[j][2] += eps;
      const auto pos = DEL(Q);

      Q[j][2] -= (2 * eps);
      const auto neg = DEL(Q);

      Q[j][2] += eps;
      const auto grad = (1 / (2 * eps)) * (pos - neg);
      for (size_t k = 0; k < DoF; ++k) J[k][j] = grad(k);
    }

    // Qkp1 = J^{-1} R where -R = J * (Qkp1 - Qk)... Q_delta = Qkp1 - Qk
    const auto Q_delta = algebra::QR(J).solve(-1 * R);

    // Update Qkp1 column for next timestep
    for (size_t j = 0; j < DoF; ++j) Q[j][2] += Q_delta(j);

    if (algebra::L2(Q_delta) < tol) break;
  }

  for (size_t i = 0; i < DoF; ++i) {
    Q[i][0] = Q(i)(1);
    Q[i][1] = Q(i)(2);
  }

  return std::tuple{Q, J};
}

}  // namespace dynamics
