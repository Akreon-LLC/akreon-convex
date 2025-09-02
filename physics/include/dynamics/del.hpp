/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#include <functional>

#include "dynamical_system.hpp"

#pragma once

namespace dynamics {

template <typename ScalarT, size_t DoF, typename DerivedSystem>
struct Result {
  DynamicalSystem<ScalarT, DoF, DerivedSystem> system;
  algebra::Matrix<ScalarT, DoF, DoF> J;
};

template <typename ScalarT, size_t DoF>
  requires(algebra::FieldLike<ScalarT>)
constexpr auto Ld(
    const algebra::Vector<ScalarT, DoF>& q0,
    const algebra::Vector<ScalarT, DoF>& q1,
    std::function<ScalarT(const algebra::Vector<ScalarT, DoF>&,
                          const algebra::Vector<ScalarT, DoF>&)>
        T,
    std::function<ScalarT(const algebra::Vector<ScalarT, DoF>&)> V,
    const double& h) {
  const auto kinetic = T(q0, q1) / (h * h);
  const auto conservative = 0.5 * (V(q0) + V(q1));
  return h * (kinetic - conservative);
}

template <typename ScalarT, size_t DoF>
  requires(algebra::FieldLike<ScalarT>)
constexpr auto D2Ld(
    const algebra::Matrix<ScalarT, DoF, 3>& Q,
    std::function<ScalarT(const algebra::Vector<ScalarT, DoF>&,
                          const algebra::Vector<ScalarT, DoF>&)>
        T,
    std::function<ScalarT(const algebra::Vector<ScalarT, DoF>&)> V,
    const ScalarT& h, ScalarT eps = 1e-8) {
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

template <typename ScalarT, size_t DoF>
  requires(algebra::FieldLike<ScalarT>)
constexpr auto D1Ld(
    const algebra::Matrix<ScalarT, DoF, 3>& Q,
    std::function<ScalarT(const algebra::Vector<ScalarT, DoF>&,
                          const algebra::Vector<ScalarT, DoF>&)>
        T,
    std::function<ScalarT(const algebra::Vector<ScalarT, DoF>&)> V,
    const ScalarT& h, ScalarT eps = 1e-8) {
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
  const auto T =
      std::bind(&DynamicalSystem<ScalarT, DoF, DerivedSystem>::T, &system,
                std::placeholders::_1, std::placeholders::_2);
  const auto V = std::bind(&DynamicalSystem<ScalarT, DoF, DerivedSystem>::V,
                           &system, std::placeholders::_1);

  const auto Qnc = std::bind(&DynamicalSystem<ScalarT, DoF, DerivedSystem>::Qnc,
                             &system, std::placeholders::_1,
                             std::placeholders::_2, std::placeholders::_3);
  // (dakre) solve DEL via root finding (newton's method)
  auto DEL = [&](const auto& Q) {
    return D2Ld(Q, T, V, h, eps) + D1Ld(Q, T, V, h, eps) +
           h * Qnc(get_col(Q, 0), get_col(Q, 2), h);
  };

  algebra::Matrix<ScalarT, DoF, DoF> J;
  algebra::Vector<ScalarT, DoF> F;
  algebra::Vector<ScalarT, DoF> Q_delta;

  auto Q = system.get_Q();
  for (size_t i = 0; i < MAX_STEP_EVAL; ++i) {
    J = make_zeros<ScalarT, DoF, DoF>();
    F = DEL(Q);
    for (size_t j = 0; j < DoF; ++j) {
      Q[j][2] += eps;
      const auto pos = DEL(Q);
      Q[j][2] -= (2 * eps);
      const auto neg = DEL(Q);
      Q[j][2] += eps;
      const auto grad = (pos - neg) / (2 * eps);
      for (size_t k = 0; k < DoF; ++k) J[j][k] = grad(k);
    }

    // Qkp1 = J^{-1} F where -F = J * (Qkp1 - Qk)... Q_delta = Qkp1 - Qk
    Q_delta = algebra::QR(J).solve(-F);

    // Update Qkp1 column for next timestep
    for (size_t j = 0; j < DoF; ++j) Q[j][2] += Q_delta(j);

    if (algebra::L2(Q_delta) < tol) break;
  }

  Result<ScalarT, DoF, DerivedSystem> result;
  result.system = DynamicalSystem<ScalarT, DoF, DerivedSystem>(Q);
  result.J = J;

  return result;
}

}  // namespace dynamics
