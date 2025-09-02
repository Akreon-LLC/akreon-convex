/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#include <iostream>
#include <tuple>

#include "dynamical_system.hpp"

#pragma once

namespace dynamics {

template <typename ScalarT, size_t DoF, typename DerivedSystem>
  requires(algebra::FieldLike<ScalarT>)
constexpr auto Ld(const algebra::Vector<ScalarT, DoF>& q0,
                  const algebra::Vector<ScalarT, DoF>& q1,
                  const DynamicalSystem<ScalarT, DoF, DerivedSystem>& system,
                  const double& h) {
  const auto kinetic = system.T(q0, q1) / (h * h);
  const auto conservative = 0.5 * (system.V(q0) + system.V(q1));
  return h * (kinetic - conservative);
}

template <typename ScalarT, size_t DoF, typename DerivedSystem>
  requires(algebra::FieldLike<ScalarT>)
constexpr auto D2Ld(const DynamicalSystem<ScalarT, DoF, DerivedSystem>& system,
                    const ScalarT& h, ScalarT eps = 1e-8) {
  auto Q = system.get_Q();
  algebra::Vector<ScalarT, DoF> grad;
  for (size_t i = 0; i < DoF; ++i) {
    Q[i][1] += eps;
    const auto pos = Ld(get_col(Q, 0), get_col(Q, 1), system, h);
    Q[i][1] -= (2 * eps);
    const auto neg = Ld(get_col(Q, 0), get_col(Q, 1), system, h);
    Q[i][1] += eps;
    grad[i] = (pos - neg) / (2 * eps);
  }
  return grad;
}

template <typename ScalarT, size_t DoF, typename DerivedSystem>
  requires(algebra::FieldLike<ScalarT>)
constexpr auto D1Ld(const DynamicalSystem<ScalarT, DoF, DerivedSystem>& system,
                    const ScalarT& h, ScalarT eps = 1e-8) {
  auto Q = system.get_Q();
  algebra::Vector<ScalarT, DoF> grad;
  for (size_t i = 0; i < DoF; ++i) {
    Q[i][1] += eps;
    const auto pos = Ld(get_col(Q, 1), get_col(Q, 2), system, h);
    Q[i][1] -= (2 * eps);
    const auto neg = Ld(get_col(Q, 1), get_col(Q, 2), system, h);
    Q[i][1] += eps;
    grad[i] = (pos - neg) / (2 * eps);
  }
  return grad;
}

template <typename ScalarT, size_t DoF, typename DerivedSystem>
constexpr auto Qnc(const DynamicalSystem<ScalarT, DoF, DerivedSystem>& system,
                   const ScalarT& h) {
  const auto Q = system.get_Q();
  return h * system.Qnc(get_col(Q, 0), get_col(Q, 2), h);
}

template <typename ScalarT, size_t DoF, typename DerivedSystem>
constexpr auto Solve(DynamicalSystem<ScalarT, DoF, DerivedSystem> system,
                     const ScalarT& h, ScalarT eps = 1e-8, ScalarT tol = 1e-3) {
  // (dakre) solve DEL via root finding (newton's method)
  auto DEL = [&](const auto& sys) -> algebra::Vector<ScalarT, DoF> {
    return D2Ld(sys, h, eps) + D1Ld(sys, h, eps) + Qnc(sys, h);
  };

  algebra::Matrix<ScalarT, DoF, DoF> J;
  algebra::Matrix<ScalarT, DoF, 3> Q;

  for (size_t i = 0; i < MAX_STEP_EVAL; ++i) {
    algebra::Vector<ScalarT, DoF> F, Q_delta;

    J = algebra::make_zeros<ScalarT, DoF, DoF>();
    F = DEL(system);
    Q = system.get_Q();
    for (size_t j = 0; j < DoF; ++j) {
      Q[j][2] += eps;
      system = DynamicalSystem<ScalarT, DoF, DerivedSystem>(Q);
      const auto pos = DEL(system);

      Q[j][2] -= (2 * eps);
      system = DynamicalSystem<ScalarT, DoF, DerivedSystem>(Q);
      const auto neg = DEL(system);

      Q[j][2] += eps;
      const auto grad = (1 / (2 * eps)) * (pos - neg);
      for (size_t k = 0; k < DoF; ++k) J[j][k] = grad(k);
    }

    // Qkp1 = J^{-1} F where -F = J * (Qkp1 - Qk)... Q_delta = Qkp1 - Qk
    Q_delta = algebra::QR(J).solve(-1 * F);

    // Update Qkp1 column for next timestep
    for (size_t j = 0; j < DoF; ++j) Q[j][2] += Q_delta(j);

    system = DynamicalSystem<ScalarT, DoF, DerivedSystem>(Q);

    std::cout << i << std::endl;

    if (algebra::L2(Q_delta) < tol) break;
  }

  for (size_t i = 0; i < DoF; ++i) {
    Q[i][0] = Q(i)(1);
    Q[i][1] = Q(i)(2);
  }
  system = DynamicalSystem<ScalarT, DoF, DerivedSystem>(Q);

  return std::tuple{system, J};
}

}  // namespace dynamics
