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

#pragma once

#include "math.hpp"

namespace dynamics {

template <typename ScalarT, std::size_t DoF, std::size_t Nu,
          typename DerivedSystem>
  requires(linalg::Field<ScalarT>)
class DynamicalSystem {
public:
  constexpr explicit DynamicalSystem(
      const linalg::Matrix<ScalarT, DoF, 3>& coords)
    requires(Nu == 0)
      : Q(coords) {}

  constexpr explicit DynamicalSystem(
      const linalg::Matrix<ScalarT, DoF, 3>& coords,
      const linalg::Vector<ScalarT, Nu>& inputs)
    requires(Nu > 0)
      : Q(coords), U(inputs) {}

  DynamicalSystem(const DynamicalSystem& system) = delete;
  DynamicalSystem(DynamicalSystem&& system) = delete;
  DynamicalSystem& operator=(const DynamicalSystem& system) = delete;
  DynamicalSystem& operator=(DynamicalSystem&& system) = delete;

  constexpr ScalarT T(const linalg::Vector<ScalarT, DoF>& q0,
                      const linalg::Vector<ScalarT, DoF>& q1) const {
    return self().T_impl(q0, q1);
  }

  constexpr ScalarT V(const linalg::Vector<ScalarT, DoF>& q) const {
    return self().V_impl(q);
  }

  constexpr linalg::Vector<ScalarT, DoF> Qnc(
      const linalg::Vector<ScalarT, DoF>& q0,
      const linalg::Vector<ScalarT, DoF>& q1,
      const linalg::Vector<ScalarT, Nu>& U_in, const ScalarT& h) const {
    return self().Qnc_impl(q0, q1, U_in, h);
  }

  constexpr auto get_Q() const { return Q; }

  constexpr auto get_U() const { return U; }

protected:
  linalg::Matrix<ScalarT, DoF, 3> Q;
  linalg::Vector<ScalarT, Nu> U;

private:
  constexpr const DerivedSystem& self() const {
    return static_cast<const DerivedSystem&>(*this);
  }
};

template <typename ScalarT, std::size_t DoF>
constexpr auto seed_Q(const linalg::Vector<ScalarT, DoF>& ic,
                      ScalarT perturbation = 1e-4) {
  linalg::Matrix<ScalarT, DoF, 3> Q;
  for (std::size_t i = 0; i < DoF; ++i) {
    Q(i, 0) = ic(i);
    Q(i, 1) = ic(i) + perturbation;
    Q(i, 2) = Q(i, 1) + (Q(i, 1) - Q(i, 0));
  }
  return Q;
}

} // namespace dynamics
