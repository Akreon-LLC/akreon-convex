/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "math.hpp"

namespace dynamics {

template <typename ScalarT, size_t DoF, typename DerivedSystem>
  requires(algebra::FieldLike<ScalarT>)
class DynamicalSystem {
 public:
  constexpr explicit DynamicalSystem(
      const algebra::Matrix<ScalarT, DoF, 3>& coords)
      : Q(coords) {}

  DynamicalSystem(const DynamicalSystem& system) = delete;
  DynamicalSystem(DynamicalSystem&& system) = delete;
  DynamicalSystem& operator=(const DynamicalSystem& system) = delete;
  DynamicalSystem& operator=(DynamicalSystem&& system) = delete;

  constexpr ScalarT T(const algebra::Vector<ScalarT, DoF>& q0,
                      const algebra::Vector<ScalarT, DoF>& q1) const {
    return static_cast<const DerivedSystem*>(this)->T_impl(q0, q1);
  }

  constexpr ScalarT V(const algebra::Vector<ScalarT, DoF>& q) const {
    return static_cast<const DerivedSystem*>(this)->V_impl(q);
  }

  constexpr algebra::Vector<ScalarT, DoF> Qnc(
      const algebra::Vector<ScalarT, DoF>& q0,
      const algebra::Vector<ScalarT, DoF>& q1, const ScalarT& h) const {
    return static_cast<const DerivedSystem*>(this)->Qnc_impl(q0, q1, h);
  }

  constexpr auto get_Q() const { return Q; }

 protected:
  algebra::Matrix<ScalarT, DoF, 3> Q;
};

template <typename ScalarT, size_t DoF>
constexpr auto seed_Q(const algebra::Vector<ScalarT, DoF>& ic,
                      ScalarT perturbation = 1e-4) {
  algebra::Matrix<ScalarT, DoF, 3> Q;
  for (size_t i = 0; i < DoF; ++i) {
    Q[i][0] = ic(i);
    Q[i][1] = ic(i) + perturbation;
    Q[i][2] = Q(i)(1) + (Q(i)(1) - Q(i)(0));
  }
  return Q;
}

}  // namespace dynamics
