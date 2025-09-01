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
  constexpr ScalarT T(const algebra::Vector<ScalarT, DoF>& q0,
                      const algebra::Vector<ScalarT, DoF>& q1) const {
    static_cast<DerivedSystem*>(this)->T(q0, q1);
  }

  constexpr ScalarT V(const algebra::Vector<ScalarT, DoF>& q) const {
    static_cast<DerivedSystem*>(this)->V(q);
  }

  constexpr ScalarT Qnc(const algebra::Vector<ScalarT, DoF>& q0,
                        const algebra::Vector<ScalarT, DoF>& q1,
                        const ScalarT& h) {
    static_cast<DerivedSystem*>(this)->Qnc(q0, q1, h);
  }

  constexpr auto GetQ() const { return Q; }

 protected:
  algebra::Matrix<ScalarT, 3, DoF> Q;
  const ScalarT G = 9.80665;
};

}  // namespace dynamics
