/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#include <catch2/catch_test_macros.hpp>
#include <iostream>

#include "dynamics.hpp"
#include "unit_test_helpers.hpp"

using namespace dynamics;

template <typename ScalarT, size_t DoF>
class Pendulum : public DynamicalSystem<ScalarT, DoF, Pendulum<ScalarT, DoF>> {
 public:
  constexpr explicit Pendulum(const algebra::Vector<ScalarT, DoF>& thetas,
                              const ScalarT& len, const ScalarT& m)
      : DynamicalSystem<ScalarT, DoF, Pendulum<ScalarT, DoF>>(thetas),
        length(len),
        mass(m) {}

 protected:
  constexpr ScalarT T_impl(const algebra::Vector<ScalarT, DoF>& q0,
                           const algebra::Vector<ScalarT, DoF>& q1) const {
    const ScalarT v = length * (q1(0) - q0(0));
    return 0.5 * mass * v * v;
  }

  constexpr ScalarT V_impl(const algebra::Vector<ScalarT, DoF>& q) const {
    return mass * this->G * length * (1 - constexpr_cos(q(0)));
  }

  constexpr ScalarT Qnc_impl(const algebra::Vector<ScalarT, DoF>& q0,
                             const algebra::Vector<ScalarT, DoF>& q1,
                             const ScalarT& h) const {
    (void)q0, (void)q1, (void)h;
    return ScalarT(0);
  }

 private:
  ScalarT length;
  ScalarT mass;
};

TEST_CASE("Test Pendulum", "[pendulum]") {
  constexpr auto pendulum =
      Pendulum<double, 1>(algebra::Vector<double, 1>{0.0}, 1.0, 2.0);
  constexpr auto Q = pendulum.get_Q();
  STATIC_REQUIRE(Q(0)(0) == 0.0);
  STATIC_REQUIRE(Q(0)(1) == 1e-4);
  STATIC_REQUIRE(Q(0)(2) == 0.0);
}
