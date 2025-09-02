/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#include <catch2/catch_test_macros.hpp>
#include <iostream>

#include "dynamics.hpp"
#include "unit_test_helpers.hpp"

using namespace dynamics;

template <typename ScalarT, size_t DoF = 1>
class Pendulum : public DynamicalSystem<ScalarT, DoF, Pendulum<ScalarT, DoF>> {
 public:
  constexpr explicit Pendulum(const algebra::Vector<ScalarT, DoF>& thetas,
                              const ScalarT& len, const ScalarT& m)
      : DynamicalSystem<ScalarT, DoF, Pendulum<ScalarT, DoF>>(thetas),
        length(len),
        mass(m) {}

  constexpr ScalarT T_impl(const algebra::Vector<ScalarT, DoF>& q0,
                           const algebra::Vector<ScalarT, DoF>& q1) const {
    const ScalarT v = length * (q1(0) - q0(0));
    return 0.5 * mass * v * v;
  }

  constexpr ScalarT V_impl(const algebra::Vector<ScalarT, DoF>& q) const {
    return mass * G * length * (1 - algebra::constexpr_cos(q(0)));
  }

  constexpr algebra::Vector<ScalarT, DoF> Qnc_impl(
      const algebra::Vector<ScalarT, DoF>& q0,
      const algebra::Vector<ScalarT, DoF>& q1, const ScalarT& h) const {
    (void)q0, (void)q1, (void)h;
    return algebra::Vector<ScalarT, DoF>();
  }

 private:
  ScalarT length;
  ScalarT mass;
  static constexpr ScalarT G = 9.81;
};

TEST_CASE("Test Pendulum", "[pendulum]") {
  SECTION("Static case") {
    constexpr auto pendulum =
        Pendulum<double>(algebra::Vector<double, 1>{0.0}, 1.0, 2.0);
    constexpr auto Q = pendulum.get_Q();
    STATIC_REQUIRE(Q(0)(0) == 0.0);
    STATIC_REQUIRE(Q(0)(1) == 1e-4);
    //STATIC_REQUIRE(Q(0)(2) == 2e-4);
    constexpr auto T = pendulum.T(algebra::Vector<double, 1>{0.0},
                                  algebra::Vector<double, 1>{0.0});
    constexpr auto V = pendulum.V(algebra::Vector<double, 1>{0.0});
    constexpr auto Qnc = pendulum.Qnc(algebra::Vector<double, 1>{0.0},
                                      algebra::Vector<double, 1>{0.0}, 0.1);
    STATIC_REQUIRE(T == 0.0);
    STATIC_REQUIRE(V == 0.0);
    STATIC_REQUIRE(Qnc == algebra::Vector<double, 1>{});
  }

  SECTION("Simulate simple case {theta = pi/4, h = 0.01, epochs = 10}") {
    algebra::Vector<double, 10> expected{0.784905, 0.783618, 0.781638, 0.778967,
                                         0.775607, 0.77156,  0.766829, 0.761417,
                                         0.755329, 0.748568};
    auto pendulum =
        Pendulum<double>(algebra::Vector<double, 1>{M_PI / 4}, 1.0, 1.0);
    const double h = 0.01;
    for (size_t i = 0; i < 10; ++i) {
      const auto result = Solve(pendulum, h);
      pendulum = static_cast<const Pendulum<double, 1>&>(std::get<0>(result));
      const auto Q = pendulum.get_Q();
      std::cout << Q(0)(0) << " " << Q(0)(1) << " " << Q(0)(2) << std::endl;
      //REQUIRE(Q(0)(2) == expected(i));
    }
  }
}
