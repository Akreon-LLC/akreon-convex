/**
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#include <catch2/catch_test_macros.hpp>

#include "dynamics.hpp"
#include "unit_test_helpers.hpp"

using namespace dynamics;
using namespace unit_test;
using namespace algebra;

template <typename ScalarT, size_t DoF = 1>
class SMD : public DynamicalSystem<ScalarT, DoF, SMD<ScalarT, DoF>> {
 public:
  constexpr explicit SMD(const Matrix<ScalarT, DoF, 3>& coords,
                         const ScalarT& k, const ScalarT& c, const ScalarT& m)
      : DynamicalSystem<ScalarT, DoF, SMD<ScalarT, DoF>>(coords),
        K(k),
        C(c),
        mass(m) {}

  constexpr ScalarT T_impl(const Vector<ScalarT, DoF>& q0,
                           const Vector<ScalarT, DoF>& q1) const {
    const ScalarT delta = q1(0) - q0(0);
    return 0.5 * mass * delta * delta;
  }

  constexpr ScalarT V_impl(const Vector<ScalarT, DoF>& q) const {
    return 0.5 * K * q(0) * q(0);
  }

  constexpr Vector<ScalarT, DoF> Qnc_impl(const Vector<ScalarT, DoF>& q0,
                                          const Vector<ScalarT, DoF>& q1,
                                          const ScalarT& h) const {
    return Vector<ScalarT, 1>{(-1 * C * (q1(0) - q0(0))) / (2 * h)};
  }

 private:
  ScalarT K, C, mass;
  static constexpr ScalarT G = 9.81;
};

TEST_CASE("Test SMD", "[smd]") {
  SECTION("Static case") {
    constexpr auto Q = seed_Q(Vector<double, 1>{0.0});

    STATIC_REQUIRE(Q(0)(0) == 0.0);
    STATIC_REQUIRE(Q(0)(1) == 1e-4);
    STATIC_REQUIRE(Q(0)(2) == 2e-4);

    constexpr auto smd = SMD<double>(Q, 1.0, 2.0, 3.0);

    constexpr auto T = smd.T(Vector<double, 1>{0.0}, Vector<double, 1>{0.0});
    constexpr auto V = smd.V(Vector<double, 1>{0.0});
    constexpr auto Qnc =
        smd.Qnc(Vector<double, 1>{0.0}, Vector<double, 1>{0.0}, 0.1);
    STATIC_REQUIRE(T == 0.0);
    STATIC_REQUIRE(V == 0.0);
    STATIC_REQUIRE(Qnc == Vector<double, 1>{});
  }

  SECTION("Simulate simple case") {
    Vector<double, 10> expected{1.00015, 1.00015,  1.0001,   0.999999, 0.99985,
                                0.99965, 0.999402, 0.999104, 0.998757, 0.99836};
    const double h = 0.01;
    auto Q = seed_Q(Vector<double, 1>{1.0});
    for (size_t i = 0; i < 10; ++i) {
      const auto result = Solve(SMD<double>(Q, 1.0, 0.5, 2.0), h);
      Q = std::get<0>(result);
      REQUIRE(L2<double, 1>(Vector<double, 1>{Q(0)(2) - expected(i)}) <= 1e-4);
    }
  }
}
