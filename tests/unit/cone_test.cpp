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

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <iostream>

#include "math.hpp"
#include "unit_test_helpers.hpp"

using namespace linalg;
using namespace convex;
using namespace affine;
using namespace conic;
using namespace unit_test;

TEST_CASE("Test cone formations", "[cone_test]") {
  SECTION("Test exp cone") {
    using Slack = Vectord<3>;
    constexpr ExpCone<Slack> exp_cone;
    STATIC_REQUIRE(exp_cone.in_cone(exp_cone.x0()));

    constexpr ExpCone<Vectord<9>> exp_cone9;
    STATIC_REQUIRE(exp_cone9.in_cone(exp_cone9.x0()));
  }

  SECTION("Test power cone") {
    using Slack = Vectord<4>;
    constexpr vec_traits<Vectord<3>> vec_t;
    constexpr PowerCone<Slack> pcone(vec_t.fill(1.0 / 3.0));
    STATIC_REQUIRE(pcone.in_cone(pcone.x0()));
  }

  SECTION("Test psd cone") {
    using Slack = Vectord<36>;
    constexpr PositiveSemidefiniteCone<Slack, 8> psd;
    STATIC_REQUIRE(psd.in_cone(psd.x0()));
  }

  SECTION("Test rsoc") {
    using Slack = Vectord<3>;
    constexpr RotatedSecondOrderCone<Slack> rsoc;
    STATIC_REQUIRE(rsoc.in_cone(rsoc.x0()));
  }

  SECTION("Test soc") {
    using Slack = Vectord<10>;
    constexpr SecondOrderCone<Slack> soc;
    STATIC_REQUIRE(soc.in_cone(soc.x0()));
  }

  SECTION("Test zoc") {
    using Slack = Vectord<5>;
    constexpr ZeroOrderCone<Slack> zoc;
    STATIC_REQUIRE(!zoc.in_cone(zoc.x0()));
  }
}

TEST_CASE("Test R^n_+ cone", "[cone_test]") {
  SECTION("Test rplus cone") {
    using Slack = Vectord<12>;
    constexpr NonnegOrthantCone<Slack> rplus;
    STATIC_REQUIRE(rplus.in_cone(rplus.x0()));
    STATIC_REQUIRE(rplus.in_cone(rplus.s0()));

    constexpr vec_traits<Slack> vec;
    constexpr auto p = vec.fill(2.0);
    STATIC_REQUIRE(rplus.alpha_max(rplus.x0(), p).value() == 0.5);

    constexpr auto H = rplus.H(rplus.x0(), rplus.s0());
    STATIC_REQUIRE(all_equal(H.H, make_identity<double, 12>()));
    STATIC_REQUIRE(H(0, 0) == 1.0);

    constexpr auto H_inv = rplus.H_inv(rplus.x0(), rplus.s0());
    STATIC_REQUIRE(all_equal(H_inv.H, make_identity<double, 12>()));
    STATIC_REQUIRE(H_inv(0, 0) == 1.0);

    constexpr auto grad = rplus.grad(rplus.x0());
    STATIC_REQUIRE(grad == vec.fill(-1.0));

    constexpr auto gamma_F = rplus.functional_proximity(rplus.x0(), rplus.s0());
    STATIC_REQUIRE(gamma_F == 12.0 * constexpr_ln(12.0).value());
  }
}
