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
    STATIC_REQUIRE(exp_cone.in_K(exp_cone.x0()));
  }

  SECTION("Test power cone") {
    using Slack = Vectord<4>;
    constexpr vec_traits<Vectord<3>> vec_t;
    constexpr PowerCone<Slack> pcone(vec_t.fill(1.0 / 3.0));
    STATIC_REQUIRE(pcone.in_K(pcone.x0()));
  }

  SECTION("Test psd cone") {
    using Slack = Vectord<36>;
    constexpr PositiveSemidefiniteCone<Slack, 8> psd;
    STATIC_REQUIRE(psd.in_K(psd.x0()));
  }

  SECTION("Test rplus cone") {
    using Slack = Vectord<12>;
    constexpr NonnegOrthantCone<Slack> rplus;
    STATIC_REQUIRE(rplus.in_K(rplus.x0()));
  }

  SECTION("Test rsoc") {
    using Slack = Vectord<3>;
    constexpr RotatedSecondOrderCone<Slack> rsoc;
    STATIC_REQUIRE(rsoc.in_K(rsoc.x0()));
  }

  SECTION("Test soc") {
    using Slack = Vectord<10>;
    constexpr SecondOrderCone<Slack> soc;
    STATIC_REQUIRE(soc.in_K(soc.x0()));
  }

  SECTION("Test zoc") {
    using Slack = Vectord<5>;
    constexpr ZeroOrderCone<Slack> zoc;
    STATIC_REQUIRE(!zoc.in_K(zoc.x0()));
  }
}
