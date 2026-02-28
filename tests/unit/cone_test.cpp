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

  SECTION("Test zoc") {
    using Slack = Vectord<5>;
    constexpr ZeroOrderCone<Slack> zoc;
    STATIC_REQUIRE(!zoc.in_cone(zoc.x0()));
  }
}

TEST_CASE("Test R^n_+ cone", "[cone_test]") {
  SECTION("Test rplus cone api") {
    using Slack = Vectord<12>;
    constexpr NonnegOrthantCone<Slack> rplus;
    STATIC_REQUIRE(rplus.in_cone(rplus.x0()));
    STATIC_REQUIRE(rplus.in_cone(rplus.y0()));

    constexpr vec_traits<Slack> vec;
    constexpr auto p = vec.fill(2.0);
    STATIC_REQUIRE(
        linalg::constexpr_abs(rplus.alpha_max(rplus.x0(), p) - 0.99) < eps);

    constexpr auto w = rplus.w(rplus.x0(), rplus.y0());
    constexpr auto ones = vec.fill(1.0);
    STATIC_REQUIRE(L2<Slack>()(w - ones) < eps);

    constexpr auto P = rplus.H(rplus.x0(), rplus.y0());
    STATIC_REQUIRE(P.apply(Slack{}) == Slack{});

    constexpr auto Pinv = rplus.H_inv(rplus.x0(), rplus.y0());
    STATIC_REQUIRE(Pinv.apply(Slack{}) == Slack{});

    constexpr auto e = vec.fill(1.0);
    STATIC_REQUIRE(rplus.e() == e);

    constexpr RplusCirc<Slack> circ;
    STATIC_REQUIRE(circ(rplus.x0(), rplus.y0()) == e);

    STATIC_REQUIRE(rplus.nu() == 12.0);
  }
}

TEST_CASE("Test SOC", "[cone_test]") {
  SECTION("Test soc cone api") {
    using Slack = Vectord<10>;
    constexpr SecondOrderCone<Slack> soc;
    STATIC_REQUIRE(soc.in_cone(soc.x0()));
    STATIC_REQUIRE(soc.in_cone(soc.y0()));

    constexpr vec_traits<Slack> vec;

    /// TODO - test alpha max
    STATIC_REQUIRE(soc.nu() == 2.0);

    constexpr Vectord<10> e{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0};
    STATIC_REQUIRE(soc.e() == e);

    constexpr SOCCirc<Slack> circ;
    constexpr auto x = vec.fill(5.5);
    STATIC_REQUIRE(circ(x, soc.e()) == x);
  }
}
