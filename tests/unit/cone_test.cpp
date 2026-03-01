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

    STATIC_REQUIRE(rplus.alpha_max(rplus.x0(), p) ==
                   rplus.alpha_max_dual(rplus.y0(), p));

    constexpr auto w = rplus.w(rplus.x0(), rplus.y0());
    constexpr auto ones = vec.fill(1.0);
    STATIC_REQUIRE(L2<Slack>()(w - ones) < eps);

    constexpr auto P = rplus.H(rplus.x0(), rplus.y0());
    STATIC_REQUIRE(P.apply(Slack{}) == Slack{});
    constexpr auto q = vec.fill(0.25);
    STATIC_REQUIRE(L2<Slack>()(P.apply(q) - q) < eps);

    constexpr auto e = vec.fill(1.0);
    STATIC_REQUIRE(rplus.e() == e);

    constexpr RplusCirc<Slack> circ;
    STATIC_REQUIRE(circ(rplus.x0(), rplus.y0()) == e);
    STATIC_REQUIRE(rplus.circ(rplus.x0(), rplus.e()) == rplus.x0());
    STATIC_REQUIRE(rplus.circ(rplus.e(), rplus.y0()) == rplus.y0());

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
    constexpr auto p = vec.fill(0.5);
    STATIC_REQUIRE(linalg::constexpr_abs(soc.alpha_max(soc.x0(), p) - 0.99) <
                   eps);
    STATIC_REQUIRE(soc.alpha_max(soc.x0(), p) ==
                   soc.alpha_max_dual(soc.y0(), p));

    constexpr auto P = soc.H(soc.x0(), soc.y0());
    STATIC_REQUIRE(P.apply(Slack{}) == Slack{});
    constexpr auto q = Slack{0.1, -0.2, 0.3, -0.4, 0.5,
                             -0.6, 0.7, -0.8, 0.9, 0.2};
    STATIC_REQUIRE(L2<Slack>()(P.apply(q) - q) < eps);

    STATIC_REQUIRE(soc.nu() == 2.0);

    constexpr Vectord<10> e{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0};
    STATIC_REQUIRE(soc.e() == e);
    STATIC_REQUIRE(L2<Slack>()(soc.w(soc.x0(), soc.y0()) - soc.e()) < eps);

    constexpr SOCCirc<Slack> circ;
    constexpr auto x = vec.fill(5.5);
    STATIC_REQUIRE(circ(x, soc.e()) == x);
    STATIC_REQUIRE(soc.circ(x, soc.e()) == x);
    STATIC_REQUIRE(soc.circ(soc.e(), x) == x);
  }
}

TEST_CASE("Test RSOC", "[cone_test]") {
  SECTION("Test rsoc cone api") {
    using Slack = Vectord<4>;
    constexpr RotatedSecondOrderCone<Slack> rsoc;
    STATIC_REQUIRE(rsoc.in_cone(rsoc.x0()));
    STATIC_REQUIRE(rsoc.in_cone(rsoc.y0()));

    constexpr vec_traits<Slack> vec;
    constexpr auto p = vec.fill(0.5);
    constexpr auto x_soc = rsoc.map(rsoc.x0());
    constexpr auto y_soc = rsoc.dual_map(rsoc.y0());
    STATIC_REQUIRE(rsoc.soc.in_cone(x_soc));
    STATIC_REQUIRE(rsoc.soc.in_cone(y_soc));

    constexpr auto p_soc = rsoc.map(p);
    constexpr auto p_soc_dual = rsoc.dual_map(p);
    STATIC_REQUIRE(rsoc.alpha_max(rsoc.x0(), p) ==
                   rsoc.soc.alpha_max(x_soc, p_soc));
    STATIC_REQUIRE(rsoc.alpha_max_dual(rsoc.y0(), p) ==
                   rsoc.soc.alpha_max_dual(y_soc, p_soc_dual));
    STATIC_REQUIRE(rsoc.alpha_max(rsoc.x0(), p) ==
                   rsoc.alpha_max_dual(rsoc.y0(), p));

    constexpr auto H_rsoc = rsoc.H(rsoc.x0(), rsoc.y0());
    constexpr auto H_soc = rsoc.soc.H(x_soc, y_soc);
    constexpr auto A = rsoc.map.linear_part();
    STATIC_REQUIRE(H_rsoc.apply(Slack{}) == Slack{});
    constexpr auto q = Slack{0.1, -0.2, 0.3, -0.4};
    constexpr auto H_pull_q =
        linalg::transpose(A) * H_soc.apply(A * q);
    STATIC_REQUIRE(L2<Slack>()(H_rsoc.apply(q) - H_pull_q) < eps);

    STATIC_REQUIRE(rsoc.nu() == 2.0);

    constexpr auto e_rsoc = rsoc.e();
    constexpr auto e_soc = rsoc.soc.e();
    STATIC_REQUIRE(L2<Slack>()(rsoc.map(e_rsoc) - e_soc) < eps);

    constexpr auto circ_rsoc = rsoc.circ(rsoc.x0(), rsoc.y0());
    constexpr auto circ_soc = rsoc.soc.circ(x_soc, y_soc);
    STATIC_REQUIRE(L2<Slack>()(rsoc.map(circ_rsoc) - circ_soc) < eps);

    constexpr auto w_rsoc = rsoc.w(rsoc.x0(), rsoc.y0());
    constexpr auto w_soc = rsoc.soc.w(x_soc, y_soc);
    STATIC_REQUIRE(L2<Slack>()(rsoc.map(w_rsoc) - w_soc) < eps);
  }
}
