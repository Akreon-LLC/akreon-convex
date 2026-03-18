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
#include <random>

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
}

TEST_CASE("Test R^n_+ cone", "[cone_test]") {
  SECTION("Test rplus cone api") {
    using Slack = Vectord<12>;
    constexpr NonnegOrthantCone<Slack> rplus;
    STATIC_REQUIRE(rplus.in_cone(rplus.x0()));
    STATIC_REQUIRE(rplus.in_dual_cone(rplus.y0()));

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

  SECTION("Stress rplus w/H on random interior points") {
    using Slack = Vectord<8>;
    constexpr NonnegOrthantCone<Slack> rplus;
    constexpr auto zero = Slack{};
    constexpr auto l2 = L2<Slack>();

    auto make_rplus_point = [](std::mt19937& rng) {
      std::uniform_real_distribution<double> coord_dist(0.05, 3.0);
      Slack x{};
      for (size_t i = 0; i < 8; ++i) x(i) = coord_dist(rng);
      return x;
    };

    std::mt19937 rng(0xBEEF);
    constexpr size_t trials = 512;
    constexpr double tol = 1e-5;
    constexpr double eta = 0.99;
    constexpr double boundary_eps = 1e-8;

    for (size_t i = 0; i < trials; ++i) {
      const auto x = make_rplus_point(rng);
      const auto y = make_rplus_point(rng);
      const auto p = (-2.0) * x;

      REQUIRE(rplus.in_cone(x));
      REQUIRE(rplus.in_dual_cone(y));

      const auto w = rplus.w(x, y);
      REQUIRE(rplus.in_cone(w));
      REQUIRE(rplus.in_dual_cone(w));

      const auto Hw = rplus.H(x, y);
      REQUIRE(l2(Hw.apply(zero)) < tol);

      QuadraticRep<Matrixd<8, 8>, RplusCirc<Slack>> P(w);
      REQUIRE(l2(x - P.apply(y)) < eps);
      Slack w_inv{};
      for (size_t j = 0; j < 8; ++j) w_inv(j) = 1.0 / w(j);
      REQUIRE(l2(rplus.circ(w, w_inv) - rplus.e()) < eps);

      QuadraticRep<Matrixd<8, 8>, RplusCirc<Slack>> P_inv(w_inv);
      REQUIRE(l2(y - P_inv.apply(x)) < eps);

      const auto hx = Hw.apply(x);
      REQUIRE(l2(hx - y) <= eps);

      const auto alpha = rplus.alpha_max(x, p);
      const auto alpha_hit = alpha / eta;
      REQUIRE(constexpr_abs(alpha_hit - 0.5) < tol);
      REQUIRE(rplus.in_cone(x + alpha * p));
      REQUIRE(!rplus.in_cone(x + (alpha_hit + boundary_eps) * p));
    }
  }
}

TEST_CASE("Test SOC", "[cone_test]") {
  SECTION("Test soc cone api") {
    using Slack = Vectord<10>;
    constexpr SecondOrderCone<Slack> soc;
    STATIC_REQUIRE(soc.in_cone(soc.x0()));
    STATIC_REQUIRE(soc.in_dual_cone(soc.y0()));

    constexpr vec_traits<Slack> vec;
    constexpr auto p = vec.fill(0.5);
    STATIC_REQUIRE(linalg::constexpr_abs(soc.alpha_max(soc.x0(), p) - 0.99) <
                   eps);
    STATIC_REQUIRE(soc.alpha_max(soc.x0(), p) ==
                   soc.alpha_max_dual(soc.y0(), p));

    constexpr auto P = soc.H(soc.x0(), soc.y0());
    STATIC_REQUIRE(P.apply(Slack{}) == Slack{});
    constexpr auto q =
        Slack{0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, 0.2};
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

  SECTION("Stress soc w/H on random interior points") {
    using Slack = Vectord<4>;
    constexpr SecondOrderCone<Slack> soc;
    constexpr auto zero = Slack{};
    constexpr auto l2 = L2<Slack>();

    auto make_soc_point = [](std::mt19937& rng) {
      std::uniform_real_distribution<double> coord_dist(-2.0, 2.0);
      std::uniform_real_distribution<double> margin_dist(0.05, 2.0);

      Slack x{};
      for (size_t i = 0; i < 3; ++i) x(i) = coord_dist(rng);

      Vectord<3> v{x(0), x(1), x(2)};
      x(3) = L2<Vectord<3>>()(v) + margin_dist(rng);
      return x;
    };

    std::mt19937 rng(0xC0FFEE);
    constexpr size_t trials = 512;
    constexpr double tol = 1e-5;
    constexpr double eta = 0.99;
    constexpr double boundary_eps = 1e-8;

    for (size_t i = 0; i < trials; ++i) {
      const auto x = make_soc_point(rng);
      const auto y = make_soc_point(rng);
      const auto p = (-2.0) * x;

      REQUIRE(soc.in_cone(x));
      REQUIRE(soc.in_dual_cone(y));

      const auto w = soc.w(x, y);
      REQUIRE(soc.in_cone(w));
      REQUIRE(soc.in_dual_cone(w));

      REQUIRE(l2(x - soc.inv(soc.inv(x))) < eps);
      REQUIRE(l2(x - soc.circ(soc.sqrt(x), soc.sqrt(x))) < eps);
      REQUIRE(l2(soc.circ(x, soc.inv(x)) - soc.e()) < eps);

      const auto x_eigs = soc.eigs(x);
      const auto det = std::get<0>(x_eigs) * std::get<1>(x_eigs);
      const auto xv = std::get<0>(splice<3>(x));
      REQUIRE(constexpr_abs(det - (x(3) * x(3) - dot(xv, xv))) < eps);

      const auto Hw = soc.H(x, y);
      REQUIRE(l2(Hw.apply(zero)) < tol);

      QuadraticRep<Matrixd<4, 4>, SOCCirc<Slack>> P(w);
      REQUIRE(l2(x - P.apply(y)) < eps);

      QuadraticRep<Matrixd<4, 4>, SOCCirc<Slack>> P_inv(soc.inv(w));
      REQUIRE(l2(y - P_inv.apply(x)) < eps);

      const auto hx = Hw.apply(x);
      REQUIRE(l2(hx - y) <= eps);

      const auto alpha = soc.alpha_max(x, p);
      const auto alpha_hit = alpha / eta;
      REQUIRE(constexpr_abs(alpha_hit - 0.5) < tol);
      REQUIRE(soc.in_cone(x + alpha * p));
      REQUIRE(!soc.in_cone(x + (alpha_hit + boundary_eps) * p));
    }
  }
}

TEST_CASE("Test PSD Cone", "[cone_test]") {
  SECTION("Test psd cone api") {
    using Slack = Vectord<3>;
    constexpr PositiveSemidefiniteCone<Slack, 2> psd;
    STATIC_REQUIRE(psd.in_cone(psd.x0()));
    STATIC_REQUIRE(psd.in_dual_cone(psd.y0()));

    constexpr auto x = Slack{2.0, 3.0, 5.0};
    constexpr auto y = Slack{7.0, 0.0, 11.0};
    constexpr auto expected = Slack{14.0, 27.0, 55.0};

    STATIC_REQUIRE(psd.circ(x, y) == expected);
    STATIC_REQUIRE(psd.circ(x, psd.e()) == x);
    STATIC_REQUIRE(psd.circ(psd.e(), x) == x);

    constexpr auto z = Slack{4.0, 2.0, 3.0};
    constexpr auto z_inv = Slack{0.375, -0.25, 0.5};
    STATIC_REQUIRE(psd.in_cone(psd.inv(z)));
    STATIC_REQUIRE(all_equal(psd.inv(z), z_inv));
    STATIC_REQUIRE(all_equal(psd.circ(z, psd.inv(z)), psd.e()));

    constexpr auto gx = Slack{4.0, 0.0, 9.0};
    constexpr auto gy = Slack{1.0, 0.0, 16.0};
    constexpr auto expected_gamma =
        -1.0 * constexpr_ln_safe(36.0) - 1.0 * constexpr_ln_safe(16.0) +
        psd.nu() * constexpr_ln_safe(148.0);
    REQUIRE(constexpr_abs(psd.gamma_F(gx, gy) - expected_gamma) < 1e-5);
  }
}
