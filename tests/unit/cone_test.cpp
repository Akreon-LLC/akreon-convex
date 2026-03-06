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

TEST_CASE("Test RSOC", "[cone_test]") {
  SECTION("Test rsoc cone api") {
    using Slack = Vectord<4>;
    constexpr RotatedSecondOrderCone<Slack> rsoc;
    STATIC_REQUIRE(rsoc.in_cone(rsoc.x0()));
    STATIC_REQUIRE(rsoc.in_cone(rsoc.y0()));

    constexpr vec_traits<Slack> vec;
    constexpr auto p = vec.fill(0.5);
    constexpr auto x_soc = rsoc.map(rsoc.x0());
    constexpr auto y_soc = rsoc.map(rsoc.y0());
    STATIC_REQUIRE(rsoc.soc.in_cone(x_soc));
    STATIC_REQUIRE(rsoc.soc.in_cone(y_soc));

    constexpr auto p_soc = rsoc.map(p);
    STATIC_REQUIRE(rsoc.alpha_max(rsoc.x0(), p) ==
                   rsoc.soc.alpha_max(x_soc, p_soc));

    constexpr auto H_rsoc = rsoc.H(rsoc.x0(), rsoc.y0());
    constexpr auto H_soc = rsoc.soc.H(x_soc, y_soc);
    constexpr auto A = rsoc.map.linear_part();
    STATIC_REQUIRE(H_rsoc.apply(Slack{}) == Slack{});
    constexpr auto q = Slack{0.1, -0.2, 0.3, -0.4};
    constexpr auto H_pull_q = linalg::transpose(A) * H_soc.apply(A * q);
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

    constexpr auto lhs = rsoc.trace(rsoc.x0(), rsoc.y0());
    constexpr auto rhs = rsoc.soc.trace(x_soc, y_soc);
    STATIC_REQUIRE(constexpr_abs(lhs - rhs) < eps);

    STATIC_REQUIRE(L2<Slack>()(rsoc.circ(rsoc.x0(), rsoc.inv(rsoc.x0())) -
                               rsoc.e()) < eps);

    STATIC_REQUIRE(rsoc.trace(rsoc.circ(rsoc.x0(), rsoc.y0()), rsoc.e()) -
                       rsoc.trace(rsoc.x0(), rsoc.y0()) <
                   eps);
  }

  SECTION("Stress rsoc w/H on random interior points") {
    using Slack = Vectord<4>;
    constexpr RotatedSecondOrderCone<Slack> rsoc;
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

    auto make_rsoc_point = [&](std::mt19937& rng) {
      const auto x_soc = make_soc_point(rng);
      return rsoc.inv_map(x_soc);
    };

    std::mt19937 rng(0xA11CE);
    constexpr size_t trials = 512;
    constexpr double tol = 1e-5;
    constexpr double eta = 0.99;
    constexpr double boundary_eps = 1e-8;
    const auto A = rsoc.map.linear_part();
    const auto A_t = linalg::transpose(A);
    std::normal_distribution<double> N(0.0, 1.0);
    std::uniform_real_distribution<double> q_dist(-1.0, 1.0);

    for (size_t i = 0; i < trials; ++i) {
      const auto x = make_rsoc_point(rng);
      const auto y = make_rsoc_point(rng);
      const auto p = (-2.0) * x;

      REQUIRE(rsoc.in_cone(x));
      REQUIRE(rsoc.in_dual_cone(y));

      const auto x_soc = rsoc.map(x);
      const auto y_soc = rsoc.map(y);
      REQUIRE(rsoc.soc.in_cone(x_soc));
      REQUIRE(rsoc.soc.in_dual_cone(y_soc));

      const auto w = rsoc.w(x, y);
      REQUIRE(rsoc.in_cone(w));
      REQUIRE(rsoc.in_dual_cone(w));
      REQUIRE(l2(rsoc.map(w) - rsoc.soc.w(x_soc, y_soc)) < eps);

      REQUIRE(l2(x - rsoc.inv(rsoc.inv(x))) < eps);
      REQUIRE(l2(rsoc.circ(x, rsoc.inv(x)) - rsoc.e()) < eps);
      REQUIRE(l2(rsoc.map(rsoc.circ(x, y)) - rsoc.soc.circ(x_soc, y_soc)) <
              eps);
      REQUIRE(constexpr_abs(rsoc.trace(x, y) - rsoc.soc.trace(x_soc, y_soc)) <
              eps);

      const auto Hw = rsoc.H(x, y);
      REQUIRE(l2(Hw.apply(zero)) < tol);

      Slack q{};
      for (size_t j = 0; j < 4; ++j) q(j) = q_dist(rng);
      const auto H_soc = rsoc.soc.H(x_soc, y_soc);
      const auto H_pull_q = A_t * H_soc.apply(A * q);
      REQUIRE(l2(Hw.apply(q) - H_pull_q) < tol);

      const auto hx = Hw.apply(x);
      REQUIRE(l2(hx - y) <= eps);

      const auto alpha = rsoc.alpha_max(x, p);
      const auto alpha_hit = alpha / eta;
      REQUIRE(constexpr_abs(alpha_hit - 0.5) < tol);
      REQUIRE(rsoc.in_cone(x + alpha * p));
      REQUIRE(!rsoc.in_cone(x + (alpha_hit + boundary_eps) * p));

      Slack dx_soc{};
      for (size_t j = 0; j < 4; ++j) dx_soc(j) = N(rng);

      const auto dx = rsoc.inv_map(dx_soc);
      const auto alpha_rsoc = rsoc.alpha_max(x, dx);
      const auto alpha_soc = rsoc.soc.alpha_max(x_soc, dx_soc);

      REQUIRE(constexpr_abs(alpha_rsoc - alpha_soc) < eps);
    }
  }
}
