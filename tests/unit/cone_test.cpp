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
#include "test_helpers.hpp"

using namespace linalg;
using namespace convex;
using namespace affine;
using namespace conic;
using namespace unit_test;

static constexpr bool ok_psd_event(const linalg::NumericalEvent& event) {
  return event == linalg::NumericalEvent::NONE ||
         event == linalg::NumericalEvent::NUMERICAL_SAFEGUARD;
}

TEST_CASE("Exponential and power cone initial points belong to their cones",
          "[cone_test]") {
  SECTION("ExpCone initial points support single and product cones") {
    using Slack = Vectord<3>;
    constexpr ExpCone<Slack> exp_cone;
    STATIC_REQUIRE(exp_cone.in_cone(exp_cone.x0()));

    constexpr ExpCone<Vectord<9>> exp_cone9;
    STATIC_REQUIRE(exp_cone9.in_cone(exp_cone9.x0()));
  }

  SECTION("PowerCone initial point satisfies equal weights") {
    using Slack = Vectord<4>;
    constexpr vector_traits<Vectord<3>> vec_t;
    constexpr PowerCone<Slack> pcone(vec_t.fill(1.0 / 3.0));
    STATIC_REQUIRE(pcone.in_cone(pcone.x0()));
  }
}

TEST_CASE("Nonnegative cone scaling and numerical safeguards", "[cone_test]") {
  SECTION("Affine correction reports denominator safeguards") {
    using Slack = Vectord<4>;
    constexpr NonnegOrthantCone<Slack> rplus;
    constexpr auto scaling = std::get<0>(rplus.H(rplus.e(), rplus.e()));
    constexpr Slack px{2.0, 1.0, -1.0, 0.0};
    constexpr Slack py{3.0, -2.0, 2.0, 1.0};
    constexpr auto interior = rplus.scaled_affine_correction(
        Slack{2.0, 4.0, 1.0, rplus.eps}, scaling, px, py);
    STATIC_REQUIRE(std::get<1>(interior) == NumericalEvent::NONE);
    STATIC_REQUIRE(std::get<0>(interior) == Slack{3.0, -0.5, -2.0, 0.0});

    // Zero, positive below the floor, and negative denominators all clamp.
    // A safeguard is reported even when the corresponding numerator is zero.
    constexpr auto clamped = rplus.scaled_affine_correction(
        Slack{2.0, 0.0, rplus.eps / 2.0, -1.0}, scaling, px, py);
    STATIC_REQUIRE(std::get<1>(clamped) == NumericalEvent::NUMERICAL_SAFEGUARD);
    STATIC_REQUIRE(std::get<0>(clamped) ==
                   Slack{3.0, -2.0 / rplus.eps, -2.0 / rplus.eps, 0.0});
  }

  SECTION("Test rplus cone api") {
    using Slack = Vectord<12>;
    constexpr NonnegOrthantCone<Slack> rplus;
    STATIC_REQUIRE(rplus.in_cone(rplus.x0()));
    STATIC_REQUIRE(rplus.in_dual_cone(rplus.y0()));

    constexpr vector_traits<Slack> vec;
    constexpr auto p = vec.fill(2.0);
    STATIC_REQUIRE(
        linalg::constexpr_abs(rplus.alpha_max(rplus.x0(), p) - 0.99) < eps);

    STATIC_REQUIRE(rplus.alpha_max(rplus.x0(), p) ==
                   rplus.alpha_max_dual(rplus.y0(), p));

    constexpr auto w_res = rplus.w(rplus.x0(), rplus.y0());
    constexpr auto w = std::get<0>(w_res);
    constexpr auto w_event = std::get<1>(w_res);
    constexpr auto ones = vec.fill(1.0);
    STATIC_REQUIRE(w_event == linalg::NumericalEvent::NONE);
    STATIC_REQUIRE(L2<Slack>()(w - ones) < eps);

    constexpr auto H_res = rplus.H(rplus.x0(), rplus.y0());
    constexpr auto P = std::get<0>(H_res);
    constexpr auto H_event = std::get<1>(H_res);
    STATIC_REQUIRE(H_event == linalg::NumericalEvent::NONE);
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
      for (std::size_t i = 0; i < 8; ++i) x(i) = coord_dist(rng);
      return x;
    };

    std::mt19937 rng(0xBEEF);
    constexpr std::size_t trials = 512;
    constexpr double tol = 1e-5;
    constexpr double eta = 0.99;
    constexpr double boundary_eps = 1e-8;

    for (std::size_t i = 0; i < trials; ++i) {
      const auto x = make_rplus_point(rng);
      const auto y = make_rplus_point(rng);
      const auto p = (-2.0) * x;

      REQUIRE(rplus.in_cone(x));
      REQUIRE(rplus.in_dual_cone(y));

      const auto [w, w_event] = rplus.w(x, y);
      REQUIRE(w_event == linalg::NumericalEvent::NONE);
      REQUIRE(rplus.in_cone(w));
      REQUIRE(rplus.in_dual_cone(w));

      const auto [Hw, H_event] = rplus.H(x, y);
      REQUIRE(H_event == linalg::NumericalEvent::NONE);
      REQUIRE(l2(Hw.apply(zero)) < tol);

      QuadraticRep<Matrixd<8, 8>, RplusCirc<Slack>> P(w);
      REQUIRE(l2(x - P.apply(y)) < eps);
      Slack w_inv{};
      for (std::size_t j = 0; j < 8; ++j) w_inv(j) = 1.0 / w(j);
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

TEST_CASE("Second-order cone scaling and interior-point identities",
          "[cone_test]") {
  SECTION("Test soc cone api") {
    using Slack = Vectord<10>;
    constexpr SecondOrderCone<Slack> soc;
    STATIC_REQUIRE(soc.in_cone(soc.x0()));
    STATIC_REQUIRE(soc.in_dual_cone(soc.y0()));

    constexpr vector_traits<Slack> vec;
    constexpr auto p = vec.fill(0.5);
    STATIC_REQUIRE(linalg::constexpr_abs(soc.alpha_max(soc.x0(), p) - 0.99) <
                   eps);
    STATIC_REQUIRE(soc.alpha_max(soc.x0(), p) ==
                   soc.alpha_max_dual(soc.y0(), p));

    constexpr auto H_res = soc.H(soc.x0(), soc.y0());
    constexpr auto P = std::get<0>(H_res);
    constexpr auto H_event = std::get<1>(H_res);
    STATIC_REQUIRE(H_event == linalg::NumericalEvent::NONE);
    STATIC_REQUIRE(P.apply(Slack{}) == Slack{});
    constexpr auto q =
        Slack{0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, 0.2};
    STATIC_REQUIRE(L2<Slack>()(P.apply(q) - q) < eps);

    STATIC_REQUIRE(soc.nu() == 2.0);
    STATIC_REQUIRE(constexpr_abs(std::get<0>(soc.gamma_F(soc.e(), soc.e()))) <
                   eps);

    constexpr Vectord<10> e{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0};
    STATIC_REQUIRE(soc.e() == e);
    constexpr auto soc_w_res = soc.w(soc.x0(), soc.y0());
    constexpr auto soc_w = std::get<0>(soc_w_res);
    constexpr auto soc_w_event = std::get<1>(soc_w_res);
    STATIC_REQUIRE(soc_w_event == linalg::NumericalEvent::NONE);
    STATIC_REQUIRE(L2<Slack>()(soc_w - soc.e()) < eps);

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
      for (std::size_t i = 0; i < 3; ++i) x(i) = coord_dist(rng);

      Vectord<3> v{x(0), x(1), x(2)};
      x(3) = L2<Vectord<3>>()(v) + margin_dist(rng);
      return x;
    };

    std::mt19937 rng(0xC0FFEE);
    constexpr std::size_t trials = 512;
    constexpr double tol = 1e-5;
    constexpr double eta = 0.99;
    constexpr double boundary_eps = 1e-8;

    for (std::size_t i = 0; i < trials; ++i) {
      const auto x = make_soc_point(rng);
      const auto y = make_soc_point(rng);
      const auto p = (-2.0) * x;

      REQUIRE(soc.in_cone(x));
      REQUIRE(soc.in_dual_cone(y));

      const auto [w, w_event] = soc.w(x, y);
      REQUIRE(w_event == linalg::NumericalEvent::NONE);
      REQUIRE(soc.in_cone(w));
      REQUIRE(soc.in_dual_cone(w));

      const auto [x_inv, x_inv_event] = soc.inv(x);
      REQUIRE(x_inv_event == linalg::NumericalEvent::NONE);
      const auto [x_inv_inv, x_inv_inv_event] = soc.inv(x_inv);
      REQUIRE(x_inv_inv_event == linalg::NumericalEvent::NONE);
      REQUIRE(l2(x - x_inv_inv) < eps);
      REQUIRE(l2(x - soc.circ(soc.sqrt(x), soc.sqrt(x))) < eps);
      REQUIRE(l2(soc.circ(x, x_inv) - soc.e()) < eps);

      const auto x_eigs = soc.eigs(x);
      const auto det = std::get<0>(x_eigs) * std::get<1>(x_eigs);
      const auto xv = std::get<0>(splice<3>(x));
      REQUIRE(constexpr_abs(det - (x(3) * x(3) - inner_product(xv, xv))) < eps);

      const auto [Hw, H_event] = soc.H(x, y);
      REQUIRE(H_event == linalg::NumericalEvent::NONE);
      REQUIRE(l2(Hw.apply(zero)) < tol);

      QuadraticRep<Matrixd<4, 4>, SOCCirc<Slack>> P(w);
      REQUIRE(l2(x - P.apply(y)) < eps);

      const auto [w_inv, w_inv_event] = soc.inv(w);
      REQUIRE(w_inv_event == linalg::NumericalEvent::NONE);
      QuadraticRep<Matrixd<4, 4>, SOCCirc<Slack>> P_inv(w_inv);
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

TEST_CASE("PSD cone scaling and interior-point identities", "[cone_test]") {
  SECTION("Test psd cone api") {
    using Slack = Vectord<3>;
    constexpr PositiveSemidefiniteCone<Slack, 2> psd;
    STATIC_REQUIRE(psd.nu() == 2.0);
    STATIC_REQUIRE(constexpr_abs(std::get<0>(psd.gamma_F(psd.e(), psd.e()))) <
                   eps);
    STATIC_REQUIRE(psd.in_cone(psd.x0()));
    STATIC_REQUIRE(psd.in_dual_cone(psd.y0()));

    constexpr auto x = Slack{2.0, 3.0, 5.0};
    constexpr auto y = Slack{7.0, 0.0, 11.0};
    constexpr auto expected = Slack{14.0, 27.0, 55.0};

    STATIC_REQUIRE(all_equal(psd.circ(x, y), expected));
    STATIC_REQUIRE(all_equal(psd.circ(x, psd.e()), x));
    STATIC_REQUIRE(all_equal(psd.circ(psd.e(), x), x));

    constexpr auto z = Slack{4.0, 2.0, 3.0};
    constexpr auto z_inv = Slack{0.3, -0.2, 0.4};
    constexpr auto z_inv_res = psd.inv(z);
    constexpr auto z_inv_calc = std::get<0>(z_inv_res);
    constexpr auto z_inv_event = std::get<1>(z_inv_res);
    STATIC_REQUIRE(z_inv_event == linalg::NumericalEvent::NONE);
    STATIC_REQUIRE(psd.in_cone(z_inv_calc));
    STATIC_REQUIRE(all_equal(z_inv_calc, z_inv));
    STATIC_REQUIRE(all_equal(make_smat<2>(z) * make_smat<2>(z_inv_calc),
                             make_identity<double, 2>(), 1e-5));
    STATIC_REQUIRE(all_equal(psd.circ(z, z_inv_calc), psd.e()));

    constexpr auto gx = Slack{4.0, 0.0, 9.0};
    constexpr auto gy = Slack{1.0, 0.0, 16.0};
    constexpr auto expected_gamma =
        -1.0 * constexpr_ln_safe(36.0) - 1.0 * constexpr_ln_safe(16.0) +
        psd.nu() * constexpr_ln_safe(148.0 / psd.nu());
    constexpr auto gamma_res = psd.gamma_F(gx, gy);
    constexpr auto gamma = std::get<0>(gamma_res);
    constexpr auto event = std::get<1>(gamma_res);
    STATIC_REQUIRE(event == linalg::NumericalEvent::NONE);
    REQUIRE(constexpr_abs(gamma - expected_gamma) < 1e-5);

    constexpr auto gw = Slack{2.0, 0.0, 0.75};
    constexpr auto w_res_psd = psd.w(gx, gy);
    constexpr auto w_psd = std::get<0>(w_res_psd);
    constexpr auto w_psd_event = std::get<1>(w_res_psd);
    STATIC_REQUIRE(ok_psd_event(w_psd_event));
    REQUIRE(psd.in_cone(w_psd));
    REQUIRE(L2<Slack>()(w_psd - gw) < 1e-5);
    const auto P = QuadraticRep<Matrixd<3, 3>, PSDCirc<Slack, 2>>(w_psd);
    REQUIRE(L2<Slack>()(P.apply(gy) - gx) < 1e-5);

    constexpr auto grad_x = Slack{-0.25, 0.0, -1.0 / 9.0};
    const auto [grad_x_calc, grad_event] = psd.grad(gx);
    REQUIRE(grad_event == linalg::NumericalEvent::NONE);
    REQUIRE(L2<Slack>()(grad_x_calc - grad_x) < 1e-5);

    const auto [Hw, H_event] = psd.H(gx, gy);
    REQUIRE(ok_psd_event(H_event));
    REQUIRE(Hw.apply(Slack{}) == Slack{});
    REQUIRE(L2<Slack>()(Hw.apply(gx) - gy) < 1e-5);

    constexpr auto q = Slack{8.0, 3.0, 9.0};
    constexpr auto Hq = Slack{2.0, 2.0, 16.0};
    REQUIRE(L2<Slack>()(Hw.apply(q) - Hq) < 1e-5);

    constexpr auto p_neg = Slack{-8.0, 0.0, -18.0};
    constexpr auto p_neg_dual = Slack{-2.0, 0.0, -32.0};
    constexpr auto p_pos = Slack{1.0, 0.0, 1.0};
    REQUIRE(constexpr_abs(psd.alpha_max(gx, p_neg) - 0.495) < 1e-5);
    REQUIRE(constexpr_abs(psd.alpha_max_dual(gy, p_neg_dual) - 0.495) < 1e-5);
    REQUIRE(constexpr_abs(psd.alpha_max(gx, p_pos) - 0.99) < 1e-5);
  }

  SECTION("Stress psd w/H on random interior points") {
    using Slack = Vectord<6>;
    constexpr PositiveSemidefiniteCone<Slack, 3> psd;
    constexpr auto zero = Slack{};
    constexpr auto l2 = L2<Slack>();

    auto make_psd_point = [](std::mt19937& rng) {
      std::uniform_real_distribution<double> diag_dist(0.5, 3.0);
      return Slack{diag_dist(rng), 0.0, diag_dist(rng), 0.0, 0.0,
                   diag_dist(rng)};
    };

    std::mt19937 rng(0xFACE);
    constexpr std::size_t trials = 256;
    constexpr double tol = 1e-5;
    constexpr double eta = 0.99;
    constexpr double boundary_eps = 1e-8;

    for (std::size_t i = 0; i < trials; ++i) {
      const auto x = make_psd_point(rng);
      const auto y = make_psd_point(rng);
      const auto p = (-2.0) * x;

      REQUIRE(psd.in_cone(x));
      REQUIRE(psd.in_dual_cone(y));

      const auto [x_inv, x_inv_event] = psd.inv(x);
      REQUIRE(x_inv_event == linalg::NumericalEvent::NONE);
      REQUIRE(psd.in_cone(x_inv));
      REQUIRE(all_equal(make_smat<3>(x) * make_smat<3>(x_inv),
                        make_identity<double, 3>(), tol));
      REQUIRE(all_equal(make_smat<3>(x_inv) * make_smat<3>(x),
                        make_identity<double, 3>(), tol));

      const auto [w, w_event] = psd.w(x, y);
      REQUIRE(ok_psd_event(w_event));
      REQUIRE(psd.in_cone(w));
      REQUIRE(psd.in_dual_cone(w));

      const auto [Hw, H_event] = psd.H(x, y);
      REQUIRE(ok_psd_event(H_event));
      REQUIRE(l2(Hw.apply(zero)) < tol);

      QuadraticRep<Matrixd<6, 6>, PSDCirc<Slack, 3>> P(w);
      REQUIRE(l2(x - P.apply(y)) < tol);

      const auto [w_inv, w_inv_event] = psd.inv(w);
      REQUIRE(w_inv_event == linalg::NumericalEvent::NONE);
      QuadraticRep<Matrixd<6, 6>, PSDCirc<Slack, 3>> P_inv(w_inv);
      REQUIRE(l2(y - P_inv.apply(x)) < tol);

      const auto hx = Hw.apply(x);
      REQUIRE(l2(hx - y) < tol);

      const auto alpha = psd.alpha_max(x, p);
      const auto alpha_hit = alpha / eta;
      REQUIRE(constexpr_abs(alpha_hit - 0.5) < tol);
      REQUIRE(psd.in_cone(x + alpha * p));
      REQUIRE(!psd.in_cone(x + (alpha_hit + boundary_eps) * p));
    }
  }

  SECTION("Near-boundary PSD alpha and functional proximity") {
    using Slack = Vectord<6>;
    constexpr PositiveSemidefiniteCone<Slack, 3> psd;
    constexpr double tol = 1e-8;
    constexpr double boundary_eps = 1e-9;
    constexpr double eta = 0.99;

    const auto X_int = make_identity<double, 3>();
    const auto X_near = Matrixd<3, 3>{
        Vectord<3>{1e-6, 0.0, 0.0},
        Vectord<3>{0.0, 1.0, 0.0},
        Vectord<3>{0.0, 0.0, 1.0},
    };
    const auto X_out = Matrixd<3, 3>{
        Vectord<3>{-1e-6, 0.0, 0.0},
        Vectord<3>{0.0, 1.0, 0.0},
        Vectord<3>{0.0, 0.0, 1.0},
    };

    const auto x_int = get_svec(X_int);
    const auto x_near = get_svec(X_near);
    const auto x_out = get_svec(X_out);
    const auto y_int = x_int;
    const auto p = get_svec(Matrixd<3, 3>{
        Vectord<3>{-1.0, 0.0, 0.0},
        Vectord<3>{0.0, 0.0, 0.0},
        Vectord<3>{0.0, 0.0, 0.0},
    });

    REQUIRE(psd.in_cone(x_int));
    REQUIRE(psd.in_cone(x_near));
    REQUIRE(!psd.in_cone(x_out));

    const auto alpha_int = psd.alpha_max(x_int, p);
    const auto alpha_near = psd.alpha_max(x_near, p);
    REQUIRE(alpha_near < alpha_int);
    REQUIRE(alpha_near < 1e-4 * alpha_int);

    const auto alpha_hit_near = alpha_near / eta;
    REQUIRE(psd.in_cone(x_near + alpha_near * p));
    REQUIRE(!psd.in_cone(x_near + (alpha_hit_near + boundary_eps) * p));

    const auto [gamma_int, gamma_int_event] = psd.gamma_F(x_int, y_int);
    const auto [gamma_near, gamma_near_event] = psd.gamma_F(x_near, y_int);
    REQUIRE(gamma_int_event == linalg::NumericalEvent::NONE);
    REQUIRE(gamma_near_event == linalg::NumericalEvent::NONE);
    REQUIRE(gamma_near > gamma_int + tol);
  }

  SECTION("Stress psd w/H on random SPD points with off-diagonals") {
    using Slack = Vectord<6>;
    constexpr PositiveSemidefiniteCone<Slack, 3> psd;
    constexpr auto zero = Slack{};
    constexpr auto l2 = L2<Slack>();

    auto make_psd_point = [](std::mt19937& rng) {
      std::uniform_real_distribution<double> diag_dist(0.75, 2.0);
      std::uniform_real_distribution<double> offdiag_dist(-0.2, 0.2);

      const double l00 = diag_dist(rng);
      const double l10 = offdiag_dist(rng);
      const double l11 = diag_dist(rng);
      const double l20 = offdiag_dist(rng);
      const double l21 = offdiag_dist(rng);
      const double l22 = diag_dist(rng);

      const auto L =
          Matrixd<3, 3>{Vectord<3>{l00, 0.0, 0.0}, Vectord<3>{l10, l11, 0.0},
                        Vectord<3>{l20, l21, l22}};
      const auto X = L * transpose(L);
      return get_svec(X);
    };

    std::mt19937 rng(0xBADA55);
    constexpr std::size_t trials = 128;
    constexpr double tol = 1e-5;
    constexpr double eta = 0.99;
    constexpr double boundary_eps = 1e-8;

    for (std::size_t i = 0; i < trials; ++i) {
      const auto x = make_psd_point(rng);
      const auto y = make_psd_point(rng);
      const auto p = (-2.0) * x;

      REQUIRE(psd.in_cone(x));
      REQUIRE(psd.in_dual_cone(y));

      const auto [x_inv, x_inv_event] = psd.inv(x);
      REQUIRE(x_inv_event == linalg::NumericalEvent::NONE);
      REQUIRE(psd.in_cone(x_inv));

      REQUIRE(l2(psd.circ(x, x_inv) - psd.e()) < tol);
      REQUIRE(l2(psd.circ(x_inv, x) - psd.e()) < tol);

      const auto [w, w_event] = psd.w(x, y);
      REQUIRE(ok_psd_event(w_event));
      REQUIRE(psd.in_cone(w));
      REQUIRE(psd.in_dual_cone(w));

      const auto W = make_smat<3>(w);
      const auto lhs = linalg::symmetric_part(W * (make_smat<3>(y) * W));
      REQUIRE(all_equal(lhs, make_smat<3>(x), tol));

      QuadraticRep<Matrixd<6, 6>, PSDCirc<Slack, 3>> P(w);
      REQUIRE(l2(x - P.apply(y)) < tol);

      const auto [w_inv, w_inv_event] = psd.inv(w);
      REQUIRE(w_inv_event == linalg::NumericalEvent::NONE);
      QuadraticRep<Matrixd<6, 6>, PSDCirc<Slack, 3>> P_inv(w_inv);
      REQUIRE(l2(y - P_inv.apply(x)) < tol);

      const auto [Hw, H_event] = psd.H(x, y);
      REQUIRE(ok_psd_event(H_event));
      REQUIRE(l2(Hw.apply(zero)) < tol);

      const auto hx = Hw.apply(x);
      REQUIRE(l2(hx - y) < tol);

      const auto alpha = psd.alpha_max(x, p);
      const auto alpha_hit = alpha / eta;
      REQUIRE(constexpr_abs(alpha_hit - 0.5) < tol);
      REQUIRE(psd.in_cone(x + alpha * p));
      REQUIRE(!psd.in_cone(x + (alpha_hit + boundary_eps) * p));
    }
  }

  SECTION("Stress 4x4 psd w/H on random SPD points with off-diagonals") {
    using Slack = Vectord<10>;
    constexpr PositiveSemidefiniteCone<Slack, 4> psd;
    constexpr auto zero = Slack{};
    constexpr auto l2 = L2<Slack>();

    auto make_psd_point = [](std::mt19937& rng) {
      std::uniform_real_distribution<double> diag_dist(0.75, 2.0);
      std::uniform_real_distribution<double> offdiag_dist(-0.2, 0.2);

      const double l00 = diag_dist(rng);
      const double l10 = offdiag_dist(rng);
      const double l11 = diag_dist(rng);
      const double l20 = offdiag_dist(rng);
      const double l21 = offdiag_dist(rng);
      const double l22 = diag_dist(rng);
      const double l30 = offdiag_dist(rng);
      const double l31 = offdiag_dist(rng);
      const double l32 = offdiag_dist(rng);
      const double l33 = diag_dist(rng);

      const auto L = Matrixd<4, 4>{
          Vectord<4>{l00, 0.0, 0.0, 0.0}, Vectord<4>{l10, l11, 0.0, 0.0},
          Vectord<4>{l20, l21, l22, 0.0}, Vectord<4>{l30, l31, l32, l33}};
      const auto X = L * transpose(L);
      return get_svec(X);
    };

    std::mt19937 rng(0xD00DFEED);
    constexpr std::size_t trials = 96;
    constexpr double tol = 1e-5;
    constexpr double eta = 0.99;
    constexpr double boundary_eps = 1e-8;

    for (std::size_t i = 0; i < trials; ++i) {
      const auto x = make_psd_point(rng);
      const auto y = make_psd_point(rng);
      const auto p = (-2.0) * x;

      REQUIRE(psd.in_cone(x));
      REQUIRE(psd.in_dual_cone(y));

      const auto [x_inv, x_inv_event] = psd.inv(x);
      REQUIRE(x_inv_event == linalg::NumericalEvent::NONE);
      REQUIRE(psd.in_cone(x_inv));

      REQUIRE(l2(psd.circ(x, x_inv) - psd.e()) < tol);
      REQUIRE(l2(psd.circ(x_inv, x) - psd.e()) < tol);

      const auto [w, w_event] = psd.w(x, y);
      REQUIRE(ok_psd_event(w_event));
      REQUIRE(psd.in_cone(w));
      REQUIRE(psd.in_dual_cone(w));

      const auto W = make_smat<4>(w);
      const auto lhs = linalg::symmetric_part(W * (make_smat<4>(y) * W));
      REQUIRE(all_equal(lhs, make_smat<4>(x), tol));

      QuadraticRep<Matrixd<10, 10>, PSDCirc<Slack, 4>> P(w);
      REQUIRE(l2(x - P.apply(y)) < tol);

      const auto [w_inv, w_inv_event] = psd.inv(w);
      REQUIRE(w_inv_event == linalg::NumericalEvent::NONE);
      QuadraticRep<Matrixd<10, 10>, PSDCirc<Slack, 4>> P_inv(w_inv);
      REQUIRE(l2(y - P_inv.apply(x)) < tol);

      const auto [Hw, H_event] = psd.H(x, y);
      REQUIRE(ok_psd_event(H_event));
      REQUIRE(l2(Hw.apply(zero)) < tol);

      const auto hx = Hw.apply(x);
      REQUIRE(l2(hx - y) < tol);

      const auto alpha = psd.alpha_max(x, p);
      const auto alpha_hit = alpha / eta;
      REQUIRE(constexpr_abs(alpha_hit - 0.5) < tol);
      REQUIRE(psd.in_cone(x + alpha * p));
      REQUIRE(!psd.in_cone(x + (alpha_hit + boundary_eps) * p));
    }
  }
}
