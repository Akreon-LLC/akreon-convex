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

TEST_CASE("Test Basic KKT System", "[kkt_system]") {
  constexpr auto H_mat = mat_traits<Matrixd<3, 3>>();
  constexpr auto H = H_mat(4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0);
  constexpr auto A_mat = mat_traits<Matrixd<2, 3>>();
  constexpr auto A = A_mat(1.0, 2.0, 0.0, 0.0, 1.0, 1.0);
  constexpr auto KKT_builder = KKTBuilder<decltype(H), decltype(A)>();
  constexpr auto b = Vectord<5>{1.0, -1.0, 1.0, -1.0, 1.0};
  constexpr auto KKT = KKT_builder.make(H, A);
  SECTION("Basic KKT") {
    constexpr auto ldl = LDL(KKT);
    STATIC_REQUIRE(ldl.is_valid());
    STATIC_REQUIRE(ldl.is_ind());

    constexpr auto res = ldl.L() * (ldl.D() * transpose(ldl.L()));
    STATIC_REQUIRE(KKT == res);

    constexpr auto x = ldl.solve(b);
    constexpr auto y = KKT * x;
    STATIC_REQUIRE(L2<decltype(b)>()(y - b) < eps);
  }

  SECTION("Basic KKT Schur") {
    constexpr auto H_inv = inverse(H);
    constexpr Hessian<decltype(H_inv), DifferentialType::PULLBACK> Hes(H_inv);
    constexpr auto x = KKT_builder.solve_schur(Hes, A, b);
    constexpr auto y = KKT * x;
    STATIC_REQUIRE(L2<decltype(b)>()(y - b) < eps);
  }
}

TEST_CASE("Test Near Singular KKT", "[kkt_system]") {
  SECTION("Near singular KKT") {
    /// (dakre) near singular -> bad pivot
    constexpr auto H_mat = mat_traits<Matrixd<2, 2>>();
    constexpr auto H = H_mat(1e-12, 0.0, 0.0, 1.0);
    constexpr auto A = make_identity<double, 2>();
    constexpr auto KKT_builder = KKTBuilder<decltype(H), decltype(A)>();
    constexpr auto KKT = KKT_builder.make(H, A);
    constexpr auto ldl = LDL(KKT);
    STATIC_REQUIRE(!ldl.is_valid());

    /// (dakre) applying tikhonov regularization to get factorization & solution
    constexpr auto KKT_reg = KKT_builder.regularize(KKT, 1e-8, -1e-8);
    constexpr auto ldl_reg = LDL(KKT_reg);
    STATIC_REQUIRE(ldl_reg.is_valid());
    STATIC_REQUIRE(ldl_reg.is_ind());

    constexpr auto res = ldl_reg.L() * (ldl_reg.D() * transpose(ldl_reg.L()));
    STATIC_REQUIRE(all_equal(KKT_reg, res));

    constexpr auto b = Vectord<4>{1.0, 1.0, 1.0, 1.0};
    constexpr auto x = ldl_reg.solve(b);
    constexpr auto y = KKT_reg * x;
    STATIC_REQUIRE(L2<decltype(b)>()(y - b) < eps);
  }
}
