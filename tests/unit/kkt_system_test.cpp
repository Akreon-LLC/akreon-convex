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

TEST_CASE("Test KKT System", "[kkt_system]") {
  SECTION("Basic KKT") {
    constexpr auto H_mat = mat_traits<Matrixd<3, 3>>();
    constexpr auto H = H_mat(4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0);
    constexpr auto A_mat = mat_traits<Matrixd<2, 3>>();
    constexpr auto A = A_mat(1.0, 2.0, 0.0, 0.0, 1.0, 1.0);
    constexpr auto KKT_mat = kkt_traits<decltype(H), decltype(A)>();
    constexpr auto KKT = KKT_mat.make(H, A);
    constexpr auto ldl = LDL(KKT);
    STATIC_REQUIRE(ldl.is_valid());
    STATIC_REQUIRE(ldl.is_ind());

    constexpr auto res = ldl.L() * (ldl.D() * transpose(ldl.L()));
    STATIC_REQUIRE(KKT == res);

    constexpr auto b = Vectord<5>{1.0, -1.0, 1.0, -1.0, 1.0};
    constexpr auto x = ldl.solve(b);
    constexpr auto y = KKT * x;
    STATIC_REQUIRE(L2<decltype(b)>()(y - b) < eps);
  }
}
