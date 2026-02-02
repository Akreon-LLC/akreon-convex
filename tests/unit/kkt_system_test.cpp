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
#if 0
  SECTION("Basic KKT") {
    auto I = make_identity<double, 3>();
    auto kkt_system = KKTSystem(I);
    REQUIRE(kkt_system.is_psd());
  }
  SECTION("Basic KKT") {
    auto KKT_mat = mat_traits<Matrixd<5, 5>>();
    auto KKT =
        KKT_mat(4.0, 1.0, 0.0, 1.0, 0.0, 1.0, 3.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0,
                0.0, 1.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0);
    auto kkt_system = KKTSystem(KKT);
    REQUIRE(kkt_system.is_psd());
  }
#endif
}
