/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <iostream>

#include "math.hpp"
#include "unit_test_helpers.hpp"

using namespace linalg;
using namespace convex;

TEST_CASE("Test convex set", "[convex_sets]") {
  SECTION("Quick test to check convex sets types") {
    struct Null;
    STATIC_REQUIRE(SetLike<ConvexSet<Matrix<double, 2, 2>, Null>>);
  }
}
