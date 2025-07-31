/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */


#include <algorithm>
#include <catch2/catch_test_macros.hpp>

#include "math.hpp"

using namespace algebra;

TEST_CASE("Test field types", "[field]") {
  STATIC_REQUIRE(FloatingPoint<double>);
  STATIC_REQUIRE(FloatingPoint<std::array<double, 1>>);
  STATIC_REQUIRE(FloatingPoint<topology::OpenSet<double, 5>>);
  STATIC_REQUIRE(FloatingPoint<topology::Topology<topology::OpenSet<double, 3>, 2>>);
}
