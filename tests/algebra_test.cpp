/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#include <algorithm>
#include <catch2/catch_test_macros.hpp>

#include "math.hpp"

using namespace algebra;

TEST_CASE("Test field types", "[field]") {
  SECTION("Floating point check") {
    STATIC_REQUIRE(FloatingPoint<double>);
    STATIC_REQUIRE(FloatingPoint<std::array<double, 1>>);
    STATIC_REQUIRE(FloatingPoint<topology::OpenSet<double, 5>>);
    STATIC_REQUIRE(
        FloatingPoint<topology::Topology<topology::OpenSet<double, 3>, 2>>);
  }

  SECTION("Field check") {
    STATIC_REQUIRE(FieldLike<double>);
    STATIC_REQUIRE(FieldLike<float>);
    STATIC_REQUIRE(FieldLike<std::complex<double>>);
    STATIC_REQUIRE(!FieldLike<int>);
    STATIC_REQUIRE(!FieldLike<unsigned>);
  }
}
