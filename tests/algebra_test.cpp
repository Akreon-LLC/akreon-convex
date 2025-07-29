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
  STATIC_REQUIRE(FloatingPoint<std::vector<double>>);
  STATIC_REQUIRE(FloatingPoint<Eigen::Vector3d>);
  STATIC_REQUIRE(FloatingPoint<Eigen::Matrix3d>);
  STATIC_REQUIRE(FloatingPoint<std::vector<Eigen::Matrix3d>>);
}
