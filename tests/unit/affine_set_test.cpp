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

TEST_CASE("Test affine set", "[affine sets]") {
  constexpr auto I = make_identity<double, 3>();
  SECTION("Implicit affine set creation check", "[Ax = b]") {
    constexpr auto b = Vector<double, 3>{1.0, 2.0, 3.0};
    constexpr auto set = ImplicitAffineSet<double, 3, 3>(I, b);
    STATIC_REQUIRE(all_equal(set.basis(), I));
    STATIC_REQUIRE(all_equal(as_col_vector(set.point()), as_col_vector(b)));
  }

  SECTION("Parametric affine set creation check", "[x0 + U t]") {
    constexpr auto x0 = Vector<double, 3>{0.5, 0.5, 0.5};
    constexpr auto set = ParametricAffineSet<double, 3, 3>(I, x0);
    STATIC_REQUIRE(all_equal(set.basis(), I));
    STATIC_REQUIRE(all_equal(as_col_vector(set.point()), as_col_vector(x0)));
  }
}
