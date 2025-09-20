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
  constexpr auto b = Vector<double, 3>{1.0, 2.0, 3.0};
  constexpr auto ones = Vector<double, 3>{1.0, 1.0, 1.0};
  constexpr auto zeros = Vector<double, 3>{0.0, 0.0, 0.0};

  constexpr auto I_map = AffineMap<double, 3, 3>(I, zeros);
  SECTION("Quick test on affine map properties") {
    STATIC_REQUIRE(all_equal(I_map.linear_part(), I));
    STATIC_REQUIRE(
        all_equal(as_col_vector(I_map.translation()), as_col_vector(zeros)));
    constexpr auto I_map_inv = I_map.inverse();
    STATIC_REQUIRE(all_equal(I_map_inv.linear_part(), I_map.linear_part()));
    STATIC_REQUIRE(all_equal(as_col_vector(I_map_inv.translation()),
                             as_col_vector(I_map.translation())));
    constexpr auto I_composed = I_map.compose(I_map_inv);
    STATIC_REQUIRE(all_equal(I_composed.linear_part(), I_map.linear_part()));
    STATIC_REQUIRE(all_equal(as_col_vector(I_composed.translation()),
                             as_col_vector(I_map.translation())));
  }

  SECTION("Implicit affine set creation check", "[Ax = b]") {
    constexpr auto set = ImplicitAffineSet<double, 3, 3>(I, b);
    STATIC_REQUIRE(set.is_valid());
    STATIC_REQUIRE(all_equal(set.basis(), I));
    STATIC_REQUIRE(all_equal(as_col_vector(set.point()), as_col_vector(b)));
    STATIC_REQUIRE(set.contains(b));
    STATIC_REQUIRE(!set.contains(Vectord<3>{0.0, 0.0, 0.0}));
    // (dakre) no matter what x we give the projection vector will always be in
    // the direction of 1, 2, 3 (b)
    constexpr auto projection = set.project(ones);
    STATIC_REQUIRE(all_equal(as_col_vector(projection), as_col_vector(b)));
    constexpr auto projection2 = set.project(zeros);
    STATIC_REQUIRE(all_equal(as_col_vector(projection2), as_col_vector(b)));

    constexpr auto set_two = set.transform(I_map);
    STATIC_REQUIRE(all_equal(set_two.basis(), set.basis()));
    STATIC_REQUIRE(
        all_equal(as_col_vector(set_two.point()), as_col_vector(set.point())));
  }

  SECTION("Parametric affine set creation check", "[x0 + U t]") {
    constexpr auto set = ParametricAffineSet<double, 3, 3>(I, b);
    STATIC_REQUIRE(set.is_valid());
    STATIC_REQUIRE(all_equal(set.basis(), I));
    STATIC_REQUIRE(all_equal(as_col_vector(set.point()), as_col_vector(b)));
    STATIC_REQUIRE(set.contains(b));
    // (dakre) this is inately different than Ax = b which for a given A that is
    // non-singular the solution is unique, however for a parametric form the
    // question is can we form a valid affine combination such that x can be
    // found provided basis U and x0 offset
    //
    // Does x - x0 lie in CS(U) ? Yes : No, since I \in R^3 forms a valid
    // orthonormal basis spanning R3 the origin does indeed lie in the CS(I)
    STATIC_REQUIRE(set.contains(Vectord<3>{0.0, 0.0, 0.0}));
    constexpr auto projection = set.project(ones);
    STATIC_REQUIRE(all_equal(as_col_vector(projection), as_col_vector(ones)));
    constexpr auto projection2 = set.project(zeros);
    STATIC_REQUIRE(all_equal(as_col_vector(projection2), as_col_vector(zeros)));

    constexpr auto set_two = set.transform(I_map);
    STATIC_REQUIRE(all_equal(set_two.basis(), set.basis()));
  }

  SECTION("Invalid parametric affine set creation") {
    constexpr auto a = Vectord<2>{1.0, 0.0};
    constexpr auto A = Matrixd<2, 2>{a, a};
    constexpr auto z = Vectord<2>{0.0, 0.0};
    constexpr auto set = ParametricAffineSet<double, 2, 2>(A, z);
    STATIC_REQUIRE(!set.is_valid());
  }
}
