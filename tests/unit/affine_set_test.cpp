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
    STATIC_REQUIRE(all_equal(I_map.translation(), zeros));
    constexpr auto I_map_inv = I_map.inverse();
    STATIC_REQUIRE(all_equal(I_map_inv.linear_part(), I_map.linear_part()));
    STATIC_REQUIRE(all_equal(I_map_inv.translation(), I_map.translation()));
    constexpr auto I_composed = I_map.compose(I_map_inv);
    STATIC_REQUIRE(all_equal(I_composed.linear_part(), I_map.linear_part()));
    STATIC_REQUIRE(all_equal(I_composed.translation(), I_map.translation()));
  }

  SECTION("Implicit affine set creation check", "[Ax = b]") {
    constexpr auto set = ImplicitAffineSet<double, 3, 3>(I, b);
    STATIC_REQUIRE(set.is_valid());
    STATIC_REQUIRE(all_equal(set.basis(), I));
    STATIC_REQUIRE(all_equal(set.point(), b));
    STATIC_REQUIRE(set.contains(b));
    STATIC_REQUIRE(!set.contains(Vectord<3>{0.0, 0.0, 0.0}));
    // (dakre) no matter what x we give the projection vector will always be in
    // the direction of 1, 2, 3 (b)
    constexpr auto projection = set.project(ones);
    STATIC_REQUIRE(all_equal(projection, b));
    constexpr auto projection2 = set.project(zeros);
    STATIC_REQUIRE(all_equal(projection2, b));

    constexpr auto set_two = set.transform(I_map);
    STATIC_REQUIRE(all_equal(set_two.basis(), set.basis()));
    STATIC_REQUIRE(all_equal(set_two.point(), set.point()));
  }

  SECTION("Parametric affine set creation check", "[x0 + U t]") {
    constexpr auto set = ParametricAffineSet<double, 3, 3>(I, b);
    STATIC_REQUIRE(set.is_valid());
    STATIC_REQUIRE(all_equal(set.basis(), I));
    STATIC_REQUIRE(all_equal(set.point(), b));
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
    STATIC_REQUIRE(all_equal(projection, ones));
    constexpr auto projection2 = set.project(zeros);
    STATIC_REQUIRE(all_equal(projection2, zeros));

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

  SECTION("Set affine set with 4x4 mat") {
    auto H = make_zeros<double, 4, 4>();
    H = set_block<0, 3, 0, 3>(H, I);
    const auto map = AffineMap<double, 3, 3>(H);

    auto A = Matrix<double, 3, 3>{};
    A[0][0] = 1.0, A[1][1] = 2.5, A[2][2] = 5.0;
    auto x0 = Vector<double, 3>{0.1, 0.1, 0.1};
    const auto set = ParametricAffineSet<double, 3, 3>(A, x0);

    const auto set_two = set.transform(map);
    REQUIRE(set_two.is_valid());
    REQUIRE(all_equal(set_two.basis(), set.basis()));
    REQUIRE(all_equal(set_two.point(), set.point()));
  }
}

TEST_CASE("Test 2d Space Operations", "[affine_sets]") {
  constexpr auto I = make_identity<double, 2>();
  constexpr auto x0 = Vectord<2>{0.0, 0.0};
  constexpr auto set = ParametricAffineSet<double, 2, 2>(I, x0);
  constexpr Rotation2d<double> R;
  SECTION("2d rotation") {
    constexpr auto R_mat = R(M_PI / 4.0);
    constexpr auto map = AffineMap2d<double>(R_mat, x0);
    constexpr auto set_rotated = set.transform(map);
    STATIC_REQUIRE(all_equal(set_rotated.basis(), R_mat * I));
    STATIC_REQUIRE(all_equal(set_rotated.point(), x0));
    constexpr auto R_mat_inv = linalg::inverse(R_mat);
    constexpr auto map2 = map.compose(AffineMap2d<double>(R_mat_inv, x0));
    constexpr auto set_id = set.transform(map2);
    STATIC_REQUIRE(all_equal(set_id.basis(), set.basis()));
    STATIC_REQUIRE(all_equal(set_id.point(), x0));
    STATIC_REQUIRE(set_id == set);
  }

    constexpr auto x1 = Vectord<2>{1.0, 2.0};
  SECTION("2d translation") {
    constexpr auto map = AffineMap2d<double>(I, x1);
    constexpr auto set_translated = set.transform(map);
    STATIC_REQUIRE(all_equal(set_translated.basis(), I));
    STATIC_REQUIRE(all_equal(set_translated.point(), x1));

    constexpr auto map2 = map.inverse();
    constexpr auto set_id = set_translated.transform(map2);
    STATIC_REQUIRE(all_equal(set_id.basis(), I));
    STATIC_REQUIRE(all_equal(set_id.point(), x0));
    STATIC_REQUIRE(set_id == set);
  }

  SECTION("2d [R | t]") {
    constexpr auto R_mat = R(M_PI / 6);
    constexpr auto x1 = Vectord<2>{1.0, 2.0};
    constexpr auto map = AffineMap2d<double>(R_mat, x1);
    constexpr auto set_Rt = set.transform(map);
    STATIC_REQUIRE(all_equal(set_Rt.basis(), R_mat * I));
    STATIC_REQUIRE(all_equal(set_Rt.point(), R_mat * x0 + x1));
    constexpr auto set_id = set_Rt.transform(map.inverse());
    STATIC_REQUIRE(set_id == set);
  }

  SECTION("2d scale") {
    constexpr auto S = Scale<double, 2>();
    constexpr auto S_mat = S(2.0, 3.0);
    constexpr auto map = AffineMap2d<double>(S_mat, x1);
    constexpr auto set_scaled = set.transform(map);
    STATIC_REQUIRE(all_equal(set_scaled.basis(), S_mat * I));
    STATIC_REQUIRE(all_equal(set_scaled.point(), S_mat * x0 + x1));
    constexpr auto set_id = set_scaled.transform(map.inverse());
    STATIC_REQUIRE(set_id == set);
  }
}

TEST_CASE("Test Rotations & Translations in Affine 3d Space", "[affine_sets]") {
  constexpr auto I = make_identity<double, 3>();
  constexpr auto x0 = Vectord<3>{0.0, 0.0, 0.0};
  constexpr auto set = ParametricAffineSet<double, 3, 3>(I, x0);
  constexpr Rotation3d<double> R;
  SECTION("3d rotation about x") {
    constexpr auto R_mat = R.x(M_PI / 2);
    constexpr auto map = AffineMap3d<double>(R_mat, x0);
    constexpr auto set_rotated = set.transform(map);
    STATIC_REQUIRE(all_equal(set_rotated.basis(), R_mat * I));
    STATIC_REQUIRE(all_equal(set_rotated.point(), x0));
    constexpr auto set_id = set_rotated.transform(map.inverse());
    STATIC_REQUIRE(set_id == set);
  }

  SECTION("3d rotation about y") {
    constexpr auto R_mat = R.y(-M_PI / 2);
    constexpr auto map = AffineMap3d<double>(R_mat, x0);
    constexpr auto set_rotated = set.transform(map);
    STATIC_REQUIRE(all_equal(set_rotated.basis(), R_mat * I));
    STATIC_REQUIRE(all_equal(set_rotated.point(), x0));
    constexpr auto set_id = set_rotated.transform(map.inverse());
    STATIC_REQUIRE(set_id == set);
  }

  SECTION("3d rotation about z") {
    constexpr auto R_mat = R.z(3 * M_PI / 2);
    constexpr auto map = AffineMap3d<double>(R_mat, x0);
    constexpr auto set_rotated = set.transform(map);
    STATIC_REQUIRE(all_equal(set_rotated.basis(), R_mat * I));
    STATIC_REQUIRE(all_equal(set_rotated.point(), x0));
    constexpr auto set_id = set_rotated.transform(map.inverse());
    STATIC_REQUIRE(set_id == set);
  }

  SECTION("3d translation") {
    constexpr auto x1 = Vectord<3>{1.0, 2.0, -5.0};
    constexpr auto map = AffineMap3d<double>(I, x1);
    constexpr auto set_translated = set.transform(map);
    STATIC_REQUIRE(all_equal(set_translated.basis(), I));
    STATIC_REQUIRE(all_equal(set_translated.point(), x1));
    constexpr auto set_id = set_translated.transform(map.inverse());
    STATIC_REQUIRE(all_equal(set_id.basis(), I));
    STATIC_REQUIRE(all_equal(set_id.point(), x0));
    STATIC_REQUIRE(set_id == set);
  }

  SECTION("3d [R | t]") {
    constexpr auto R_mat = R.z(-M_PI / 6);
    constexpr auto x1 = Vectord<3>{0.0, 0.2, -10.0};
    constexpr auto map = AffineMap3d<double>(R_mat, x1);
    constexpr auto set_Rt = set.transform(map);
    STATIC_REQUIRE(all_equal(set_Rt.basis(), R_mat * I));
    STATIC_REQUIRE(all_equal(set_Rt.point(), R_mat * x0 + x1));
    constexpr auto set_id = set_Rt.transform(map.inverse());
    STATIC_REQUIRE(set_id == set);
  }

  SECTION("3d scale x & y and reflection with scale on z") {
    constexpr auto S = Scale<double, 3>();
    constexpr auto S_mat = S(2.0, 3.0, -4.0);
    constexpr auto map = AffineMap3d<double>(S_mat, x0);
    constexpr auto set_scaled = set.transform(map);
    STATIC_REQUIRE(all_equal(set_scaled.basis(), S_mat * I));
    STATIC_REQUIRE(all_equal(set_scaled.point(), x0));
    constexpr auto set_id = set_scaled.transform(map.inverse());
    STATIC_REQUIRE(set_id == set);
  }
}
