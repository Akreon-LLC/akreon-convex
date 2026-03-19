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

#include "math.hpp"
#include "unit_test_helpers.hpp"

using namespace linalg;
using namespace affine;

TEST_CASE("Test affine set", "[affine_sets]") {
  constexpr auto I = make_identity<double, 3>();
  constexpr auto b = Vector<double, 3>{1.0, 2.0, 3.0};
  constexpr auto ones = Vector<double, 3>{1.0, 1.0, 1.0};
  constexpr auto zeros = Vector<double, 3>{0.0, 0.0, 0.0};

  constexpr auto I_map = AffineMap3d<double>(I, zeros);
  SECTION("Quick test on affine map properties") {
    STATIC_REQUIRE(all_equal(I_map.linear_part(), I));
    STATIC_REQUIRE(all_equal(I_map.translation(), zeros));
    constexpr auto I_map_inv = I_map.inverse();
    STATIC_REQUIRE(all_equal(I_map_inv.linear_part(), I_map.linear_part()));
    STATIC_REQUIRE(all_equal(I_map_inv.translation(), I_map.translation()));
    constexpr auto I_composed = I_map.compose(I_map_inv);
    STATIC_REQUIRE(all_equal(I_composed.linear_part(), I_map.linear_part()));
    STATIC_REQUIRE(all_equal(I_composed.translation(), I_map.translation()));
    STATIC_REQUIRE(I_map(ones) == ones);

    constexpr auto F =
        Matrix<double, 2, 2>{Vectord<2>{1.0, 2.0}, Vectord<2>{4.2, -2.0}};
    constexpr auto f_b = Vectord<2>{1.0, 1.0};
    constexpr auto G =
        Matrix<double, 2, 2>{Vectord<2>{1.0, 2.0}, Vectord<2>{4.2, -2.0}};
    constexpr auto g_b = Vectord<2>{-1.0, 2.0};
    constexpr auto F_map = AffineMap2d<double>(F, f_b);
    constexpr auto G_map = AffineMap2d<double>(G, g_b);
    constexpr auto I_map2 =
        AffineMap2d<double>(make_identity<double, 2>(), Vectord<2>{});
    STATIC_REQUIRE(F_map.compose(I_map2) == I_map2.compose(F_map));
    STATIC_REQUIRE(G_map.compose(I_map2) == I_map2.compose(G_map));
    constexpr auto res1 = F_map.compose(G_map);
    constexpr auto res2 = G_map.compose(F_map);
    STATIC_REQUIRE(all_equal(res1.linear_part(), F * G));
    STATIC_REQUIRE(all_equal(res2.linear_part(), G * F));
    STATIC_REQUIRE(all_equal(res1.translation(), F * g_b + f_b));
    STATIC_REQUIRE(all_equal(res2.translation(), G * f_b + g_b));
    STATIC_REQUIRE(F_map.compose(G_map).compose(I_map2) ==
                   F_map.compose(G_map.compose(I_map2)));
  }

  SECTION("Parametric affine set creation check", "[x0 + U t]") {
    constexpr auto set = ParametricAffineSet(I, b);
    STATIC_REQUIRE(set.is_well_formed());
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
    constexpr auto set = ParametricAffineSet(A, z);
    STATIC_REQUIRE(!set.is_well_formed());
  }

  SECTION("Set affine set with 4x4 mat") {
    constexpr auto H_mat = mat_traits<Matrixd<4, 4>>();
    constexpr auto H = H_mat(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                             1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    constexpr auto map = AffineMap3d<double>(H);

    constexpr auto A_mat = mat_traits<Matrixd<3, 3>>();
    constexpr auto A = A_mat(1.0, 0.0, 0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 5.0);
    constexpr auto x0 = Vector<double, 3>{0.1, 0.1, 0.1};
    constexpr auto set = ParametricAffineSet(A, x0);

    constexpr const auto set_two = set.transform(map);
    STATIC_REQUIRE(set_two.is_well_formed());
    STATIC_REQUIRE(all_equal(set_two.basis(), set.basis()));
    STATIC_REQUIRE(all_equal(set_two.point(), set.point()));
  }
}

TEST_CASE("Test 2d Space Operations", "[affine_sets]") {
  constexpr auto I = make_identity<double, 2>();
  constexpr auto x0 = Vectord<2>{0.0, 0.0};
  constexpr auto set = ParametricAffineSet(I, x0);
  constexpr Rotation2d<double> R;
  SECTION("2d rotation") {
    constexpr auto R_mat = R(M_PI / 4.0);
    constexpr auto map = AffineMap2d<double>(R_mat, x0);
    constexpr auto set_rotated = set.transform(map);
    STATIC_REQUIRE(all_equal(set_rotated.basis(), R_mat * I));
    STATIC_REQUIRE(all_equal(set_rotated.point(), x0));
    constexpr auto R_mat_inv_res =
        linalg::MatrixInverse<decltype(R_mat)>()(R_mat);
    constexpr auto R_mat_inv = std::get<0>(R_mat_inv_res);
    constexpr auto R_mat_inv_event = std::get<1>(R_mat_inv_res);
    STATIC_REQUIRE(R_mat_inv_event == linalg::NumericalEvent::NONE);
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

  SECTION("2d shear x dir by y") {
    constexpr auto H = Shear<double, 2>();
    constexpr auto H_mat = H(0, 1, 5.0);
    constexpr auto map = AffineMap2d<double>(H_mat, x1);
    constexpr auto set_sheared = set.transform(map);
    STATIC_REQUIRE(all_equal(set_sheared.basis(), H_mat * I));
    STATIC_REQUIRE(all_equal(set_sheared.point(), H_mat * x0 + x1));
    constexpr auto set_id = set_sheared.transform(map.inverse());
    STATIC_REQUIRE(set_id == set);
  }
}

TEST_CASE("Test 3d Affine Space Operations", "[affine_sets]") {
  constexpr auto I = make_identity<double, 3>();
  constexpr auto x0 = Vectord<3>{0.0, 0.0, 0.0};
  constexpr auto set = ParametricAffineSet(I, x0);
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

  SECTION("2d shear y dir by z") {
    constexpr auto H = Shear<double, 3>();
    constexpr auto H_mat = H(1, 2, -1.5);
    constexpr auto map = AffineMap3d<double>(H_mat, x0);
    constexpr auto set_sheared = set.transform(map);
    STATIC_REQUIRE(all_equal(set_sheared.basis(), H_mat * I));
    STATIC_REQUIRE(all_equal(set_sheared.point(), x0));
    constexpr auto set_id = set_sheared.transform(map.inverse());
    STATIC_REQUIRE(set_id == set);
  }

  SECTION("Rigid body transformation") {
    constexpr double psi = M_PI / 2;
    constexpr double theta = M_PI / 4;
    constexpr double phi = -M_PI / 8;
    constexpr auto x1 = Vector<double, 3>{2.0, -3.1, 1.2};
    constexpr auto R_mat = R.x(phi) * (R.y(theta) * R.z(psi));
    constexpr auto rigid_body_map_T = AffineMap3d<double>(R_mat, x1);
    constexpr auto set_rigid_body_T = set.transform(rigid_body_map_T);
    STATIC_REQUIRE(all_equal(set_rigid_body_T.basis(), R_mat));
    STATIC_REQUIRE(all_equal(set_rigid_body_T.point(), R_mat * x0 + x1));
    constexpr auto set_id =
        set_rigid_body_T.transform(rigid_body_map_T.inverse());
    STATIC_REQUIRE(set_id == set);
    constexpr auto rigid_body_map_id =
        rigid_body_map_T.compose(rigid_body_map_T.inverse());
    STATIC_REQUIRE(set_rigid_body_T.transform(rigid_body_map_id) ==
                   set_rigid_body_T);
  }
}
