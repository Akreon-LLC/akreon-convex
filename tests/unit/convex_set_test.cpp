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

TEST_CASE("Test affine subspace", "[convex_sets]") {
  SECTION("Test affine subspace") {
    using Space = Matrixd<1, 3>;
    constexpr auto I = make_identity<double, 3>();
    constexpr auto affine_set =
        AffineSubspace<Space, Matrixd<3, 3>>(I, Vectord<3>{});
    STATIC_REQUIRE(affine_set.is_well_formed());
    STATIC_REQUIRE(affine_set.contains(Vectord<3>{}));
    STATIC_REQUIRE(!affine_set.contains(Vectord<3>{1.0, 1.0, 1.0}));

    constexpr auto cone_block = std::get<0>(affine_set.canonicalize());
    constexpr auto A = make_identity<double, 4>();
    STATIC_REQUIRE(cone_block.map == affine::AffineMap(A, Vectord<4>{}));
    constexpr auto cones = cone_block.cones;
    constexpr auto sz = std::tuple_size<decltype(cones)>::value;
    STATIC_REQUIRE(sz == 2);
    STATIC_REQUIRE(cone_block.validate());
  }
}

TEST_CASE("Test halfspace convex set", "[convex_sets]") {
  SECTION("Test hyperplane") {
    constexpr Vector<double, 3> point{0.1, 0.2, 0.3};
    constexpr double b = 0.6;
    constexpr Hyperplane<Matrix<double, 1, 3>> hyperplane(point, b);
    STATIC_REQUIRE(hyperplane.is_well_formed());
    STATIC_REQUIRE(!hyperplane.contains(Vectord<3>{}));
    STATIC_REQUIRE(hyperplane.contains(Vectord<3>{1.0, 1.0, 1.0}));
    STATIC_REQUIRE(!hyperplane.contains(Vectord<3>{1.0, 1.0, 2.0}));

    STATIC_REQUIRE(
        !Hyperplane<Matrix<double, 1, 1>>(Vectord<1>{}, b).is_well_formed());
    constexpr auto I = make_identity<double, 3>();
    constexpr auto hyperplane_id =
        hyperplane.transform(AffineMap<Matrix<double, 3, 3>>(I, Vectord<3>{}));
    STATIC_REQUIRE(hyperplane_id == hyperplane);

    constexpr auto cone_block = std::get<0>(hyperplane.canonicalize());
    constexpr mat_traits<Matrix<double, 2, 4>> A_mat;
    constexpr auto A =
        A_mat(point(0), point(1), point(2), -1 * b, 0.0, 0.0, 0.0, 1.0);

    STATIC_REQUIRE(cone_block.map == affine::AffineMap(A, Vectord<2>{}));

    constexpr auto cones = cone_block.cones;
    constexpr auto sz = std::tuple_size<decltype(cones)>::value;
    STATIC_REQUIRE(sz == 2);
    STATIC_REQUIRE(cone_block.validate());
  }

  SECTION("Test closed lower halfspace") {
    constexpr Vector<double, 2> point{1.0, 2.0};
    constexpr double b = 1.0;
    constexpr Halfspace<Matrix<double, 1, 2>, HalfspaceType::CLH> halfspace(
        point, b);
    STATIC_REQUIRE(halfspace.is_well_formed());
    STATIC_REQUIRE(halfspace.contains(Vectord<2>{}));
    STATIC_REQUIRE(halfspace.contains(Vectord<2>{-1.0, 0.0}));
    STATIC_REQUIRE(halfspace.contains(Vectord<2>{-1.0, -2.0}));
    STATIC_REQUIRE(halfspace.contains(Vectord<2>{1.0, 0.0}));
    STATIC_REQUIRE(!halfspace.contains(Vectord<2>{1.0, 2.0}));
    STATIC_REQUIRE(!halfspace.contains(Vectord<2>{-1.0, 5.0}));

    constexpr auto cone_block = std::get<0>(halfspace.canonicalize());
    constexpr mat_traits<Matrix<double, 2, 3>> A_mat;
    constexpr auto A = A_mat(-1 * point(0), -1 * point(1), b, 0.0, 0.0, 1.0);

    STATIC_REQUIRE(cone_block.map == affine::AffineMap(A, Vectord<2>{}));

    constexpr auto cones = cone_block.cones;
    constexpr auto sz = std::tuple_size<decltype(cones)>::value;
    STATIC_REQUIRE(sz == 2);
    STATIC_REQUIRE(cone_block.validate());
  }

  SECTION("Test open lower halfspace") {
    constexpr Vector<double, 4> point{2.0, 1.0, -2.0, 5.0};
    constexpr double b = 10.0;
    constexpr auto halfspace =
        Halfspace<Matrix<double, 1, 4>, HalfspaceType::OLH>(point, b);

    STATIC_REQUIRE(halfspace.is_well_formed());
    STATIC_REQUIRE(halfspace.contains(Vectord<4>{}));
    STATIC_REQUIRE(!halfspace.contains(Vectord<4>{0.0, 0.0, 0.0, 2.0}));
    STATIC_REQUIRE(halfspace.contains(Vectord<4>{0.0, 0.0, 0.0, 1.9}));
    STATIC_REQUIRE(halfspace.contains(Vectord<4>{0.5, 2.0, 2.0, 1.0}));
    STATIC_REQUIRE(!halfspace.contains(Vectord<4>{1.0, 1.0, 0.0, 5.0}));
  }

  SECTION("Test closed upper halfspace") {
    constexpr Vector<double, 3> point{1.0, 2.0, 1.0};
    constexpr double b = 2.0;
    constexpr Halfspace<Matrix<double, 1, 3>, HalfspaceType::CUH> halfspace(
        point, b);
    STATIC_REQUIRE(halfspace.is_well_formed());
    STATIC_REQUIRE(!halfspace.contains(Vectord<3>{}));
    STATIC_REQUIRE(halfspace.contains(Vectord<3>{0.0, 1.0, 0.0}));
    STATIC_REQUIRE(halfspace.contains(Vectord<3>{0.1, 1.0, 2.0}));
    STATIC_REQUIRE(!halfspace.contains(Vectord<3>{0.1, 1.0, -2.0}));

    constexpr auto halfspace_id =
        halfspace.transform(AffineMap<Matrix<double, 3, 3>>(
            make_identity<double, 3>(), Vectord<3>{}));
    STATIC_REQUIRE(halfspace_id == halfspace);

    constexpr auto t = Vectord<3>{1.0, 5.0, 0.0};
    constexpr auto halfspace_trans = halfspace.transform(
        AffineMap<Matrix<double, 3, 3>>(make_identity<double, 3>(), t));
    STATIC_REQUIRE(halfspace_trans != halfspace_id);

    constexpr auto cone_block = std::get<0>(halfspace.canonicalize());
    constexpr mat_traits<Matrix<double, 2, 4>> A_mat;
    constexpr auto A = A_mat(-1 * point(0), -1 * point(1), -1 * point(2), b,
                             0.0, 0.0, 0.0, 1.0);

    STATIC_REQUIRE(cone_block.map == affine::AffineMap(A, Vectord<2>{}));

    constexpr auto cones = cone_block.cones;
    constexpr auto sz = std::tuple_size<decltype(cones)>::value;
    STATIC_REQUIRE(sz == 2);
    STATIC_REQUIRE(cone_block.validate());
  }

  SECTION("Test open upper halfspace") {
    constexpr Vector<double, 1> point{1.0};
    constexpr double b = 5.0;
    constexpr Halfspace<Matrix<double, 1, 1>, HalfspaceType::OUH> halfspace(
        point, b);
    STATIC_REQUIRE(halfspace.is_well_formed());
    STATIC_REQUIRE(!halfspace.contains(Vectord<1>{}));
    STATIC_REQUIRE(!halfspace.contains(Vectord<1>{0.1}));
    STATIC_REQUIRE(!halfspace.contains(Vectord<1>{5.0}));
    STATIC_REQUIRE(halfspace.contains(Vectord<1>{6.0}));
  }
}

TEST_CASE("Test ellipsoid set", "[convex_sets]") {
  SECTION("Quick test to check convex sets types") {
    STATIC_REQUIRE(
        ConvexSetLike<Ellipsoid<Matrix<double, 1, 5>, EllipsoidType::OE>>);
    STATIC_REQUIRE(
        ConvexSetLike<Ellipsoid<Matrix<float, 1, 2>, EllipsoidType::CE>>);
  }

  SECTION("Test open ellipsoid set") {
    constexpr auto c = Vector<double, 2>{1.5, 2.0};
    constexpr auto P = make_identity<double, 2>();
    constexpr auto ellipsoid =
        Ellipsoid<Matrix<double, 1, 2>, EllipsoidType::OE>(c, P);
    STATIC_REQUIRE(ellipsoid.is_well_formed());
    STATIC_REQUIRE(!ellipsoid.contains(Vectord<2>{}));
    STATIC_REQUIRE(ellipsoid.contains(c));
    STATIC_REQUIRE(ellipsoid.contains(Vectord<2>{c(0) + 0.1, c(1) - 0.5}));
    STATIC_REQUIRE(!ellipsoid.contains(Vectord<2>{1.5, 1.0}));

    constexpr auto ellipsoid_id =
        ellipsoid.transform(AffineMap<decltype(P)>(P, Vectord<2>{}));
    STATIC_REQUIRE(ellipsoid_id == ellipsoid);

    constexpr mat_traits<Matrix<double, 2, 2>> R_mat;
    constexpr auto R = R_mat(M_PI / 2, 0.0, 0.0, M_PI / 2);
    constexpr auto ellipsoid_90 =
        ellipsoid.transform(AffineMap<decltype(R)>(R, Vectord<2>{}));
    STATIC_REQUIRE(ellipsoid_90 ==
                   Ellipsoid<Matrix<double, 1, 2>, EllipsoidType::OE>(
                       R * c, transpose(inverse(R)) * inverse(R)));

#if 0
    constexpr auto cone = ellipsoid.cone_view();
    STATIC_REQUIRE(!cone.contains(std::tuple(Vectord<2>{}, 0.0)));
    STATIC_REQUIRE(cone.contains(std::tuple(Vectord<2>{c(0), c(1)}, 1.0)));
    STATIC_REQUIRE(
        cone.contains(std::tuple(Vectord<2>{c(0) + 0.1, c(1) - 0.5}, 1.0)));
    STATIC_REQUIRE(!cone.contains(std::tuple(Vectord<2>{1.5, 1.0}, 1.0)));
    STATIC_REQUIRE(cone.is_proper() == ConeCheck::False);
#endif
  }

  SECTION("Test closed ellipsoid set") {
    constexpr auto c = Vector<double, 2>{1.5, 2.0};
    constexpr auto P = make_identity<double, 2>();
    constexpr auto ellipsoid =
        Ellipsoid<Matrix<double, 1, 2>, EllipsoidType::CE>(c, P);
    REQUIRE(ellipsoid.is_well_formed());
    STATIC_REQUIRE(!ellipsoid.contains(Vectord<2>{}));
    STATIC_REQUIRE(ellipsoid.contains(c));
    STATIC_REQUIRE(ellipsoid.contains(Vectord<2>{c(0) + 0.1, c(1) - 0.5}));
    STATIC_REQUIRE(ellipsoid.contains(Vectord<2>{1.5, 1.0}));

#if 0
    constexpr auto cone = ellipsoid.cone_view();
    STATIC_REQUIRE(cone.contains(std::tuple(Vectord<2>{c(0), c(1)}, 1.0)));
    STATIC_REQUIRE(
        cone.contains(std::tuple(Vectord<2>{c(0) + 0.1, c(1) - 0.5}, 1.0)));
    STATIC_REQUIRE(cone.contains(std::tuple(Vectord<2>{1.5, 1.0}, 1.0)));
    STATIC_REQUIRE(cone.is_proper() == ConeCheck::True);
#endif
  }
}

TEST_CASE("Polyhedron", "[convex_sets]") {
  using Space = Matrix<double, 1, 3>;
  mat_traits<Matrix<double, 7, 3>> A_mat;
  constexpr auto A =
      A_mat(1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0, -1.0, 1.0, 1.0, 0.0);
  STATIC_REQUIRE(mat_rows_v<decltype(A)> == 7);
  STATIC_REQUIRE(mat_cols_v<decltype(A)> == 3);
  constexpr Vector<double, 7> b{1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0};

  SECTION("Test Closed Polyhedron", "[A x <= b]") {
    constexpr Polyhedron<Space, HalfspaceType::CLH, decltype(A)> poly(A, b);

    STATIC_REQUIRE(poly.is_well_formed());
    STATIC_REQUIRE(poly.contains(Vector<double, 3>{}));
    STATIC_REQUIRE(!poly.contains(Vector<double, 3>{1.0, 2.0, 3.0}));
    STATIC_REQUIRE(poly.contains(Vector<double, 3>{1.0, 0.0, 0.0}));

    constexpr auto blocks = poly.canonicalize();
    constexpr auto n_blocks = std::tuple_size<decltype(blocks)>::value;
    STATIC_REQUIRE(n_blocks == 7);

    constexpr auto cone_block = std::get<0>(blocks);
    constexpr mat_traits<Matrix<double, 2, 4>> P_mat;
    constexpr auto P = P_mat(-1.0, 0.0, 0.0, b(0), 0.0, 0.0, 0.0, 1.0);

    STATIC_REQUIRE(cone_block.map == affine::AffineMap(P, Vectord<2>{}));

    constexpr auto cones = cone_block.cones;
    constexpr auto sz = std::tuple_size<decltype(cones)>::value;
    STATIC_REQUIRE(sz == 2);
    STATIC_REQUIRE(cone_block.validate());
  }

  SECTION("Test Open Polyhedron", "[A x < b]") {
    constexpr Polyhedron<Space, HalfspaceType::OLH, decltype(A)> poly(A, b);

    STATIC_REQUIRE(poly.is_well_formed());
    STATIC_REQUIRE(!poly.contains(Vector<double, 3>{}));
    STATIC_REQUIRE(!poly.contains(Vector<double, 3>{1.0, 2.0, 3.0}));
    STATIC_REQUIRE(!poly.contains(Vector<double, 3>{1.0, 0.0, 0.0}));
    STATIC_REQUIRE(poly.contains(Vector<double, 3>{0.1, 0.1, 0.1}));
  }

  mat_traits<Matrix<double, 2, 3>> C_mat;
  constexpr auto C = C_mat(1.0, 1.0, 1.0, 2.0, 2.0, 2.0);
  constexpr Vector<double, 2> d{1.0, 2.0};

  constexpr auto I = make_identity<double, 3>();
  SECTION("Test Closed Polyhedron", "[A x <= b, C x == d]") {
    constexpr Polyhedron<Space, HalfspaceType::CLH, decltype(A), decltype(C)>
        poly(A, b, C, d);

    STATIC_REQUIRE(poly.is_well_formed());
    STATIC_REQUIRE(!poly.contains(Vector<double, 3>{}));
    STATIC_REQUIRE(poly.contains(Vector<double, 3>{1.0, 0.0, 0.0}));
    STATIC_REQUIRE(poly.contains(Vector<double, 3>{0.0, 1.0, 0.0}));
    STATIC_REQUIRE(poly.contains(Vector<double, 3>{0.0, 0.0, 1.0}));
    STATIC_REQUIRE(poly.transform(AffineMap<Matrixd<3, 3>>(I, Vectord<3>{})) ==
                   poly);

    constexpr auto blocks = poly.canonicalize();
    constexpr auto n_blocks = std::tuple_size<decltype(blocks)>::value;
    STATIC_REQUIRE(n_blocks == 9);

    constexpr auto cone_block = std::get<8>(blocks);
    constexpr mat_traits<Matrix<double, 2, 4>> P_mat;
    constexpr auto P = P_mat(2.0, 2.0, 2.0, -1 * d(1), 0.0, 0.0, 0.0, 1.0);

    STATIC_REQUIRE(cone_block.map == affine::AffineMap(P, Vectord<2>{}));

    constexpr auto cones = cone_block.cones;
    constexpr auto sz = std::tuple_size<decltype(cones)>::value;
    STATIC_REQUIRE(sz == 2);
    STATIC_REQUIRE(cone_block.validate());
  }

  SECTION("Test Open Polyhedron", "[A x < b, C x == d]") {
    constexpr Polyhedron<Space, HalfspaceType::OLH, decltype(A), decltype(C)>
        poly(A, b, C, d);

    STATIC_REQUIRE(poly.is_well_formed());
    STATIC_REQUIRE(!poly.contains(Vector<double, 3>{}));
    STATIC_REQUIRE(!poly.contains(Vector<double, 3>{1.0, 0.0, 0.0}));
    STATIC_REQUIRE(!poly.contains(Vector<double, 3>{0.0, 1.0, 0.0}));
    STATIC_REQUIRE(!poly.contains(Vector<double, 3>{0.0, 0.0, 1.0}));
    STATIC_REQUIRE(poly.contains(Vector<double, 3>{0.2, 0.3, 0.5}));
    STATIC_REQUIRE(poly.transform(AffineMap<Matrixd<3, 3>>(I, Vectord<3>{})) ==
                   poly);
  }
}

TEST_CASE("Test Simplex", "[convex_sets]") {
  SECTION("Test Simplex", "[{ x | x >= 0, sum_i x_i = 1}]") {
    constexpr Simplex<Matrixd<1, 3>> simplex;
    STATIC_REQUIRE(simplex.is_well_formed());
    STATIC_REQUIRE(simplex == simplex);
    STATIC_REQUIRE(simplex.contains(Vectord<3>{0.8, 0.1, 0.1}));
    STATIC_REQUIRE(!simplex.contains(Vectord<3>{-0.8, 0.1, 0.1}));
    STATIC_REQUIRE(!simplex.contains(Vectord<3>{0.8, 0.1, 0.0}));
    constexpr auto I = make_identity<double, 3>();
    constexpr auto poly =
        simplex.transform(AffineMap<Matrixd<3, 3>>(I, Vectord<3>{}));
    STATIC_REQUIRE(poly.contains(Vectord<3>{0.8, 0.1, 0.1}));
    STATIC_REQUIRE(!poly.contains(Vectord<3>{-0.8, 0.1, 0.1}));
    STATIC_REQUIRE(!poly.contains(Vectord<3>{0.8, 0.1, 0.0}));

    constexpr auto cone_block = std::get<0>(simplex.canonicalize());
    constexpr mat_traits<Matrix<double, 5, 4>> A_mat;
    constexpr auto A =
        A_mat(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0,
              -1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 1.0);

    STATIC_REQUIRE(cone_block.map == affine::AffineMap(A, Vectord<5>{}));

    constexpr auto cones = cone_block.cones;
    constexpr auto sz = std::tuple_size<decltype(cones)>::value;
    STATIC_REQUIRE(sz == 3);
    STATIC_REQUIRE(cone_block.validate());
  }
}

TEST_CASE("Test LMI", "[convex_sets]") {
  SECTION("Test LMI", "[A(x) <= B]") {
    constexpr auto I = make_identity<double, 3>();
    constexpr LMI<Matrixd<1, 3>> set(I, I, I, I);
    STATIC_REQUIRE(set.is_well_formed());
    STATIC_REQUIRE(!set.contains(Vectord<3>{1.0, 2.0, 3.0}));
    STATIC_REQUIRE(set.contains(Vectord<3>{}));
    STATIC_REQUIRE(set.contains(Vectord<3>{0.2, 0.5, 0.1}));
    STATIC_REQUIRE(set.contains(Vectord<3>{1. / 3., 1. / 3., 1. / 3.}));

    constexpr auto cone_block = std::get<0>(set.canonicalize());
    constexpr mat_traits<Matrix<double, 7, 4>> F_mat;
    constexpr auto F = F_mat(-1.0, -1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0,
                             -1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, -1.0, -1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    STATIC_REQUIRE(cone_block.map == affine::AffineMap(F, Vectord<7>{}));

    constexpr auto cones = cone_block.cones;
    constexpr auto sz = std::tuple_size<decltype(cones)>::value;
    STATIC_REQUIRE(sz == 2);
    STATIC_REQUIRE(cone_block.validate());
  }
}

TEST_CASE("Test norm ball convex set", "[convex_sets]") {
  SECTION("Test open L2 norm ball") {
    constexpr auto c = Vector<double, 2>{1.0, 3.0};
    constexpr auto r = 2.0;
    constexpr auto ball =
        NormBall<Matrix<double, 1, 2>, NormBallType::L2OB>(c, r);
    STATIC_REQUIRE(ball.is_well_formed());
    STATIC_REQUIRE(!ball.contains(Vector<double, 2>{}));
    STATIC_REQUIRE(ball.contains(c));
    STATIC_REQUIRE(ball.contains(Vector<double, 2>{c(0) + 0.1, c(1) - 0.2}));
    STATIC_REQUIRE(!ball.contains(Vector<double, 2>{c(0) + 2.0, c(1)}));

    constexpr auto P = r * r * make_identity<double, 2>();
    constexpr auto ellipse =
        Ellipsoid<Matrix<double, 1, 2>, EllipsoidType::OE>(c, P);
    constexpr auto res = ball.transform(AffineMap<Matrix<double, 2, 2>>(
        make_identity<double, 2>(), Vectord<2>{}));
    STATIC_REQUIRE(res == ellipse);
  }

  SECTION("Test closed L2 norm ball") {
    constexpr auto c = Vector<float, 2>{1.f, 3.f};
    constexpr auto r = 2.f;
    constexpr auto ball =
        NormBall<Matrix<float, 1, 2>, NormBallType::L2CB>(c, r);
    STATIC_REQUIRE(ball.is_well_formed());
    STATIC_REQUIRE(!ball.contains(Vector<float, 2>{}));
    STATIC_REQUIRE(ball.contains(c));
    STATIC_REQUIRE(ball.contains(Vector<float, 2>{c(0) + 0.1f, c(1) - 0.2f}));
    STATIC_REQUIRE(ball.contains(Vector<float, 2>{c(0) + 2.f, c(1)}));

    constexpr auto P = r * r * make_identity<float, 2>();
    constexpr auto ellipse =
        Ellipsoid<Matrix<float, 1, 2>, EllipsoidType::CE>(c, P);
    constexpr auto res = ball.transform(AffineMap<Matrix<float, 2, 2>>(
        make_identity<float, 2>(), Vectorf<2>{}));
    STATIC_REQUIRE(res == ellipse);

    constexpr mat_traits<Matrix<float, 3, 3>> F_mat;
    constexpr auto F = F_mat(1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, r);
    constexpr linalg::Vector<float, 3> g{-1 * c(0), -1 * c(1), 0.f};
    constexpr auto cone_block = std::get<0>(ball.canonicalize());
    STATIC_REQUIRE(cone_block.map == affine::AffineMap(F, g));
    constexpr auto cones = cone_block.cones;
    constexpr auto sz = std::tuple_size<decltype(cones)>::value;
    STATIC_REQUIRE(sz == 2);
    STATIC_REQUIRE(cone_block.validate());
  }

  SECTION("Test closed L1 norm ball") {
    constexpr auto c = Vector<double, 4>{0.0, 1.2, -0.8, 2.0};
    constexpr auto r = 1.0;
    constexpr auto ball =
        NormBall<Matrix<double, 1, 4>, NormBallType::L1CB>(c, r);
    STATIC_REQUIRE(ball.is_well_formed());
    STATIC_REQUIRE(ball.contains(c));
    STATIC_REQUIRE(!ball.contains(Vectord<4>{}));
    STATIC_REQUIRE(ball.contains(Vectord<4>{c(0), c(1), c(2), c(3) + 0.1}));
    STATIC_REQUIRE(ball.contains(Vectord<4>{c(0) + 1.0, c(1), c(2), c(3)}));

    constexpr auto I = make_identity<double, 4>();
    constexpr auto res =
        ball.transform(AffineMap<Matrixd<4, 4>>(I, Vectord<4>{}));

    STATIC_REQUIRE(res.is_well_formed());
    STATIC_REQUIRE(
        !res.contains(Vectord<8>{c(0), c(1), c(2), c(3), r, r, r, r}));
    STATIC_REQUIRE(
        res.contains(Vectord<8>{c(0), c(1), c(2), c(3), 0.0, 0.0, 0.0, 0.0}));
    STATIC_REQUIRE(res.contains(Vectord<8>{c(0) + 0.3, c(1) - 0.2, c(2) + 0.1,
                                           c(3) - 0.1, 0.3, 0.2, 0.1, 0.1}));

    constexpr mat_traits<Matrix<double, 14, 9>> F_mat;
    constexpr auto F =
        F_mat(1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0.,
              0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0.,
              0., 0., 1., 0., -1., 0., 0., 0., 1., 0., 0., 0., 0., 0., -1., 0.,
              0., 0., 1., 0., 0., 0., 0., 0., -1., 0., 0., 0., 1., 0., 0., 0.,
              0., 0., -1., 0., 0., 0., 1., 0., 0., 0., 0., 0., -1., -1., -1.,
              -1., r, 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
              1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
              0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.);

    constexpr linalg::Vector<double, 14> g{
        -1 * c(0), -1 * c(1), -1 * c(2), -1 * c(3), c(0), c(1), c(2),
        c(3),      0.,        0.,        0.,        0.,   0.,   0.};

    constexpr auto cone_block = std::get<0>(ball.canonicalize());
    STATIC_REQUIRE(cone_block.map == affine::AffineMap(F, g));
    constexpr auto cones = cone_block.cones;
    constexpr auto sz = std::tuple_size<decltype(cones)>::value;
    STATIC_REQUIRE(sz == 5);
    STATIC_REQUIRE(cone_block.validate());
  }

  SECTION("Test open L1 norm ball") {
    constexpr auto c = Vector<float, 4>{0.f, 1.2f, -0.8f, 2.f};
    constexpr auto r = 1.f;
    constexpr auto ball =
        NormBall<Matrix<float, 1, 4>, NormBallType::L1OB>(c, r);
    STATIC_REQUIRE(ball.is_well_formed());
    STATIC_REQUIRE(ball.contains(c));
    STATIC_REQUIRE(!ball.contains(Vectorf<4>{}));
    STATIC_REQUIRE(ball.contains(Vectorf<4>{c(0), c(1), c(2), c(3) + 0.1f}));
    STATIC_REQUIRE(!ball.contains(Vectorf<4>{c(0) + 1.f, c(1), c(2), c(3)}));
  }

  SECTION("Test open Linf norm ball") {
    constexpr auto c = Vector<double, 3>{1.0, 2.0, 3.0};
    constexpr auto r = 2.0;
    constexpr auto ball =
        NormBall<Matrix<double, 1, 3>, NormBallType::LinfOB>(c, r);
    STATIC_REQUIRE(ball.is_well_formed());
    STATIC_REQUIRE(ball.contains(c));
    STATIC_REQUIRE(!ball.contains(Vectord<3>{}));
    STATIC_REQUIRE(ball.contains(Vectord<3>{c(0) + 1.0, c(1), c(2)}));
  }

  SECTION("Test closed  Linf norm ball") {
    constexpr auto c = Vector<float, 3>{1.f, 2.f, 3.f};
    constexpr auto r = 2.f;
    constexpr auto ball =
        NormBall<Matrix<float, 1, 3>, NormBallType::LinfCB>(c, r);
    STATIC_REQUIRE(ball.is_well_formed());
    STATIC_REQUIRE(ball.contains(c));
    STATIC_REQUIRE(!ball.contains(Vectorf<3>{}));
    STATIC_REQUIRE(ball.contains(Vectorf<3>{c(0) + 1.f, c(1), c(2)}));

    constexpr auto I = make_identity<float, 3>();
    constexpr auto res =
        ball.transform(AffineMap<Matrixf<3, 3>>(I, Vectorf<3>{}));
    STATIC_REQUIRE(res.is_well_formed());
    STATIC_REQUIRE(res.contains(c));
    STATIC_REQUIRE(!res.contains(Vectorf<3>{}));
    STATIC_REQUIRE(ball.contains(Vectorf<3>{c(0) + 1.f, c(1), c(2)}));

    constexpr mat_traits<Matrix<float, 7, 4>> F_mat;
    constexpr auto F = F_mat(1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f,
                             1.f, 0.f, -1.f, 0.f, 0.f, 0.f, 0.f, -1.f, 0.f, 0.f,
                             0.f, 0.f, -1.f, 0.f, 0.f, 0.f, 0.f, r);
    constexpr linalg::Vector<float, 7> g{-1 * c(0), -1 * c(1), -1 * c(2), c(0),
                                         c(1),      c(2),      0.f};
    constexpr auto cone_block = std::get<0>(ball.canonicalize());
    STATIC_REQUIRE(cone_block.map == affine::AffineMap(F, g));
    constexpr auto cones = cone_block.cones;
    constexpr auto sz = std::tuple_size<decltype(cones)>::value;
    STATIC_REQUIRE(sz == 3);
    STATIC_REQUIRE(cone_block.validate());
  }
}
