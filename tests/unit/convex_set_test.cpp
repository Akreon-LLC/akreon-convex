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

TEST_CASE("Test halfspace convex set", "[convex_sets]") {
  SECTION("Quick test to check convex sets types") {
    STATIC_REQUIRE(VectorSpace<Matrix<double, 1, 1>>);
    STATIC_REQUIRE(ConvexSetLike<Hyperplane<Matrix<double, 1, 4>>>);
    STATIC_REQUIRE(
        ConvexSetLike<Halfspace<Matrix<double, 1, 2>, HalfspaceType::OUH>>);
    STATIC_REQUIRE(
        ConvexSetLike<Halfspace<Matrix<float, 1, 10>, HalfspaceType::CLH>>);
  }

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

    constexpr ProjectiveMap<Matrix<double, 3, 3>, Matrix<double, 1, 3>>
        proj_map(I, Vectord<3>{}, as_row_vector(Vectord<3>{}), 1.0);
    constexpr auto x_proj =
        hyperplane.project(proj_map, Vectord<3>{1.0, 1.0, 1.0});
    STATIC_REQUIRE(x_proj.has_value());
    STATIC_REQUIRE(x_proj.value() == x_proj);
    STATIC_REQUIRE(
        !hyperplane.project(proj_map, Vectord<3>{1.0, 1.0, 2.0}).has_value());

    constexpr auto hyperplane_cone = hyperplane.cone_view();
    STATIC_REQUIRE(hyperplane_cone.contains(Vectord<4>{}));
    STATIC_REQUIRE(!hyperplane_cone.contains(Vectord<4>{0.0, 0.0, 0.0, 1.0}));
    STATIC_REQUIRE(hyperplane_cone.contains(Vectord<4>{1.0, 1.0, 1.0, 1.0}));
    STATIC_REQUIRE(!hyperplane_cone.contains(Vectord<4>{1.0, 1.0, 2.0, 1.0}));
    STATIC_REQUIRE(!hyperplane_cone.is_proper());
    constexpr Vectord<4> lifted_point{point(0), point(1), point(2), 1.0};
    STATIC_REQUIRE(
        !hyperplane_cone.less_than(lifted_point, lifted_point).has_value());
    STATIC_REQUIRE(!hyperplane_cone.less_than_strict(lifted_point, lifted_point)
                        .has_value());
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

    constexpr auto halfspace_cone = halfspace.cone_view();
    STATIC_REQUIRE(halfspace_cone.contains(Vectord<3>{}));
    STATIC_REQUIRE(halfspace_cone.contains(Vectord<3>{-1.0, 0.0, 1.0}));
    STATIC_REQUIRE(halfspace_cone.contains(Vectord<3>{-1.0, -2.0, 1.0}));
    STATIC_REQUIRE(halfspace_cone.contains(Vectord<3>{1.0, 0.0, 1.0}));
    STATIC_REQUIRE(!halfspace_cone.contains(Vectord<3>{1.0, 2.0, 1.0}));
    STATIC_REQUIRE(!halfspace_cone.contains(Vectord<3>{-1.0, 5.0, 1.0}));
    STATIC_REQUIRE(!halfspace_cone.is_proper());

    constexpr Vectord<3> lifted_point{point(0), point(1), 1.0};
    STATIC_REQUIRE(
        !halfspace_cone.less_than(lifted_point, lifted_point).has_value());
    STATIC_REQUIRE(!halfspace_cone.less_than_strict(lifted_point, lifted_point)
                        .has_value());
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

    constexpr auto I = make_identity<double, 4>();
    constexpr ProjectiveMap<Matrix<double, 4, 4>, Matrix<double, 1, 4>>
        proj_map(I, Vectord<4>{}, as_row_vector(Vectord<4>{}), 1.0);
    constexpr auto x_proj =
        halfspace.project(proj_map, Vectord<4>{0.0, 0.0, 0.0, 1.9});
    STATIC_REQUIRE(x_proj.has_value());
    STATIC_REQUIRE(x_proj.value() == x_proj);
    STATIC_REQUIRE(!halfspace.project(proj_map, Vectord<4>{0.0, 0.0, 0.0, 2.0})
                        .has_value());

    constexpr auto halfspace_cone = halfspace.cone_view();
    STATIC_REQUIRE(!halfspace_cone.contains(Vectord<5>{}));
    STATIC_REQUIRE(
        !halfspace_cone.contains(Vectord<5>{0.0, 0.0, 0.0, 2.0, 1.0}));
    STATIC_REQUIRE(
        halfspace_cone.contains(Vectord<5>{0.0, 0.0, 0.0, 1.9, 1.0}));
    STATIC_REQUIRE(
        halfspace_cone.contains(Vectord<5>{0.5, 2.0, 2.0, 1.0, 1.0}));
    STATIC_REQUIRE(
        !halfspace_cone.contains(Vectord<5>{1.0, 1.0, 0.0, 5.0, 1.0}));
    STATIC_REQUIRE(!halfspace_cone.is_proper());
    constexpr Vectord<5> lifted_point{point(0), point(1), point(2), point(3),
                                      1.0};
    STATIC_REQUIRE(
        !halfspace_cone.less_than(lifted_point, lifted_point).has_value());
    STATIC_REQUIRE(!halfspace_cone.less_than_strict(lifted_point, lifted_point)
                        .has_value());
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

    constexpr auto halfspace_cone = halfspace.cone_view();
    STATIC_REQUIRE(halfspace_cone.contains(Vectord<4>{}));
    STATIC_REQUIRE(halfspace_cone.contains(Vectord<4>{0.0, 1.0, 0.0, 1.0}));
    STATIC_REQUIRE(halfspace_cone.contains(Vectord<4>{0.1, 1.0, 2.0, 1.0}));
    STATIC_REQUIRE(!halfspace_cone.contains(Vectord<4>{0.1, 1.0, -2.0, 1.0}));
    STATIC_REQUIRE(!halfspace_cone.is_proper());

    constexpr Vectord<4> lifted_point{point(0), point(1), point(2), 1.0};
    STATIC_REQUIRE(
        !halfspace_cone.less_than(lifted_point, lifted_point).has_value());
    STATIC_REQUIRE(!halfspace_cone.less_than_strict(lifted_point, lifted_point)
                        .has_value());
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

    constexpr auto halfspace_cone = halfspace.cone_view();
    STATIC_REQUIRE(!halfspace_cone.contains(Vectord<2>{}));
    STATIC_REQUIRE(!halfspace_cone.contains(Vectord<2>{0.1, 1.0}));
    STATIC_REQUIRE(!halfspace_cone.contains(Vectord<2>{5.0, 1.0}));
    STATIC_REQUIRE(halfspace_cone.contains(Vectord<2>{6.0, 1.0}));
    STATIC_REQUIRE(!halfspace_cone.is_proper());
    constexpr Vectord<2> lifted_point{point(0), 1.0};
    STATIC_REQUIRE(
        !halfspace_cone.less_than(lifted_point, lifted_point).has_value());
    STATIC_REQUIRE(!halfspace_cone.less_than_strict(lifted_point, lifted_point)
                        .has_value());
  }
}

TEST_CASE("Test ellipsoid set", "[convex_sets]") {
  SECTION("Quick test to check convex sets types") {
    STATIC_REQUIRE(SetLike<Ellipsoid<Matrix<double, 1, 5>, EllipsoidType::OE>>);
    STATIC_REQUIRE(SetLike<Ellipsoid<Matrix<float, 1, 2>, EllipsoidType::CE>>);
  }

  SECTION("Test open ellipsoid set") {
    constexpr auto c = Vector<double, 2>{1.5, 2.0};
    constexpr auto P = make_identity<double, 2>();
    constexpr auto ellipsoid =
        Ellipsoid<Matrix<double, 1, 2>, EllipsoidType::OE>(c, P);
    REQUIRE(ellipsoid.is_well_formed());
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

    constexpr auto I = make_identity<double, 2>();
    constexpr ProjectiveMap<Matrix<double, 2, 2>, Matrix<double, 1, 2>>
        proj_map(I, Vectord<2>{}, as_row_vector(Vectord<2>{}), 1.0);
    constexpr auto x_proj = ellipsoid.project(proj_map, c);
    STATIC_REQUIRE(x_proj.has_value());
    STATIC_REQUIRE(x_proj.value() == x_proj);
    STATIC_REQUIRE(!ellipsoid.project(proj_map, Vectord<2>{}).has_value());

    constexpr auto cone = ellipsoid.cone_view();
    STATIC_REQUIRE(!cone.contains(Vectord<3>{}));
    STATIC_REQUIRE(cone.contains(Vectord<3>{c(0), c(1), 1.0}));
    STATIC_REQUIRE(cone.contains(Vectord<3>{c(0) + 0.1, c(1) - 0.5, 1.0}));
    STATIC_REQUIRE(!cone.contains(Vectord<3>{1.5, 1.0, 1.0}));
    STATIC_REQUIRE(!cone.is_proper());
    constexpr Vectord<3> lifted_c{c(0), c(1), 1.0};
    STATIC_REQUIRE(!cone.less_than(lifted_c, lifted_c).has_value());
    STATIC_REQUIRE(!cone.less_than_strict(lifted_c, lifted_c).has_value());
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

    constexpr auto cone = ellipsoid.cone_view();
    STATIC_REQUIRE(cone.contains(Vectord<3>{c(0), c(1), 1.0}));
    STATIC_REQUIRE(cone.contains(Vectord<3>{c(0) + 0.1, c(1) - 0.5, 1.0}));
    STATIC_REQUIRE(cone.contains(Vectord<3>{1.5, 1.0, 1.0}));
    STATIC_REQUIRE(cone.is_proper());
    constexpr Vectord<3> lifted_c{c(0), c(1), 1.0};
    STATIC_REQUIRE(cone.less_than_strict(lifted_c, lifted_c).has_value());
    STATIC_REQUIRE(cone.less_than(lifted_c, lifted_c).value());
    STATIC_REQUIRE(cone.less_than_strict(lifted_c, lifted_c).has_value());
    STATIC_REQUIRE(!cone.less_than_strict(lifted_c, lifted_c).value());
  }
}

TEST_CASE("Polyhedron", "[convex_sets]") {
  SECTION("Quick test to check convex sets types") {
    STATIC_REQUIRE(
        ConvexSetLike<Polyhedron<Matrix<double, 2, 1>, HalfspaceType::OUH,
                                 Matrix<double, 2, 2>>>);
    STATIC_REQUIRE(
        ConvexSetLike<Polyhedron<Matrix<float, 2, 1>, HalfspaceType::CLH,
                                 Matrix<float, 2, 2>>>);
  }

  using Space = Matrix<double, 1, 3>;  // points in R^3
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

    constexpr auto I = make_identity<double, 3>();
    constexpr ProjectiveMap<Matrix<double, 3, 3>, Matrix<double, 1, 3>>
        proj_map(I, Vectord<3>{}, as_row_vector(Vectord<3>{}), 1.0);
    constexpr auto x_proj = poly.project(proj_map, Vectord<3>{});
    STATIC_REQUIRE(x_proj.has_value());
    STATIC_REQUIRE(x_proj.value() == x_proj);
    STATIC_REQUIRE(
        !poly.project(proj_map, Vectord<3>{1.0, 2.0, 3.0}).has_value());

    constexpr auto cone = poly.cone_view();
    STATIC_REQUIRE(cone.contains(Vector<double, 4>{}));
    STATIC_REQUIRE(!cone.contains(Vector<double, 4>{1.0, 2.0, 3.0, 1.0}));
    STATIC_REQUIRE(cone.contains(Vector<double, 4>{1.0, 0.0, 0.0, 1.0}));
  }

  SECTION("Test Open Polyhedron", "[A x < b]") {
    constexpr Polyhedron<Space, HalfspaceType::OLH, decltype(A)> poly(A, b);

    STATIC_REQUIRE(poly.is_well_formed());
    STATIC_REQUIRE(!poly.contains(Vector<double, 3>{}));
    STATIC_REQUIRE(!poly.contains(Vector<double, 3>{1.0, 2.0, 3.0}));
    STATIC_REQUIRE(!poly.contains(Vector<double, 3>{1.0, 0.0, 0.0}));
    STATIC_REQUIRE(poly.contains(Vector<double, 3>{0.1, 0.1, 0.1}));

    constexpr auto cone = poly.cone_view();
    STATIC_REQUIRE(!cone.contains(Vector<double, 4>{}));
    STATIC_REQUIRE(!cone.contains(Vector<double, 4>{1.0, 2.0, 3.0, 1.0}));
    STATIC_REQUIRE(!cone.contains(Vector<double, 4>{1.0, 0.0, 0.0, 1.0}));
    STATIC_REQUIRE(cone.contains(Vector<double, 4>{0.1, 0.1, 0.1, 1.0}));
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

    constexpr auto cone = poly.cone_view();
    STATIC_REQUIRE(cone.contains(Vector<double, 4>{}));
    STATIC_REQUIRE(cone.contains(Vector<double, 4>{1.0, 0.0, 0.0, 1.0}));
    STATIC_REQUIRE(cone.contains(Vector<double, 4>{0.0, 1.0, 0.0, 1.0}));
    STATIC_REQUIRE(cone.contains(Vector<double, 4>{0.0, 0.0, 1.0, 1.0}));
  }

  SECTION("Test Closed Polyhedron", "[A x < b, C x == d]") {
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

    constexpr auto cone = poly.cone_view();
    STATIC_REQUIRE(!cone.contains(Vector<double, 4>{}));
    STATIC_REQUIRE(!cone.contains(Vector<double, 4>{1.0, 0.0, 0.0, 1.0}));
    STATIC_REQUIRE(!cone.contains(Vector<double, 4>{0.0, 1.0, 0.0, 1.0}));
    STATIC_REQUIRE(!cone.contains(Vector<double, 4>{0.0, 0.0, 1.0, 1.0}));
    STATIC_REQUIRE(cone.contains(Vector<double, 4>{0.2, 0.3, 0.5, 1.0}));
  }
}

TEST_CASE("Test Simplex", "[convex_sets]") {
  SECTION("Quick test to check convex sets types") {
    STATIC_REQUIRE(ConvexSetLike<Simplex<Matrix<double, 1, 3>>>);
    STATIC_REQUIRE(ConvexSetLike<Simplex<Matrix<float, 1, 5>>>);
  }

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

    constexpr ProjectiveMap<Matrix<double, 3, 3>, Matrix<double, 1, 3>>
        proj_map(I, Vectord<3>{}, as_row_vector(Vectord<3>{}), 1.0);
    constexpr auto x_proj =
        simplex.project(proj_map, Vectord<3>{0.7, 0.2, 0.1});
    STATIC_REQUIRE(x_proj.has_value());
    STATIC_REQUIRE(x_proj.value() == x_proj);
    STATIC_REQUIRE(!simplex.project(proj_map, Vectord<3>{}).has_value());

    constexpr auto cone = simplex.cone_view();
    STATIC_REQUIRE(cone.contains(Vectord<4>{0.8, 0.1, 0.1, 1.0}));
    STATIC_REQUIRE(!cone.contains(Vectord<4>{-0.8, 0.1, 0.1, 1.0}));
    STATIC_REQUIRE(!cone.contains(Vectord<4>{0.8, 0.1, 0.0, 1.0}));
  }
}

TEST_CASE("Test Nonneg Orthant", "[convex_sets]") {
  SECTION("Quick test to check convex sets types") {
    STATIC_REQUIRE(ConvexSetLike<NonnegOrthant<Matrix<double, 1, 1>>>);
    STATIC_REQUIRE(ConvexConeLike<
                   NonnegOrthant<Matrix<double, 1, 1>, ViewTag::ConvexCone>>);
  }

  SECTION("Test nonneg orthant", "[R^n_+]") {
    constexpr NonnegOrthant<Matrixd<1, 2>> set;
    STATIC_REQUIRE(set.is_well_formed());
    STATIC_REQUIRE(set == set);
    STATIC_REQUIRE(set.contains(Vectord<2>{1.0, 0.0}));
    STATIC_REQUIRE(!set.contains(Vectord<2>{-0.1, 0.1}));
    constexpr auto I = make_identity<double, 2>();
    STATIC_REQUIRE(set.transform(AffineMap<Matrixd<2, 2>>(I, Vectord<2>{})) ==
                   set);
    STATIC_REQUIRE(
        !set.project(ProjectiveMap<Matrix<double, 2, 2>, Matrix<double, 1, 2>>(
                         I, Vectord<2>{}, as_row_vector(Vectord<2>{}), 0.0),
                     Vectord<2>{})
             .has_value());
    constexpr auto cone = set.cone_view();
    STATIC_REQUIRE(cone.contains(Vectord<2>{1.0, 0.0}));
    STATIC_REQUIRE(!cone.contains(Vectord<2>{-0.1, 0.1}));
  }
}

TEST_CASE("Test LMI", "[convex_sets]") {
  SECTION("Quick test to check convex sets types") {
    STATIC_REQUIRE(
        ConvexConeLike<
            LMI<Matrix<double, 1, 2>, LMIType::Strict, ViewTag::ConvexCone>>);
    STATIC_REQUIRE(ConvexConeLike<LMI<Matrix<double, 3, 1>, LMIType::NonStrict,
                                      ViewTag::ConvexCone>>);
  }

  SECTION("Test Strict LMI", "[A(x) <= B]") {
    constexpr auto I = make_identity<double, 2>();
    constexpr LMI<Matrixd<1, 2>, LMIType::Strict> set(I, I, I);
    STATIC_REQUIRE(set.is_well_formed());
    STATIC_REQUIRE(!set.contains(Vectord<2>{1.0, 2.0}));
    STATIC_REQUIRE(set.contains(Vectord<2>{}));
    STATIC_REQUIRE(set.contains(Vectord<2>{0.2, 0.5}));
    STATIC_REQUIRE(!set.contains(Vectord<2>{0.5, 0.5}));
    constexpr auto cone = set.cone_view();
    STATIC_REQUIRE(!cone.contains(Vectord<2>{1.0, 2.0}));
    STATIC_REQUIRE(cone.contains(Vectord<2>{}));
    STATIC_REQUIRE(cone.contains(Vectord<2>{0.2, 0.5}));
    STATIC_REQUIRE(!cone.contains(Vectord<2>{0.5, 0.5}));
  }

  SECTION("Test Nonstrict LMI", "[A(x) <= B]") {
    constexpr auto I = make_identity<double, 3>();
    constexpr LMI<Matrixd<1, 3>, LMIType::NonStrict> set(I, I, I, I);
    STATIC_REQUIRE(set.is_well_formed());
    STATIC_REQUIRE(!set.contains(Vectord<3>{1.0, 2.0, 3.0}));
    STATIC_REQUIRE(set.contains(Vectord<3>{}));
    STATIC_REQUIRE(set.contains(Vectord<3>{0.2, 0.5, 0.1}));
    STATIC_REQUIRE(set.contains(Vectord<3>{1. / 3., 1. / 3., 1. / 3.}));
    constexpr auto cone = set.cone_view();
    STATIC_REQUIRE(!cone.contains(Vectord<3>{1.0, 2.0, 3.0}));
    STATIC_REQUIRE(cone.contains(Vectord<3>{}));
    STATIC_REQUIRE(cone.contains(Vectord<3>{0.2, 0.5, 0.1}));
    STATIC_REQUIRE(cone.contains(Vectord<3>{1. / 3., 1. / 3., 1. / 3.}));
  }
}

TEST_CASE("PSD cone", "[convex_sets]") {
  SECTION("Quick test to check convex sets types") {
    STATIC_REQUIRE(ConvexConeLike<PsdCone<SymmetricMatrix<Matrix<double, 2, 2>>,
                                          ViewTag::ConvexCone>>);
    STATIC_REQUIRE(
        ConvexConeLike<PsdCone<SymmetricMatrix<Matrix<float, 10, 10>>,
                               ViewTag::ConvexCone>>);
  }

  SECTION("PSD Cones") {
    constexpr auto A =
        Matrixd<2, 2>{Vectord<2>{2.0, 1.0}, Vectord<2>{1.0, 2.0}};
    constexpr auto B =
        Matrixd<2, 2>{Vectord<2>{4.0, -1.0}, Vectord<2>{-1.0, 3.0}};
    constexpr auto I = make_identity<double, 2>();
    constexpr auto Z = Matrixd<2, 2>{};
    constexpr auto cone = PsdCone<SymmetricMatrix<Matrix<double, 2, 2>>>();
    STATIC_REQUIRE(cone.is_well_formed());
    STATIC_REQUIRE(cone.contains(A));
    STATIC_REQUIRE(cone.contains(B));
    STATIC_REQUIRE(cone.contains(I));
    STATIC_REQUIRE(cone.contains(Z));

    STATIC_REQUIRE(cone.transform(AffineMap<Matrix<double, 2, 2>>(
                       make_identity<double, 2>(), Vectord<2>{})) == cone);

    constexpr ProjectiveMap<Matrix<double, 2, 2>, Matrix<double, 1, 2>>
        proj_map(I, Vectord<2>{}, as_row_vector(Vectord<2>{}), 1.0);
    constexpr auto x_proj = cone.project(proj_map, Matrixd<2, 2>{});
    STATIC_REQUIRE(!x_proj.has_value());

    constexpr auto cone_v = cone.cone_view();
    STATIC_REQUIRE(cone_v.contains(A));
    STATIC_REQUIRE(cone_v.contains(B));
    STATIC_REQUIRE(cone_v.contains(I));
    STATIC_REQUIRE(cone_v.contains(Z));
  }
}

TEST_CASE("Test norm cone", "[convex_sets]") {
  SECTION("Quick test to check convex sets types") {
    STATIC_REQUIRE(VectorSpace<Matrix<double, 1, 1>>);
    STATIC_REQUIRE(
        ConvexConeLike<NormCone<Matrix<double, 1, 2>, NormConeType::LinfOC,
                                ViewTag::ConvexCone>>);
    STATIC_REQUIRE(
        ConvexConeLike<NormCone<Matrix<float, 1, 10>, NormConeType::L1CC,
                                ViewTag::ConvexCone>>);
  }

  SECTION("Test open L2 norm cone") {
    constexpr auto t = constexpr_sqrt(5.0);
    constexpr auto c = Vector<double, 2>{1.0, t};
    constexpr auto cone = NormCone<Matrix<double, 1, 2>, NormConeType::L2OC>();
    STATIC_REQUIRE(cone.is_well_formed());
    STATIC_REQUIRE(!cone.contains(Vector<double, 2>{}));
    STATIC_REQUIRE(cone.contains(c));
    STATIC_REQUIRE(!cone.contains(Vector<double, 2>{c(0) + 2.0, c(1)}));
    constexpr auto cone_v = cone.cone_view();
    STATIC_REQUIRE(!cone_v.contains(Vector<double, 2>{}));
    STATIC_REQUIRE(cone_v.contains(c));
    STATIC_REQUIRE(!cone_v.contains(Vector<double, 2>{c(0) + 2.0, c(1)}));

    constexpr auto I = make_identity<double, 2>();
    constexpr ProjectiveMap<Matrix<double, 2, 2>, Matrix<double, 1, 2>>
        proj_map(I, Vectord<2>{}, as_row_vector(Vectord<2>{}), 1.0);
    constexpr auto x_proj = cone.project(proj_map, Vectord<2>{});
    STATIC_REQUIRE(!x_proj.has_value());
  }

  SECTION("Test closed L2 norm cone") {
    constexpr auto t = constexpr_sqrt(5.f);
    constexpr auto c = Vector<float, 2>{1.f, t};
    constexpr auto cone = NormCone<Matrix<float, 1, 2>, NormConeType::L2CC>();
    STATIC_REQUIRE(cone.is_well_formed());
    STATIC_REQUIRE(cone.contains(Vector<float, 2>{}));
    STATIC_REQUIRE(cone.contains(c));
    STATIC_REQUIRE(!cone.contains(Vector<float, 2>{c(0) + 2.f, c(1)}));
    constexpr auto cone_v = cone.cone_view();
    STATIC_REQUIRE(cone_v.contains(Vector<float, 2>{}));
    STATIC_REQUIRE(cone_v.contains(c));
    STATIC_REQUIRE(!cone_v.contains(Vector<float, 2>{c(0) + 2.f, c(1)}));
  }

  SECTION("Test open L1 norm cone") {
    constexpr auto t = 3.0;
    constexpr auto c = Vector<double, 2>{1.0, t};
    constexpr auto cone = NormCone<Matrix<double, 1, 2>, NormConeType::L1OC>();
    STATIC_REQUIRE(cone.is_well_formed());
    STATIC_REQUIRE(!cone.contains(Vector<double, 2>{}));
    STATIC_REQUIRE(cone.contains(c));
    STATIC_REQUIRE(!cone.contains(Vector<double, 2>{c(0) + 2.0, c(1)}));
    constexpr auto cone_v = cone.cone_view();
    STATIC_REQUIRE(!cone_v.contains(Vector<double, 2>{}));
    STATIC_REQUIRE(cone_v.contains(c));
    STATIC_REQUIRE(!cone_v.contains(Vector<double, 2>{c(0) + 2.0, c(1)}));

    constexpr auto I = make_identity<double, 2>();
    constexpr ProjectiveMap<Matrix<double, 2, 2>, Matrix<double, 1, 2>>
        proj_map(I, Vectord<2>{}, as_row_vector(Vectord<2>{}), 1.0);
    constexpr auto x_proj = cone.project(proj_map, Vectord<2>{});
    STATIC_REQUIRE(!x_proj.has_value());
  }

  SECTION("Test closed L1 norm cone") {
    constexpr auto t = 3.f;
    constexpr auto c = Vector<float, 2>{1.f, t};
    constexpr auto cone = NormCone<Matrix<float, 1, 2>, NormConeType::L1CC>();
    STATIC_REQUIRE(cone.is_well_formed());
    STATIC_REQUIRE(cone.contains(Vector<float, 2>{}));
    STATIC_REQUIRE(cone.contains(c));
    STATIC_REQUIRE(cone.contains(Vector<float, 2>{c(0) + 2.f, c(1)}));
    STATIC_REQUIRE(!cone.contains(Vector<float, 2>{c(0) + 4.f, c(1)}));
    constexpr auto cone_v = cone.cone_view();
    STATIC_REQUIRE(cone_v.contains(Vector<float, 2>{}));
    STATIC_REQUIRE(cone_v.contains(c));
    STATIC_REQUIRE(cone_v.contains(Vector<float, 2>{c(0) + 2.f, c(1)}));
    STATIC_REQUIRE(!cone_v.contains(Vector<float, 2>{c(0) + 4.f, c(1)}));
  }

  SECTION("Test open Linf norm cone") {
    constexpr auto t = 2.0;
    constexpr auto c = Vector<double, 2>{1.0, t};
    constexpr auto cone =
        NormCone<Matrix<double, 1, 2>, NormConeType::LinfOC>();
    STATIC_REQUIRE(cone.is_well_formed());
    STATIC_REQUIRE(!cone.contains(Vector<double, 2>{}));
    STATIC_REQUIRE(cone.contains(c));
    STATIC_REQUIRE(!cone.contains(Vector<double, 2>{c(0) + 2.0, c(1)}));
    constexpr auto cone_v = cone.cone_view();
    STATIC_REQUIRE(!cone_v.contains(Vector<double, 2>{}));
    STATIC_REQUIRE(cone_v.contains(c));
    STATIC_REQUIRE(!cone_v.contains(Vector<double, 2>{c(0) + 2.0, c(1)}));

    constexpr auto I = make_identity<double, 2>();
    constexpr ProjectiveMap<Matrix<double, 2, 2>, Matrix<double, 1, 2>>
        proj_map(I, Vectord<2>{}, as_row_vector(Vectord<2>{}), 1.0);
    constexpr auto x_proj = cone.project(proj_map, Vectord<2>{});
    STATIC_REQUIRE(!x_proj.has_value());
  }

  SECTION("Test closed Linf norm cone") {
    constexpr auto t = 2.f;
    constexpr auto c = Vector<float, 2>{1.f, t};
    constexpr auto cone = NormCone<Matrix<float, 1, 2>, NormConeType::LinfCC>();
    STATIC_REQUIRE(cone.is_well_formed());
    STATIC_REQUIRE(cone.contains(Vector<float, 2>{}));
    STATIC_REQUIRE(cone.contains(c));
    STATIC_REQUIRE(cone.contains(Vector<float, 2>{c(0) + 1.f, c(1)}));
    STATIC_REQUIRE(!cone.contains(Vector<float, 2>{c(0) + 3.f, c(1)}));
    constexpr auto cone_v = cone.cone_view();
    STATIC_REQUIRE(cone_v.contains(Vector<float, 2>{}));
    STATIC_REQUIRE(cone_v.contains(c));
    STATIC_REQUIRE(cone_v.contains(Vector<float, 2>{c(0) + 1.f, c(1)}));
    STATIC_REQUIRE(!cone_v.contains(Vector<float, 2>{c(0) + 3.f, c(1)}));
  }
}

TEST_CASE("Test norm ball convex set", "[convex_sets]") {
  SECTION("Quick test to check convex sets types") {
    STATIC_REQUIRE(VectorSpace<Matrix<double, 1, 1>>);
    STATIC_REQUIRE(
        ConvexSetLike<NormBall<Matrix<double, 1, 2>, NormBallType::L2OB>>);
    STATIC_REQUIRE(
        ConvexSetLike<NormBall<Matrix<double, 1, 5>, NormBallType::L1OB>>);
    STATIC_REQUIRE(
        ConvexSetLike<NormBall<Matrix<float, 1, 10>, NormBallType::L2CB>>);
    STATIC_REQUIRE(
        ConvexSetLike<NormBall<Matrix<float, 1, 1>, NormBallType::L2OB>>);
  }

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

    constexpr auto I = make_identity<double, 2>();
    constexpr ProjectiveMap<Matrix<double, 2, 2>, Matrix<double, 1, 2>>
        proj_map(I, Vectord<2>{}, as_row_vector(Vectord<2>{}), 1.0);
    constexpr auto x_proj = ball.project(proj_map, c);
    STATIC_REQUIRE(x_proj.has_value());
    STATIC_REQUIRE(x_proj.value() == x_proj);
    STATIC_REQUIRE(!ball.project(proj_map, Vectord<2>{}).has_value());

    constexpr auto cone = ball.cone_view();
    constexpr auto lifted_c = Vectord<3>{c(0), c(1), r};
    STATIC_REQUIRE(cone.contains(Vectord<3>{c(0) + 0.1, c(1) - 0.2, r}));
    STATIC_REQUIRE(cone.contains(Vectord<3>{c(0) + 2.0, c(1), r}));
    STATIC_REQUIRE(!cone.contains(Vectord<3>{c(0) + 2.0, c(1), 1.0}));
    STATIC_REQUIRE(!cone.is_proper());
    STATIC_REQUIRE(!cone.less_than(lifted_c, lifted_c).has_value());
    STATIC_REQUIRE(!cone.less_than_strict(lifted_c, lifted_c).has_value());
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

    constexpr auto cone = ball.cone_view();
    constexpr auto lifted_c = Vectorf<3>{c(0), c(1), r};
    STATIC_REQUIRE(cone.contains(Vectorf<3>{c(0) + 0.1f, c(1) - 0.2f, 1.f}));
    STATIC_REQUIRE(cone.contains(Vectorf<3>{c(0) + 2.f, c(1), 1.f}));
    STATIC_REQUIRE(cone.is_proper());
    STATIC_REQUIRE(cone.less_than(lifted_c, lifted_c).has_value());
    STATIC_REQUIRE(cone.less_than(lifted_c, lifted_c).value());
    STATIC_REQUIRE(cone.less_than_strict(
                           lifted_c, Vectorf<3>{c(0) + 1.f, c(1) + 1.f, r * r})
                       .value());
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

    constexpr ProjectiveMap<Matrix<double, 4, 4>, Matrix<double, 1, 4>>
        proj_map(I, Vectord<4>{}, as_row_vector(Vectord<4>{}), 1.0);
    constexpr auto x_proj = ball.project(proj_map, c);
    STATIC_REQUIRE(x_proj.has_value());
    STATIC_REQUIRE(x_proj.value() == x_proj);
    STATIC_REQUIRE(!ball.project(proj_map, Vectord<4>{}).has_value());

    constexpr auto cone = ball.cone_view();
    constexpr auto lifted_c = Vectord<5>{c(0), c(1), c(2), c(3), r};
    STATIC_REQUIRE(cone.contains(Vectord<5>{}));
    STATIC_REQUIRE(
        cone.contains(Vectord<5>{c(0), c(1), c(2), c(3) + 0.1, 1.0}));
    STATIC_REQUIRE(
        cone.contains(Vectord<5>{c(0) + 1.0, c(1), c(2), c(3), 1.0}));
    STATIC_REQUIRE(cone.is_proper());
    STATIC_REQUIRE(cone.less_than(lifted_c, lifted_c).has_value());
    STATIC_REQUIRE(cone.less_than(lifted_c, lifted_c).value());
    STATIC_REQUIRE(cone.less_than_strict(lifted_c, lifted_c).has_value());
    STATIC_REQUIRE(!cone.less_than_strict(lifted_c, lifted_c).value());
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

    constexpr auto cone = ball.cone_view();
    constexpr auto lifted_c = Vectorf<5>{c(0), c(1), c(2), c(3), r};
    STATIC_REQUIRE(cone.contains(lifted_c));
    STATIC_REQUIRE(!cone.contains(Vectorf<5>{}));
    STATIC_REQUIRE(cone.contains(Vectorf<5>{c(0), c(1), c(2), c(3) + 0.1f, r}));
    STATIC_REQUIRE(!cone.contains(Vectorf<5>{c(0) + 1.f, c(1), c(2), c(3), r}));
    STATIC_REQUIRE(!cone.is_proper());
    STATIC_REQUIRE(!cone.less_than(lifted_c, lifted_c).has_value());
    STATIC_REQUIRE(!cone.less_than_strict(lifted_c, lifted_c).has_value());
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

    constexpr auto I = make_identity<double, 3>();
    constexpr ProjectiveMap<Matrix<double, 3, 3>, Matrix<double, 1, 3>>
        proj_map(I, Vectord<3>{}, as_row_vector(Vectord<3>{}), 1.0);
    constexpr auto x_proj = ball.project(proj_map, c);
    STATIC_REQUIRE(x_proj.has_value());
    STATIC_REQUIRE(x_proj.value() == x_proj);
    STATIC_REQUIRE(!ball.project(proj_map, Vectord<3>{}).has_value());

    constexpr auto cone = ball.cone_view();
    constexpr auto lifted_c = Vectord<4>{1.0, 2.0, 3.0, r};
    STATIC_REQUIRE(cone.contains(lifted_c));
    STATIC_REQUIRE(!cone.contains(Vectord<4>{}));
    STATIC_REQUIRE(cone.contains(Vectord<4>{c(0) + 1.0, c(1), c(2), r}));
    STATIC_REQUIRE(!cone.is_proper());
    STATIC_REQUIRE(!cone.less_than(lifted_c, lifted_c).has_value());
    STATIC_REQUIRE(!cone.less_than_strict(lifted_c, lifted_c).has_value());
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

    constexpr auto cone = ball.cone_view();
    constexpr auto lifted_c = Vectorf<4>{c(0), c(1), c(2), r};
    STATIC_REQUIRE(cone.contains(lifted_c));
    STATIC_REQUIRE(cone.contains(Vectorf<4>{}));
    STATIC_REQUIRE(cone.contains(Vectorf<4>{c(0) + 1.f, c(1), c(2), r}));
    STATIC_REQUIRE(cone.is_proper());
    STATIC_REQUIRE(cone.less_than(lifted_c, lifted_c).has_value());
    STATIC_REQUIRE(cone.less_than(lifted_c, lifted_c).value());
    STATIC_REQUIRE(
        cone.less_than_strict(lifted_c, Vectorf<4>{2.f, 2.f, 2.f, r * r})
            .value());
  }
}
