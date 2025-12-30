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
    constexpr auto x = Variable<PosVarMeta<Space>>();
    constexpr auto I = make_identity<double, 3>();
    constexpr auto affine_set =
        AffineSubspace<AffineMeta<Space, Space>, decltype(x)>(I, Vectord<3>{},
                                                              x);
    STATIC_REQUIRE(affine_set.is_well_formed());
    STATIC_REQUIRE(affine_set.contains(Vectord<3>{}));
    STATIC_REQUIRE(!affine_set.contains(Vectord<3>{1.0, 1.0, 1.0}));

#if 0
    constexpr auto cone_block = affine_set.canonicalize();
    constexpr auto A = make_identity<double, 4>();
    STATIC_REQUIRE(cone_block.map == affine::AffineMap(A, Vectord<4>{}));
    constexpr auto cones = cone_block.cones;
    constexpr auto sz = std::tuple_size<decltype(cones)>::value;
    STATIC_REQUIRE(sz == 2);
    STATIC_REQUIRE(cone_block.validate());
#endif
  }
}

TEST_CASE("Test ellipsoid set", "[convex_sets]") {
  SECTION("Test ellipsoid set (implicitly closed)") {
    using Space = Matrixd<1, 2>;
    constexpr auto c = Vector<double, 2>{1.5, 2.0};
    constexpr auto P = make_identity<double, 2>();
    constexpr auto x = Variable<FreeVarMeta<Space>>();
    constexpr auto ellipsoid =
        Ellipsoid<EllipsoidMeta<Space>, decltype(x)>(c, P, x);
    REQUIRE(ellipsoid.is_well_formed());
    STATIC_REQUIRE(!ellipsoid.contains(Vectord<2>{}));
    STATIC_REQUIRE(ellipsoid.contains(c));
    STATIC_REQUIRE(ellipsoid.contains(Vectord<2>{c(0) + 0.1, c(1) - 0.5}));
    STATIC_REQUIRE(ellipsoid.contains(Vectord<2>{1.5, 1.0}));

#if 0
    constexpr auto cone_block = ellipsoid.canonicalize();
    constexpr auto F = make_identity<double, 3>();
    STATIC_REQUIRE(
        cone_block.map ==
        affine::AffineMap(F, Vectord<3>{-1.0 * c(0), -1.0 * c(1), 0.0}));

    constexpr auto cones = cone_block.cones;
    constexpr auto sz = std::tuple_size<decltype(cones)>::value;
    STATIC_REQUIRE(sz == 1);
    STATIC_REQUIRE(cone_block.validate());
#endif
  }
}

TEST_CASE("Test halfspace convex set", "[convex_sets]") {
  SECTION("Test hyperplane") {
    using Space = Matrixd<1, 3>;
    constexpr Vector<double, 3> point{0.1, 0.2, 0.3};
    constexpr auto x = Variable<FreeVarMeta<Space>>();
    constexpr double b = 0.6;
    constexpr Hyperplane<HyperplaneMeta<Space>, decltype(x)> hyperplane(point,
                                                                        b, x);
    STATIC_REQUIRE(hyperplane.is_well_formed());
    STATIC_REQUIRE(!hyperplane.contains(Vectord<3>{}));
    STATIC_REQUIRE(hyperplane.contains(Vectord<3>{1.0, 1.0, 1.0}));
    STATIC_REQUIRE(!hyperplane.contains(Vectord<3>{1.0, 1.0, 2.0}));

    constexpr auto I = make_identity<double, 3>();
    constexpr auto hyperplane_id =
        hyperplane.transform(AffineMap<Matrix<double, 3, 3>>(I, Vectord<3>{}));
    STATIC_REQUIRE(hyperplane_id == hyperplane);

#if 0
    constexpr auto cone_block = hyperplane.canonicalize();
    constexpr mat_traits<Matrix<double, 2, 4>> A_mat;
    constexpr auto A =
        A_mat(point(0), point(1), point(2), -1 * b, 0.0, 0.0, 0.0, 1.0);

    STATIC_REQUIRE(cone_block.map == affine::AffineMap(A, Vectord<2>{}));

    constexpr auto cones = cone_block.cones;
    constexpr auto sz = std::tuple_size<decltype(cones)>::value;
    STATIC_REQUIRE(sz == 2);
    STATIC_REQUIRE(cone_block.validate());
#endif
  }

  SECTION("Test closed lower halfspace") {
    using Space = Matrixd<1, 2>;
    constexpr Vector<double, 2> point{1.0, 2.0};
    constexpr auto x = Variable<FreeVarMeta<Space>>();
    constexpr double b = 1.0;
    constexpr Halfspace<HalfspaceMeta<Space>, decltype(x), HalfspaceType::LH>
        halfspace(point, b, x);
    STATIC_REQUIRE(halfspace.is_well_formed());
    STATIC_REQUIRE(halfspace.contains(Vectord<2>{}));
    STATIC_REQUIRE(halfspace.contains(Vectord<2>{-1.0, 0.0}));
    STATIC_REQUIRE(halfspace.contains(Vectord<2>{-1.0, -2.0}));
    STATIC_REQUIRE(halfspace.contains(Vectord<2>{1.0, 0.0}));
    STATIC_REQUIRE(!halfspace.contains(Vectord<2>{1.0, 2.0}));
    STATIC_REQUIRE(!halfspace.contains(Vectord<2>{-1.0, 5.0}));

#if 0
    constexpr auto cone_block = halfspace.canonicalize();
    constexpr mat_traits<Matrix<double, 2, 3>> A_mat;
    constexpr auto A = A_mat(-1 * point(0), -1 * point(1), b, 0.0, 0.0, 1.0);

    STATIC_REQUIRE(cone_block.map == affine::AffineMap(A, Vectord<2>{}));

    constexpr auto cones = cone_block.cones;
    constexpr auto sz = std::tuple_size<decltype(cones)>::value;
    STATIC_REQUIRE(sz == 2);
    STATIC_REQUIRE(cone_block.validate());
#endif
  }

  SECTION("Test closed upper halfspace") {
    using Space = Matrixd<1, 3>;
    constexpr Vector<double, 3> point{1.0, 2.0, 1.0};
    constexpr auto x = Variable<FreeVarMeta<Space>>();
    constexpr double b = 2.0;
    constexpr Halfspace<HalfspaceMeta<Space>, decltype(x), HalfspaceType::UH>
        halfspace(point, b, x);
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

#if 0
    constexpr auto cone_block = halfspace.canonicalize();
    constexpr mat_traits<Matrix<double, 2, 4>> A_mat;
    constexpr auto A = A_mat(-1 * point(0), -1 * point(1), -1 * point(2), b,
                             0.0, 0.0, 0.0, 1.0);

    STATIC_REQUIRE(cone_block.map == affine::AffineMap(A, Vectord<2>{}));

    constexpr auto cones = cone_block.cones;
    constexpr auto sz = std::tuple_size<decltype(cones)>::value;
    STATIC_REQUIRE(sz == 2);
    STATIC_REQUIRE(cone_block.validate());
#endif
  }
}

TEST_CASE("Test LMI", "[convex_sets]") {
  SECTION("Test LMI", "[A(x) <= B]") {
    using Space = linalg::Matrixd<1, 3>;
    constexpr auto I = make_identity<double, 3>();
    constexpr auto x = Variable<FreeVarMeta<Space>>();
    constexpr linalg::Vector<linalg::Matrixd<3, 3>, 3> A{I, I, I};
    constexpr LMI<LMIMeta<Space>, decltype(x)> set(A, I, x);
    STATIC_REQUIRE(set.is_well_formed());
    STATIC_REQUIRE(!set.contains(Vectord<3>{1.0, 2.0, 3.0}));
    STATIC_REQUIRE(set.contains(Vectord<3>{}));
    STATIC_REQUIRE(set.contains(Vectord<3>{0.2, 0.5, 0.1}));
    STATIC_REQUIRE(set.contains(Vectord<3>{1. / 3., 1. / 3., 1. / 3.}));

#if 0
    constexpr auto cone_block = set.canonicalize();
    constexpr mat_traits<Matrix<double, 7, 4>> F_mat;
    constexpr auto F = F_mat(-1.0, -1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0,
                             -1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, -1.0, -1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    STATIC_REQUIRE(cone_block.map == affine::AffineMap(F, Vectord<7>{}));

    constexpr auto cones = cone_block.cones;
    constexpr auto sz = std::tuple_size<decltype(cones)>::value;
    STATIC_REQUIRE(sz == 2);
    STATIC_REQUIRE(cone_block.validate());
#endif
  }
}

TEST_CASE("Test Simplex", "[convex_sets]") {
  SECTION("Test Simplex", "[{ x | x >= 0, sum_i x_i = 1}]") {
    using Space = Matrixd<1, 3>;
    constexpr auto x = Variable<FreeVarMeta<Space>>();
    constexpr Simplex<SimplexMeta<Space>, decltype(x)> simplex(x);
    STATIC_REQUIRE(simplex.is_well_formed());
    STATIC_REQUIRE(simplex == simplex);
    STATIC_REQUIRE(simplex.contains(Vectord<3>{0.8, 0.1, 0.1}));
    STATIC_REQUIRE(!simplex.contains(Vectord<3>{-0.8, 0.1, 0.1}));
    STATIC_REQUIRE(!simplex.contains(Vectord<3>{0.8, 0.1, 0.0}));

    /// (dakre) TODO- should this be an intersection betwen poly & affine
    /// subspace
    constexpr auto I = make_identity<double, 3>();
    constexpr auto poly =
        simplex.transform(AffineMap<Matrixd<3, 3>>(I, Vectord<3>{}));
    STATIC_REQUIRE(poly.contains(Vectord<3>{0.8, 0.1, 0.1}));
    STATIC_REQUIRE(!poly.contains(Vectord<3>{-0.8, 0.1, 0.1}));
    // STATIC_REQUIRE(!poly.contains(Vectord<3>{0.8, 0.1, 0.0}));

#if 0
    constexpr auto cone_block = simplex.canonicalize();
    constexpr mat_traits<Matrix<double, 4, 4>> A_mat;
    constexpr auto A = A_mat(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                             1.0, 0.0, -1.0, -1.0, -1.0, 1.0);

    STATIC_REQUIRE(cone_block.map == affine::AffineMap(A, Vectord<4>{}));

    constexpr auto cones = cone_block.cones;
    constexpr auto sz = std::tuple_size<decltype(cones)>::value;
    STATIC_REQUIRE(sz == 2);
    STATIC_REQUIRE(cone_block.validate());
#endif
  }
}

TEST_CASE("Polyhedron", "[convex_sets]") {
  SECTION("Test Closed Polyhedron", "[A x <= b]") {
    using Space = Matrix<double, 1, 3>;
    mat_traits<Matrix<double, 7, 3>> A_mat;
    constexpr auto A =
        A_mat(1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0,
              0.0, 1.0, 0.0, 0.0, -1.0, 1.0, 1.0, 0.0);
    STATIC_REQUIRE(mat_rows_v<decltype(A)> == 7);
    STATIC_REQUIRE(mat_cols_v<decltype(A)> == 3);
    using Dom = Matrixd<1, 3>;
    using Codom = Matrixd<1, 7>;
    constexpr auto x = Variable<FreeVarMeta<Space>>();
    constexpr Vector<double, 7> b{1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0};
    constexpr Polyhedron<PolyhedronMeta<Dom, Codom>, decltype(x),
                         HalfspaceType::LH>
        poly(A, b, x);

    STATIC_REQUIRE(poly.is_well_formed());
    STATIC_REQUIRE(poly.contains(Vector<double, 3>{}));
    STATIC_REQUIRE(!poly.contains(Vector<double, 3>{1.0, 2.0, 3.0}));
    STATIC_REQUIRE(poly.contains(Vector<double, 3>{1.0, 0.0, 0.0}));

#if 0
    constexpr auto cone_block = poly.canonicalize();

    constexpr mat_traits<Matrix<double, 2, 4>> P_mat;
    constexpr auto P = P_mat(-1.0, 0.0, 0.0, b(0), 0.0, 0.0, 0.0, 1.0);

    constexpr auto F_slice =
        linalg::get_block<0, 2, 0, 4>(cone_block.map.linear_part());
    STATIC_REQUIRE(linalg::all_equal(F_slice, P));

    constexpr auto cones = cone_block.cones;
    constexpr auto sz = std::tuple_size<decltype(cones)>::value;
    STATIC_REQUIRE(sz == 14);
    STATIC_REQUIRE(cone_block.validate());
#endif
  }
}

TEST_CASE("Test norm ball convex set", "[convex_sets]") {
  SECTION("Test closed L2 norm ball") {
    using Space = Matrixf<1, 2>;
    constexpr auto c = Vector<float, 2>{1.f, 3.f};
    constexpr auto r = 2.f;
    constexpr auto x = Variable<FreeVarMeta<Space>>();
    constexpr auto ball =
        NormBall<NormBallMeta<Space>, decltype(x), NormBallType::L2>(c, r, x);
    STATIC_REQUIRE(ball.is_well_formed());
    STATIC_REQUIRE(!ball.contains(Vector<float, 2>{}));
    STATIC_REQUIRE(ball.contains(c));
    STATIC_REQUIRE(ball.contains(Vector<float, 2>{c(0) + 0.1f, c(1) - 0.2f}));
    STATIC_REQUIRE(ball.contains(Vector<float, 2>{c(0) + 2.f, c(1)}));

    constexpr auto P = r * r * make_identity<float, 2>();
    constexpr auto ellipse =
        Ellipsoid<EllipsoidMeta<Matrix<float, 1, 2>>, decltype(x)>(c, P, x);
    constexpr auto res = ball.transform(AffineMap<Matrix<float, 2, 2>>(
        make_identity<float, 2>(), Vectorf<2>{}));
    STATIC_REQUIRE(res == ellipse);

#if 0
    constexpr mat_traits<Matrix<float, 3, 3>> F_mat;
    constexpr auto F = F_mat(1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, r);
    constexpr linalg::Vector<float, 3> g{-1 * c(0), -1 * c(1), 0.f};
    constexpr auto cone_block = ball.canonicalize();
    STATIC_REQUIRE(cone_block.map == affine::AffineMap(F, g));
    constexpr auto cones = cone_block.cones;
    constexpr auto sz = std::tuple_size<decltype(cones)>::value;
    STATIC_REQUIRE(sz == 2);
    STATIC_REQUIRE(cone_block.validate());
#endif
  }

  SECTION("Test closed L1 norm ball") {
    using Space = Matrixd<1, 4>;
    constexpr auto c = Vector<double, 4>{0.0, 1.2, -0.8, 2.0};
    constexpr auto r = 1.0;
    constexpr auto x = Variable<FreeVarMeta<Space>>();
    constexpr auto ball =
        NormBall<NormBallMeta<Space>, decltype(x), NormBallType::L1>(c, r, x);
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

#if 0
    constexpr mat_traits<Matrix<double, 9, 9>> F_mat;
    constexpr auto F = F_mat(
        1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0.,
        0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0.,
        -1., 0., 0., 0., 1., 0., 0., 0., 0., 0., -1., 0., 0., 0., 1., 0., 0.,
        0., 0., 0., -1., 0., 0., 0., 1., 0., 0., 0., 0., 0., -1., 0., 0., 0.,
        1., 0., 0., 0., 0., 0., -1., -1., -1., -1., r);

    constexpr linalg::Vector<double, 9> g{
        -1 * c(0), -1 * c(1), -1 * c(2), -1 * c(3), c(0), c(1), c(2), c(3), 0.};

    constexpr auto cone_block = ball.canonicalize();
    STATIC_REQUIRE(cone_block.map == affine::AffineMap(F, g));
    constexpr auto cones = cone_block.cones;
    constexpr auto sz = std::tuple_size<decltype(cones)>::value;
    STATIC_REQUIRE(sz == 3);
    STATIC_REQUIRE(cone_block.validate());
#endif
  }

  SECTION("Test closed  Linf norm ball") {
    using Space = Matrixf<1, 3>;
    constexpr auto c = Vector<float, 3>{1.f, 2.f, 3.f};
    constexpr auto r = 2.f;
    constexpr auto x = Variable<FreeVarMeta<Space>>();
    constexpr auto ball =
        NormBall<NormBallMeta<Space>, decltype(x), NormBallType::Linf>(c, r, x);
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

#if 0
    constexpr mat_traits<Matrix<float, 6, 4>> F_mat;
    constexpr auto F =
        F_mat(1.f, 0.f, 0.f, r, 0.f, 1.f, 0.f, r, 0.f, 0.f, 1.f, r, -1.f, 0.f,
              0.f, r, 0.f, -1.f, 0.f, r, 0.f, 0.f, -1.f, r);
    constexpr linalg::Vector<float, 6> g{-1 * c(0), -1 * c(1), -1 * c(2),
                                         c(0),      c(1),      c(2)};
    constexpr auto cone_block = ball.canonicalize();
    STATIC_REQUIRE(cone_block.map == affine::AffineMap(F, g));
    constexpr auto cones = cone_block.cones;
    constexpr auto sz = std::tuple_size<decltype(cones)>::value;
    STATIC_REQUIRE(sz == 2);
    STATIC_REQUIRE(cone_block.validate());
#endif
  }
}
