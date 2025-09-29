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
    STATIC_REQUIRE(SetLike<Halfspace<Matrix<double, 1, 2>>>);
    STATIC_REQUIRE(SetLike<Halfspace<Matrix<float, 1, 10>>>);
  }

  SECTION("Test closed lower halfspace") {
    constexpr Vector<double, 2> point{1.0, 2.0};
    constexpr double b = 1.0;
    constexpr Halfspace<Matrix<double, 1, 2>> halfspace(point, b,
                                                        HalfspaceType::CLH);
    STATIC_REQUIRE(halfspace.is_valid());
    STATIC_REQUIRE(halfspace.contains(Vectord<2>{}));
    STATIC_REQUIRE(halfspace.contains(Vectord<2>{-1.0, 0.0}));
    STATIC_REQUIRE(halfspace.contains(Vectord<2>{-1.0, -2.0}));
    STATIC_REQUIRE(halfspace.contains(Vectord<2>{1.0, 0.0}));
    STATIC_REQUIRE(!halfspace.contains(Vectord<2>{1.0, 2.0}));
    STATIC_REQUIRE(!halfspace.contains(Vectord<2>{-1.0, 5.0}));
  }

  SECTION("Test open lower halfspace") {
    constexpr Vector<double, 4> point{2.0, 1.0, -2.0, 5.0};
    constexpr double b = 10.0;
    constexpr auto halfspace =
        Halfspace<Matrix<double, 1, 4>>(point, b, HalfspaceType::OLH);

    STATIC_REQUIRE(halfspace.is_valid());
    STATIC_REQUIRE(halfspace.contains(Vectord<4>{}));
    STATIC_REQUIRE(!halfspace.contains(Vectord<4>{0.0, 0.0, 0.0, 2.0}));
    STATIC_REQUIRE(halfspace.contains(Vectord<4>{0.0, 0.0, 0.0, 1.9}));
    STATIC_REQUIRE(halfspace.contains(Vectord<4>{0.5, 2.0, 2.0, 1.0}));
    STATIC_REQUIRE(!halfspace.contains(Vectord<4>{1.0, 1.0, 0.0, 5.0}));
  }

  SECTION("Test closed upper halfspace") {
    constexpr Vector<double, 3> point{1.0, 2.0, 1.0};
    constexpr double b = 2.0;
    constexpr Halfspace<Matrix<double, 1, 3>> halfspace(point, b,
                                                        HalfspaceType::CUH);
    STATIC_REQUIRE(halfspace.is_valid());
    STATIC_REQUIRE(!halfspace.contains(Vectord<3>{}));
    STATIC_REQUIRE(halfspace.contains(Vectord<3>{0.0, 1.0, 0.0}));
    STATIC_REQUIRE(halfspace.contains(Vectord<3>{0.1, 1.0, 2.0}));
    STATIC_REQUIRE(!halfspace.contains(Vectord<3>{0.1, 1.0, -2.0}));
  }

  SECTION("Test open upper halfspace") {
    constexpr Vector<double, 1> point{1.0};
    constexpr double b = 5.0;
    constexpr Halfspace<Matrix<double, 1, 1>> halfspace(point, b,
                                                        HalfspaceType::OUH);
    STATIC_REQUIRE(halfspace.is_valid());
    STATIC_REQUIRE(!halfspace.contains(Vectord<1>{}));
    STATIC_REQUIRE(!halfspace.contains(Vectord<1>{0.1}));
    STATIC_REQUIRE(!halfspace.contains(Vectord<1>{5.0}));
    STATIC_REQUIRE(halfspace.contains(Vectord<1>{6.0}));
  }
}

TEST_CASE("Test norm ball convex set", "[convex_sets]") {
  SECTION("Quick test to check convex sets types") {
    STATIC_REQUIRE(VectorSpace<Matrix<double, 1, 1>>);
    STATIC_REQUIRE(SetLike<NormBall<Matrix<double, 1, 2>>>);
    STATIC_REQUIRE(SetLike<NormBall<Matrix<std::complex<double>, 1, 5>>>);
    STATIC_REQUIRE(SetLike<NormBall<Matrix<float, 1, 10>>>);
    STATIC_REQUIRE(SetLike<NormBall<Matrix<std::complex<float>, 1, 1>>>);
  }

  SECTION("Test open L2 norm ball") {
    constexpr auto c = Vector<double, 2>{1.0, 3.0};
    constexpr auto r = 2.0;
    constexpr auto ball =
        NormBall<Matrix<double, 1, 2>>(c, r, NormBallType::L2OB);
    STATIC_REQUIRE(ball.is_valid());
    STATIC_REQUIRE(!ball.contains(Vector<double, 2>{}));
    STATIC_REQUIRE(ball.contains(c));
    STATIC_REQUIRE(ball.contains(Vector<double, 2>{c(0) + 0.1, c(1) - 0.2}));
    STATIC_REQUIRE(!ball.contains(Vector<double, 2>{c(0) + 2.0, c(1)}));
  }

  SECTION("Test closed L2 norm ball") {
    constexpr auto c = Vector<float, 2>{1.f, 3.f};
    constexpr auto r = 2.f;
    constexpr auto ball =
        NormBall<Matrix<float, 1, 2>>(c, r, NormBallType::L2CB);
    STATIC_REQUIRE(ball.is_valid());
    STATIC_REQUIRE(!ball.contains(Vector<float, 2>{}));
    STATIC_REQUIRE(ball.contains(c));
    STATIC_REQUIRE(ball.contains(Vector<float, 2>{c(0) + 0.1f, c(1) - 0.2f}));
    STATIC_REQUIRE(ball.contains(Vector<float, 2>{c(0) + 2.f, c(1)}));
  }

  SECTION("Test open L1 norm ball") {
    constexpr auto c = Vector<std::complex<double>, 4>{0.0, 1.2, -0.8, 2.0};
    constexpr auto r = 1.0;
    constexpr auto ball =
        NormBall<Matrix<std::complex<double>, 1, 4>>(c, r, NormBallType::L1OB);
    STATIC_REQUIRE(ball.is_valid());
    STATIC_REQUIRE(ball.contains(c));
    STATIC_REQUIRE(!ball.contains(Vectorcd<4>{}));
    STATIC_REQUIRE(ball.contains(
        Vectorcd<4>{c(0), c(1), c(2), c(3) + std::complex<double>{0.1, 0.2}}));
    STATIC_REQUIRE(!ball.contains(
        Vectorcd<4>{c(0) + std::complex<double>(1.0), c(1), c(2), c(3)}));
  }

  SECTION("Test closed L1 norm ball") {
    constexpr auto c = Vector<std::complex<float>, 4>{0.f, 1.2f, -0.8f, 2.f};
    constexpr auto r = 1.f;
    constexpr auto ball =
        NormBall<Matrix<std::complex<float>, 1, 4>>(c, r, NormBallType::L1CB);
    STATIC_REQUIRE(ball.is_valid());
    STATIC_REQUIRE(ball.contains(c));
    STATIC_REQUIRE(!ball.contains(Vectorcf<4>{}));
    STATIC_REQUIRE(ball.contains(
        Vectorcf<4>{c(0), c(1), c(2), c(3) + std::complex<float>{0.1f, 0.2f}}));
    STATIC_REQUIRE(ball.contains(
        Vectorcf<4>{c(0) + std::complex<float>(1.f), c(1), c(2), c(3)}));
  }

  SECTION("Test open Linf norm ball") {
    constexpr auto c = Vector<double, 3>{1.0, 2.0, 3.0};
    constexpr auto r = 2.0;
    constexpr auto ball =
        NormBall<Matrix<double, 1, 3>>(c, r, NormBallType::LinfOB);
    STATIC_REQUIRE(ball.is_valid());
    STATIC_REQUIRE(ball.contains(c));
    STATIC_REQUIRE(!ball.contains(Vectord<3>{}));
    STATIC_REQUIRE(ball.contains(Vectord<3>{c(0) + 1.0, c(1), c(2)}));
  }

  SECTION("Test closed  Linf norm ball") {
    constexpr auto c = Vector<float, 3>{1.f, 2.f, 3.f};
    constexpr auto r = 2.f;
    constexpr auto ball =
        NormBall<Matrix<float, 1, 3>>(c, r, NormBallType::LinfCB);
    STATIC_REQUIRE(ball.is_valid());
    STATIC_REQUIRE(ball.contains(c));
    STATIC_REQUIRE(!ball.contains(Vectorf<3>{}));
    STATIC_REQUIRE(ball.contains(Vectorf<3>{c(0) + 1.f, c(1), c(2)}));
  }
}

TEST_CASE("Test ellipsoid set", "[convex_sets]") {
  // (dakre) TODO current issue with eig decomp and static compile time checking
  // effecting this
#if 0
  SECTION("Quick test to check convex sets types") {
    STATIC_REQUIRE(SetLike<Ellipsoid<Matrix<double, 1, 5>>>);
    STATIC_REQUIRE(SetLike<Ellipsoid<Matrix<float, 1, 1>>>);
  }
#endif

  SECTION("Test open ellipsoid set") {
    constexpr auto c = Vector<double, 2>{1.5, 2.0};
    constexpr auto P = make_identity<double, 2>();
    constexpr auto ellipsoid =
        Ellipsoid<Matrix<double, 1, 2>>(c, P, EllipsoidType::OE);
    REQUIRE(ellipsoid.is_valid());
    STATIC_REQUIRE(!ellipsoid.contains(Vectord<2>{}));
    STATIC_REQUIRE(ellipsoid.contains(c));
    STATIC_REQUIRE(ellipsoid.contains(Vectord<2>{c(0) + 0.1, c(1) - 0.5}));
    STATIC_REQUIRE(!ellipsoid.contains(Vectord<2>{1.5, 1.0}));
  }

  SECTION("Test closed ellipsoid set") {
    constexpr auto c = Vector<double, 2>{1.5, 2.0};
    constexpr auto P = make_identity<double, 2>();
    constexpr auto ellipsoid =
        Ellipsoid<Matrix<double, 1, 2>>(c, P, EllipsoidType::CE);
    REQUIRE(ellipsoid.is_valid());
    STATIC_REQUIRE(!ellipsoid.contains(Vectord<2>{}));
    STATIC_REQUIRE(ellipsoid.contains(c));
    STATIC_REQUIRE(ellipsoid.contains(Vectord<2>{c(0) + 0.1, c(1) - 0.5}));
    STATIC_REQUIRE(ellipsoid.contains(Vectord<2>{1.5, 1.0}));
  }
}

TEST_CASE("Test norm cone", "[convex_sets]") {
  SECTION("Quick test to check convex sets types") {
    STATIC_REQUIRE(VectorSpace<Matrix<double, 1, 1>>);
    STATIC_REQUIRE(SetLike<NormCone<Matrix<double, 1, 2>>>);
    STATIC_REQUIRE(SetLike<NormCone<Matrix<std::complex<double>, 1, 5>>>);
    STATIC_REQUIRE(SetLike<NormCone<Matrix<float, 1, 10>>>);
    STATIC_REQUIRE(SetLike<NormCone<Matrix<std::complex<float>, 1, 1>>>);
  }

  SECTION("Test open L2 norm cone") {
    constexpr auto c = Vector<double, 2>{1.0, 2.0};
    constexpr auto t = constexpr_sqrt(5.0);
    constexpr auto cone = NormCone<Matrix<double, 1, 2>>(t, NormBallType::L2OB);
    STATIC_REQUIRE(cone.is_valid());
    STATIC_REQUIRE(cone.contains(Vector<double, 2>{}));
    STATIC_REQUIRE(!cone.contains(c));
    STATIC_REQUIRE(!cone.contains(Vector<double, 2>{c(0) + 2.0, c(1)}));
  }

  SECTION("Test closed L2 norm ball") {
    constexpr auto c = Vector<float, 2>{1.f, 2.f};
    constexpr auto t = constexpr_sqrt(5.f);
    constexpr auto cone = NormCone<Matrix<float, 1, 2>>(t, NormBallType::L2CB);
    STATIC_REQUIRE(cone.is_valid());
    STATIC_REQUIRE(cone.contains(Vector<float, 2>{}));
    STATIC_REQUIRE(cone.contains(c));
    STATIC_REQUIRE(!cone.contains(Vector<float, 2>{c(0) + 2.f, c(1)}));
  }

  SECTION("Test open L1 norm cone") {
    constexpr auto c = Vector<double, 2>{1.0, 2.0};
    constexpr auto t = 3.0;
    constexpr auto cone = NormCone<Matrix<double, 1, 2>>(t, NormBallType::L1OB);
    STATIC_REQUIRE(cone.is_valid());
    STATIC_REQUIRE(cone.contains(Vector<double, 2>{}));
    STATIC_REQUIRE(!cone.contains(c));
    STATIC_REQUIRE(!cone.contains(Vector<double, 2>{c(0) + 2.0, c(1)}));
  }

  SECTION("Test closed L1 norm ball") {
    constexpr auto c = Vector<float, 2>{1.f, 2.f};
    constexpr auto t = 3.f;
    constexpr auto cone = NormCone<Matrix<float, 1, 2>>(t, NormBallType::L1CB);
    STATIC_REQUIRE(cone.is_valid());
    STATIC_REQUIRE(cone.contains(Vector<float, 2>{}));
    STATIC_REQUIRE(cone.contains(c));
    STATIC_REQUIRE(!cone.contains(Vector<float, 2>{c(0) + 2.f, c(1)}));
  }

  SECTION("Test open Linf norm cone") {
    constexpr auto c = Vector<double, 2>{1.0, 2.0};
    constexpr auto t = 2.0;
    constexpr auto cone =
        NormCone<Matrix<double, 1, 2>>(t, NormBallType::LinfOB);
    STATIC_REQUIRE(cone.is_valid());
    STATIC_REQUIRE(cone.contains(Vector<double, 2>{}));
    STATIC_REQUIRE(!cone.contains(c));
    STATIC_REQUIRE(!cone.contains(Vector<double, 2>{c(0) + 2.0, c(1)}));
  }

  SECTION("Test closed Linf norm cone") {
    constexpr auto c = Vector<float, 2>{1.f, 2.f};
    constexpr auto t = 2.f;
    constexpr auto cone =
        NormCone<Matrix<float, 1, 2>>(t, NormBallType::LinfCB);
    STATIC_REQUIRE(cone.is_valid());
    STATIC_REQUIRE(cone.contains(Vector<float, 2>{}));
    STATIC_REQUIRE(cone.contains(c));
    STATIC_REQUIRE(!cone.contains(Vector<float, 2>{c(0) + 2.f, c(1)}));
  }
}

TEST_CASE("PSD cone", "[convex_sets]") {
  SECTION("PSD Cones") {
    //constexpr auto A =
    //    Matrixd<2, 2>{Vectord<2>{2.0, 1.0}, Vectord<2>{1.0, 2.0}};
    //constexpr auto B =
    //    Matrixd<2, 2>{Vectord<2>{4.0, -1.0}, Vectord<2>{-1.0, 3.0}};
    //constexpr auto I = make_identity<double, 3>();
    //constexpr auto Z = Matrixd<2, 2>{};
    constexpr auto cone = PsdCone<Matrix<double, 2, 2>>();
    REQUIRE(cone.is_valid());
#if 0
    REQUIRE(cone.contains(A));
    REQUIRE(cone.contains(B));
    REQUIRE(cone.contains(I));
    REQUIRE(cone.contains(Z));
#endif
  }
}
