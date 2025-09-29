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
