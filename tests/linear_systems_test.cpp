/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#include <algorithm>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "math.hpp"

using namespace topology;
using namespace algebra;
using namespace spaces;

template <typename T>
concept DefaultConstructible = requires { T{}; };

constexpr double eps = 1e-10;

TEST_CASE("Test euclidean space", "[E]") {
  using OSD2 = OpenSet<double, 2>;
  using Derived = Topology<OSD2, 2>;

  STATIC_REQUIRE(DefaultConstructible<EuclideanSpace<Derived>>);

  using VD5 = Vectord<5>;
  using Derived2 = Topology<VD5, 9>;
  STATIC_REQUIRE(DefaultConstructible<EuclideanSpace<Derived2>>);
}

TEST_CASE("Test linear system", "[y=Ax]") {
  constexpr auto x = Vectord<3>{3.f, -2.f, 1.f};
  constexpr auto A = Vector<Vectord<3>, 3>{x, x, x};
  constexpr auto y = A * x;
  STATIC_REQUIRE(y(0) == 14.f);
  STATIC_REQUIRE(y(1) == 14.f);
  STATIC_REQUIRE(y(2) == 14.f);
}

TEST_CASE("Test QR Decomposition", "[A = QR]") {
  constexpr auto x1 = Vectord<3>{1.f, 1.f, 0.f};
  constexpr auto x2 = Vectord<3>{1.f, 0.f, 1.f};
  constexpr auto x3 = Vectord<3>{0.f, 1.f, 1.f};
  constexpr auto A = Matrix<double, 3, 3>{x1, x2, x3};
  constexpr auto result = qr(A);
  constexpr auto Q = std::get<0>(result);
  constexpr auto R = std::get<1>(result);
  constexpr auto A_prime = Q * R;
  REQUIRE(A_prime(0)(0) == Catch::Approx(A(0)(1)).margin(eps));
  REQUIRE(A_prime(0)(1) == Catch::Approx(A(0)(1)).margin(eps));
  REQUIRE(A_prime(0)(2) == Catch::Approx(A(0)(2)).margin(eps));
  REQUIRE(A_prime(1)(0) == Catch::Approx(A(1)(0)).margin(eps));
  REQUIRE(A_prime(1)(1) == Catch::Approx(A(1)(1)).margin(eps));
  REQUIRE(A_prime(1)(2) == Catch::Approx(A(1)(2)).margin(eps));
  REQUIRE(A_prime(2)(0) == Catch::Approx(A(2)(0)).margin(eps));
  REQUIRE(A_prime(2)(1) == Catch::Approx(A(2)(1)).margin(eps));
  REQUIRE(A_prime(2)(2) == Catch::Approx(A(2)(2)).margin(eps));

  STATIC_REQUIRE(R(1)(0) == 0);
  STATIC_REQUIRE(R(2)(0) == 0);
  STATIC_REQUIRE(R(2)(0) == 0);
  STATIC_REQUIRE(R(2)(1) == 0);

  constexpr auto QTQ = transpose(Q) * Q;
  REQUIRE(QTQ(0)(0) == Catch::Approx(1).margin(eps));
  REQUIRE(QTQ(0)(1) == Catch::Approx(0).margin(eps));
  REQUIRE(QTQ(0)(2) == Catch::Approx(0).margin(eps));
  REQUIRE(QTQ(1)(0) == Catch::Approx(0).margin(eps));
  REQUIRE(QTQ(1)(1) == Catch::Approx(1).margin(eps));
  REQUIRE(QTQ(1)(2) == Catch::Approx(0).margin(eps));
  REQUIRE(QTQ(2)(0) == Catch::Approx(0).margin(eps));
  REQUIRE(QTQ(2)(1) == Catch::Approx(0).margin(eps));
  REQUIRE(QTQ(2)(2) == Catch::Approx(1).margin(eps));
}
