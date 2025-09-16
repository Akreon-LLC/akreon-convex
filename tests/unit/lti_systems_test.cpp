/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#include <algorithm>
#include <catch2/catch_test_macros.hpp>

#include "controls.hpp"
#include "unit_test_helpers.hpp"

using namespace lti;
using namespace linalg;

TEST_CASE("Test DLTI System", "[dot(x) = Ax + Bu]") {
  SECTION("Unstable system but reachable") {
    constexpr Vector<double, 2> r1{1.0, 2.0};
    constexpr Vector<double, 2> r2{3.0, 4.0};
    constexpr Matrix<double, 2, 2> A{r1, r2};
    constexpr Vector<double, 1> a{1.0};
    constexpr Vector<double, 1> b{2.0};
    constexpr Matrix<double, 2, 1> B{a, b};
    constexpr Matrix<double, 1, 2> C{r1};

    constexpr auto ss = LTIDiscreteSystem(A, B, C);
    STATIC_REQUIRE(ss.is_stable() == Stability::Unstable);
    STATIC_REQUIRE(ss.is_reachable());
  }

  SECTION("Unstable and unreachable") {
    Matrix<double, 3, 3> A;
    A[0][0] = 2, A[1][1] = -1, A[2][2] = 3;
    Matrix<double, 3, 1> B;
    B[1][0] = 1;
    auto ss = LTIDiscreteSystem(A, B, make_identity<double, 3>());

    REQUIRE(ss.is_stable() == Stability::Unstable);
    REQUIRE(!ss.is_reachable());
  }

  SECTION("Marginally stable autonomous system that's not reachable") {
    constexpr auto A = make_identity<double, 3>();
    constexpr auto C = make_identity<double, 3>();
    constexpr auto ss = LTIDiscreteSystem(A, Matrix<double, 3, 0>{}, C);

    // (dakre) TODO similar condition as before with complex value evaluated at
    // compile time
    REQUIRE(ss.is_stable() == Stability::Marginal);
    // (dakre) \lambda I - A = [0 0] rank 0 => not reachable
    REQUIRE(!ss.is_reachable());

    constexpr auto dot_x = ss.step(Vectord<3>{1.0, 2.0, 3.0});
    STATIC_REQUIRE(dot_x(0) == 1.0);
    STATIC_REQUIRE(dot_x(1) == 2.0);
    STATIC_REQUIRE(dot_x(2) == 3.0);

    constexpr double h = 0.1;
    constexpr auto y = ss.output(h * dot_x);
    STATIC_REQUIRE(constexpr_abs(y(0) - 0.1) < unit_test::eps);
    STATIC_REQUIRE(constexpr_abs(y(1) - 0.2) < unit_test::eps);
    STATIC_REQUIRE(constexpr_abs(y(2) - 0.3) < unit_test::eps);
  }

  SECTION("Marginally stable and reachable") {
    const auto A = make_identity<double, 3>();
    const auto B = make_identity<double, 3>();
    auto ss = LTIDiscreteSystem(A, B, make_identity<double, 3>());
    REQUIRE(ss.is_stable() == Stability::Marginal);
    REQUIRE(ss.is_reachable());
  }

  SECTION("Asymptotically stable system that's not reachable") {
    Matrix<double, 2, 2> A;
    A[0][0] = 0.5, A[1][1] = 0.8;
    auto ss = LTIDiscreteSystem(A, Matrix<double, 2, 0>{},
                                make_identity<double, 2>());
    REQUIRE(ss.is_stable() == Stability::Asymptotic);

    // (dakre) \lambda I - A = [0 0] rank 0 => not reachable
    REQUIRE(!ss.is_reachable());

    const auto dot_x = ss.step(Vectord<2>{1.0, 1.0});
    REQUIRE(dot_x(0) == 0.5);
    REQUIRE(dot_x(1) == 0.8);

    const double h = 1.0;
    const auto y = ss.output(h * dot_x);
    REQUIRE(constexpr_abs(y(0) - 0.5) < unit_test::eps);
    REQUIRE(constexpr_abs(y(1) - 0.8) < unit_test::eps);
  }

  SECTION("Asymptotically stable system that is reachable") {
    Matrix<double, 2, 2> A;
    A[0][0] = 0.5, A[1][1] = 0.8;
    auto B = make_identity<double, 2>();
    auto ss = LTIDiscreteSystem(A, B, make_identity<double, 2>());
    REQUIRE(ss.is_stable() == Stability::Asymptotic);
    REQUIRE(ss.is_reachable());
  }
}
