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
  SECTION("Unstable / reachable / observable") {
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
    STATIC_REQUIRE(ss.is_observable());
  }

  SECTION("Unstable / reachable / unobservable") {
    constexpr Vector<double, 2> r1{1.0, 2.0};
    constexpr Vector<double, 2> r2{3.0, 4.0};
    constexpr Matrix<double, 2, 2> A{r1, r2};
    constexpr Vector<double, 1> a{1.0};
    constexpr Vector<double, 1> b{2.0};
    constexpr Matrix<double, 2, 1> B{a, b};
    constexpr Matrix<double, 1, 2> C{};

    constexpr auto ss = LTIDiscreteSystem(A, B, C);
    STATIC_REQUIRE(ss.is_stable() == Stability::Unstable);
    STATIC_REQUIRE(ss.is_reachable());
    STATIC_REQUIRE(!ss.is_observable());
  }

  SECTION("Unstable / unreachable / observable") {
    Matrix<double, 3, 3> A;
    A[0][0] = 2, A[1][1] = -1, A[2][2] = 3;
    Matrix<double, 3, 1> B;
    B[1][0] = 1;
    auto ss = LTIDiscreteSystem(A, B, make_identity<double, 3>());

    REQUIRE(ss.is_stable() == Stability::Unstable);
    REQUIRE(!ss.is_reachable());
    REQUIRE(ss.is_observable());
  }

  SECTION("Unstable / unreachable / unobservable") {
    Matrix<double, 3, 3> A;
    A[0][0] = 2, A[1][1] = -1, A[2][2] = 3;
    Matrix<double, 3, 1> B;
    B[1][0] = 1;
    Matrix<double, 1, 3> C;
    C[0][0] = -2;
    auto ss = LTIDiscreteSystem(A, B, C);

    REQUIRE(ss.is_stable() == Stability::Unstable);
    REQUIRE(!ss.is_reachable());
    REQUIRE(!ss.is_observable());
  }

  SECTION("Marginally stable / reachable / observable") {
    constexpr auto A = make_identity<double, 3>();
    constexpr auto C = make_identity<double, 3>();
    constexpr auto ss = LTIDiscreteSystem(A, Matrix<double, 3, 0>{}, C);

    // (dakre) TODO similar condition as before with complex value evaluated at
    // compile time
    REQUIRE(ss.is_stable() == Stability::Marginal);
    // (dakre) \lambda I - A = [0 0] rank 0 => not reachable
    REQUIRE(!ss.is_reachable());
    REQUIRE(ss.is_observable());

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

  SECTION("Marginally stable / reachable / unobservable") {
    const auto A = make_identity<double, 3>();
    const auto B = make_identity<double, 3>();
    auto ss = LTIDiscreteSystem(A, B, make_zeros<double, 1, 3>());
    REQUIRE(ss.is_stable() == Stability::Marginal);
    REQUIRE(ss.is_reachable());
    REQUIRE(!ss.is_observable());
  }

  SECTION("Marginally stable / unreachable / observable") {
    const auto A = make_identity<double, 3>();
    Matrix<double, 3, 1> B;
    B[0][0] = 1.0;
    auto ss = LTIDiscreteSystem(A, B, make_identity<double, 3>());
    REQUIRE(ss.is_stable() == Stability::Marginal);
    REQUIRE(!ss.is_reachable());
    REQUIRE(ss.is_observable());
  }

  SECTION("Marginally stable / unreachable / unobservable") {
    const auto A = make_identity<double, 3>();
    Matrix<double, 3, 1> B;
    B[0][0] = 1.0, B[1][0] = 2.0, B[2][0] = 3.0;
    auto ss = LTIDiscreteSystem(A, B, make_zeros<double, 3, 3>());
    REQUIRE(ss.is_stable() == Stability::Marginal);
    REQUIRE(!ss.is_reachable());
    REQUIRE(!ss.is_observable());
  }

  SECTION("Asymptotically stable / reachable / observable") {
    auto A = 0.1 * make_identity<double, 4>();
    auto B = make_identity<double, 4>();
    auto C = make_identity<double, 4>();
    auto ss = LTIDiscreteSystem(A, B, C);
    REQUIRE(ss.is_stable() == Stability::Asymptotic);
    REQUIRE(ss.is_reachable());
    REQUIRE(ss.is_observable());
  }

  SECTION("Asymptotically stable / reachable / unobservable") {
    auto A = 0.5 * make_identity<double, 4>();
    auto B = make_identity<double, 4>();
    Matrix<double, 1, 4> C;
    C[0][0] = 5.0;
    auto ss = LTIDiscreteSystem(A, B, C);
    REQUIRE(ss.is_stable() == Stability::Asymptotic);
    REQUIRE(ss.is_reachable());
    REQUIRE(!ss.is_observable());
  }

  SECTION("Asymptotically stable / unreachable / observable") {
    Matrix<double, 2, 2> A;
    A[0][0] = 0.5, A[1][1] = 0.8;
    auto ss = LTIDiscreteSystem(A, Matrix<double, 2, 0>{},
                                make_identity<double, 2>());
    REQUIRE(ss.is_stable() == Stability::Asymptotic);

    // (dakre) \lambda I - A = [0 0] rank 0 => not reachable
    REQUIRE(!ss.is_reachable());
    REQUIRE(ss.is_observable());

    const auto dot_x = ss.step(Vectord<2>{1.0, 1.0});
    REQUIRE(dot_x(0) == 0.5);
    REQUIRE(dot_x(1) == 0.8);

    const double h = 1.0;
    const auto y = ss.output(h * dot_x);
    REQUIRE(constexpr_abs(y(0) - 0.5) < unit_test::eps);
    REQUIRE(constexpr_abs(y(1) - 0.8) < unit_test::eps);
  }

  SECTION("Asymptotically stable / unreachable / unobservable") {
    Matrix<double, 2, 2> A;
    A[0][0] = 0.5, A[1][1] = 0.8;
    auto ss = LTIDiscreteSystem(A, Matrix<double, 2, 0>{},
                                make_zeros<double, 1, 2>());
    REQUIRE(ss.is_stable() == Stability::Asymptotic);

    // (dakre) \lambda I - A = [0 0] rank 0 => not reachable
    REQUIRE(!ss.is_reachable());
    REQUIRE(!ss.is_observable());
  }

  SECTION("Asymptotically stable system / reachable / observable") {
    Matrix<double, 2, 2> A;
    A[0][0] = 0.5, A[1][1] = 0.8;
    auto B = make_identity<double, 2>();
    auto ss = LTIDiscreteSystem(A, B, make_identity<double, 2>());
    REQUIRE(ss.is_stable() == Stability::Asymptotic);
    REQUIRE(ss.is_reachable());
    REQUIRE(ss.is_observable());
  }
}
