/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#include <algorithm>
#include <catch2/catch_test_macros.hpp>

#include "math.hpp"

using namespace algebra;
using namespace topology;

TEST_CASE("Test field types", "[field]") {
  SECTION("Floating point check") {
    STATIC_REQUIRE(FloatingPoint<double>);
    STATIC_REQUIRE(FloatingPoint<std::array<double, 1>>);
    STATIC_REQUIRE(FloatingPoint<topology::OpenSet<double, 5>>);
    STATIC_REQUIRE(
        FloatingPoint<topology::Topology<topology::OpenSet<double, 3>, 2>>);
  }

  SECTION("Field check") {
    STATIC_REQUIRE(FieldLike<double>);
    STATIC_REQUIRE(FieldLike<float>);
    STATIC_REQUIRE(FieldLike<std::complex<double>>);
    STATIC_REQUIRE(!FieldLike<int>);
    STATIC_REQUIRE(!FieldLike<unsigned>);
  }
}

TEST_CASE("Test vector space types", "[vector_space]") {
  SECTION("Static vector checks") {
    constexpr auto A = Vectord<2>{1.f, 2.f};
    constexpr auto B = Vectord<2>{-1.f, 5.f};
    constexpr auto C = add(A, B);
    constexpr auto D = A + B;
    STATIC_REQUIRE(C(0) == D(0));
    STATIC_REQUIRE(C(1) == D(1));
    STATIC_REQUIRE(C(0) == 0.f);
    STATIC_REQUIRE(C(1) == 7.f);

    constexpr double alpha = 2.f;
    constexpr auto E = alpha * A;
    constexpr auto F = scalar_multiplication(alpha, A);
    STATIC_REQUIRE(E(0) == F(0));
    STATIC_REQUIRE(E(1) == F(1));
    STATIC_REQUIRE(E(0) == 2.f);
    STATIC_REQUIRE(E(1) == 4.f);

    STATIC_REQUIRE(9.f == dot(A, B));
    STATIC_REQUIRE(10.f != dot(A, B));

    constexpr auto G = A - B;
    STATIC_REQUIRE(G(0) == 2.f);
    STATIC_REQUIRE(G(1) == -3.f);
    STATIC_REQUIRE(dot(G, G) == 13.f);
    STATIC_REQUIRE(L2(G) > 0);
    STATIC_REQUIRE(L2(G) == euclidean_distance(A, B));
    STATIC_REQUIRE(L1(G) > 0);
    STATIC_REQUIRE(L1(G) == manhattan_distance(A, B));
    STATIC_REQUIRE(max_norm(G) == 3.f);
  }

  SECTION("Static matrix checks") {
    constexpr auto A = Vectord<2>{1.f, 2.f};
    constexpr auto B = Matrixd<2, 2>{A, A};
    constexpr auto C = B + B;
    constexpr auto D = add(B, B);
    STATIC_REQUIRE(C(0)(0) == D(0)(0));
    STATIC_REQUIRE(C(0)(1) == D(0)(1));
    STATIC_REQUIRE(C(1)(0) == D(1)(0));
    STATIC_REQUIRE(C(1)(1) == D(1)(1));
    STATIC_REQUIRE(C(0)(0) == 2.f);
    STATIC_REQUIRE(C(0)(1) == 4.f);
    STATIC_REQUIRE(C(1)(0) == 2.f);
    STATIC_REQUIRE(C(1)(1) == 4.f);

    constexpr double alpha = 3.f;
    constexpr auto E = scalar_multiplication(alpha, B);
    constexpr auto F = alpha * B;
    STATIC_REQUIRE(E(0)(0) == F(0)(0));
    STATIC_REQUIRE(E(0)(1) == F(0)(1));
    STATIC_REQUIRE(E(1)(0) == F(1)(0));
    STATIC_REQUIRE(E(1)(1) == F(1)(1));
    STATIC_REQUIRE(E(0)(0) == 3.f);
    STATIC_REQUIRE(E(0)(1) == 6.f);
    STATIC_REQUIRE(E(1)(0) == 3.f);
    STATIC_REQUIRE(E(1)(1) == 6.f);

    constexpr auto BT = transpose(B);
    STATIC_REQUIRE(BT(0)(0) == B(0)(0));
    STATIC_REQUIRE(BT(0)(1) == B(1)(0));
    STATIC_REQUIRE(BT(1)(0) == B(0)(1));
    STATIC_REQUIRE(BT(1)(1) == B(1)(1));

    // (dakre) TODO figure out cleaner way to build vectors & matrices
    constexpr auto G = Matrixd<3, 1>{Vectord<3>{1.f, 2.f, 3.f}};
    constexpr auto H =
        Matrixd<1, 3>{Vectord<1>{1.f}, Vectord<1>{2.f}, Vectord<1>{3.f}};

    STATIC_REQUIRE(number_of_rows_v<decltype(G)> ==
                   number_of_cols_v<decltype(H)>);
    STATIC_REQUIRE(number_of_cols_v<decltype(G)> ==
                   number_of_rows_v<decltype(H)>);
    constexpr auto GT = transpose(G);
    constexpr auto HT = transpose(H);
    STATIC_REQUIRE(G == HT);
    STATIC_REQUIRE(GT == H);

    STATIC_REQUIRE(trace(B) == 3.f);
  }
}
