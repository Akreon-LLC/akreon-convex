/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <iostream>

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
    constexpr auto A = Vectord<2>{1.0, 2.0};
    constexpr auto B = Vectord<2>{-1.0, 5.0};
    constexpr auto C = add(A, B);
    constexpr auto D = A + B;
    STATIC_REQUIRE(C == D);
    STATIC_REQUIRE(C(0) == 0.0);
    STATIC_REQUIRE(C(1) == 7.0);

    constexpr double alpha = 2.0;
    constexpr auto E = alpha * A;
    constexpr auto F = scalar_multiplication(alpha, A);
    STATIC_REQUIRE(E == F);
    STATIC_REQUIRE(E(0) == 2.0);
    STATIC_REQUIRE(E(1) == 4.0);

    STATIC_REQUIRE(9.0 == dot(A, B));
    STATIC_REQUIRE(10.0 != dot(A, B));

    constexpr std::complex<double> c(1.0, 2.0);
    constexpr auto X = Vectorcd<5>{c, c, c, c, c};
    constexpr auto result = dot(conjugate(X), X);
    STATIC_REQUIRE(result == std::complex<double>{25, 0});

    constexpr auto G = A - B;
    STATIC_REQUIRE(G(0) == 2.0);
    STATIC_REQUIRE(G(1) == -3.0);
    STATIC_REQUIRE(dot(G, G) == 13.0);
    STATIC_REQUIRE(L2(G) > 0);
    STATIC_REQUIRE(L2(G) == euclidean_distance(A, B));
    STATIC_REQUIRE(L1(G) > 0);
    STATIC_REQUIRE(L1(G) == manhattan_distance(A, B));
    STATIC_REQUIRE(max_norm(G) == 3.0);

    constexpr auto normalized_A = normalize(A, L2<double, 2>);
    constexpr auto norm_A = L2(A);
    STATIC_REQUIRE((1 / norm_A) * A == normalized_A);
    constexpr auto normalized_B = normalize(B, max_norm<double, 2>);
    constexpr auto norm_B = max_norm(B);
    STATIC_REQUIRE((1 / norm_B) * B == normalized_B);

    constexpr auto all_lt_one = [](const auto& vec) {
      bool flag = true;
      for (const auto& x : vec) flag &= x <= 1;
      return flag;
    };
    STATIC_REQUIRE(all_lt_one(normalized_B));

    constexpr auto GT = transpose(G);
    STATIC_REQUIRE(GT(0)(0) == 2.0);
    STATIC_REQUIRE(GT(1)(0) == -3.0);
    constexpr auto GTT = transpose(GT);
    STATIC_REQUIRE(GTT == G);

    constexpr auto G_col_vec = as_col_vector(G);
    constexpr auto G_row_vec = as_row_vector(G);
    STATIC_REQUIRE(std::same_as<decltype(G_col_vec), const Matrixd<2, 1>>);
    STATIC_REQUIRE(std::same_as<decltype(G_row_vec), const Matrixd<1, 2>>);

    constexpr auto G_outer_prod = outer_product(G_col_vec, G_row_vec);
    STATIC_REQUIRE(G_outer_prod(0)(0) == 4.0);
    STATIC_REQUIRE(G_outer_prod(0)(1) == -6.0);
    STATIC_REQUIRE(G_outer_prod(1)(0) == -6.0);
    STATIC_REQUIRE(G_outer_prod(1)(1) == 9.0);

    constexpr auto Zeros = Vectord<3>(0.0, 0.0, 0.0);
    STATIC_REQUIRE(cross(Zeros, Zeros) == Zeros);

    constexpr auto ex = Vectord<3>(1.0, 0.0, 0.0);
    constexpr auto ey = Vectord<3>(0.0, 1.0, 0.0);
    constexpr auto ez = Vectord<3>(0.0, 0.0, 1.0);
    STATIC_REQUIRE(cross(ex, ey) == ez);
    STATIC_REQUIRE(cross(ey, ex) == -1 * ez);
    STATIC_REQUIRE(cross(ez, ex) == ey);
    STATIC_REQUIRE(cross(ex, ez) == -1 * ey);
    STATIC_REQUIRE(cross(ey, ez) == ex);
    STATIC_REQUIRE(cross(ez, ey) == -1 * ex);

    STATIC_REQUIRE(
        cross(Vectord<3>(1.0, 2.0, 3.0), Vectord<3>(4.0, 5.0, 6.0)) ==
        Vectord<3>(-3.0, 6.0, -3.0));
  }

  SECTION("Static matrix checks") {
    constexpr auto A = Vectord<2>{1.0, 2.0};
    constexpr auto B = Matrixd<2, 2>{A, A};
    constexpr auto C = B + B;
    constexpr auto D = add(B, B);

    STATIC_REQUIRE(all_equal(C, D));
    STATIC_REQUIRE(C(0)(0) == 2.0);
    STATIC_REQUIRE(C(0)(1) == 4.0);
    STATIC_REQUIRE(C(1)(0) == 2.0);
    STATIC_REQUIRE(C(1)(1) == 4.0);

    constexpr auto B_col_0 = get_col(B, 0);
    STATIC_REQUIRE(B_col_0(0) == B(0)(0));
    STATIC_REQUIRE(B_col_0(1) == B(1)(0));
    constexpr auto B_col_1 = get_col(B, 1);
    STATIC_REQUIRE(B_col_1(0) == B(0)(1));
    STATIC_REQUIRE(B_col_1(1) == B(1)(1));
    constexpr auto B_row_0 = get_row(B, 0);
    STATIC_REQUIRE(B_row_0 == B(0));
    constexpr auto B_row_1 = get_row(B, 1);
    STATIC_REQUIRE(B_row_1 == B(1));

    constexpr double alpha = 3.0;
    constexpr auto E = scalar_multiplication(alpha, B);
    constexpr auto F = alpha * B;
    STATIC_REQUIRE(E == F);
    STATIC_REQUIRE(E(0)(0) == 3.0);
    STATIC_REQUIRE(E(0)(1) == 6.0);
    STATIC_REQUIRE(E(1)(0) == 3.0);
    STATIC_REQUIRE(E(1)(1) == 6.0);

    constexpr auto BT = transpose(B);
    STATIC_REQUIRE(rows_equal_cols(BT, B));

    // (dakre) TODO figure out cleaner way to build vectors & matrices
    constexpr auto G = Matrixd<1, 3>{Vectord<3>{1.0, 2.0, 3.0}};
    constexpr auto H =
        Matrixd<3, 1>{Vectord<1>{1.0}, Vectord<1>{2.0}, Vectord<1>{3.0}};

    STATIC_REQUIRE(number_of_rows_v<decltype(G)> ==
                   number_of_cols_v<decltype(H)>);
    STATIC_REQUIRE(number_of_cols_v<decltype(G)> ==
                   number_of_rows_v<decltype(H)>);
    constexpr auto GT = transpose(G);
    constexpr auto HT = transpose(H);
    STATIC_REQUIRE(G == HT);
    STATIC_REQUIRE(GT == H);

    STATIC_REQUIRE(trace(B) == 3.0);

    constexpr auto c = std::complex<double>(1.0, 2.0);   // 1 + 2i
    constexpr auto c_conj = std::conj(c);                // 1 - 2i
    constexpr auto d = std::complex<double>(5.0, -1.0);  // 5 - 1i
    constexpr auto d_conj = std::conj(d);                // 5 + 1i
    constexpr auto J = Vectorcd<2>{c, d};
    constexpr auto K = Matrixcd<2, 2>{J, J};
    // K = [[c d]; [c d]] -> K^* = [[\bar{c} \bar{c}] [\bar{d} \bar{d}]]
    constexpr auto KH = conjugate_transpose(K);
    STATIC_REQUIRE(KH(0)(0) == c_conj);
    STATIC_REQUIRE(KH(0)(1) == c_conj);
    STATIC_REQUIRE(KH(1)(0) == d_conj);
    STATIC_REQUIRE(KH(1)(1) == d_conj);

    constexpr auto L = K * KH;
    STATIC_REQUIRE(L(0)(0) == c * c_conj + d * d_conj);
    STATIC_REQUIRE(L(0)(1) == c * c_conj + d * d_conj);
    STATIC_REQUIRE(L(1)(0) == c * c_conj + d * d_conj);
    STATIC_REQUIRE(L(1)(1) == c * c_conj + d * d_conj);
    STATIC_REQUIRE(frobenius_norm(K) == constexpr_sqrt(trace(L).real()));

    constexpr auto I = make_identity<double, 3>();
    constexpr auto I_diag = get_diag(I);
    STATIC_REQUIRE(I_diag == Vectord<3>{1.0, 1.0, 1.0});
    constexpr auto off_diag = I - make_diag(I_diag);
    constexpr auto Zeros = make_zeros<double, 3, 3>();
    STATIC_REQUIRE(all_equal(off_diag, Zeros));

    STATIC_REQUIRE(I * I == I);

    STATIC_REQUIRE(frobenius_norm(I) == constexpr_sqrt<double>(3));

    STATIC_REQUIRE(is_symmetric(I));

    constexpr auto a = std::complex<double>(2.0, 0.0);
    constexpr auto b = std::complex<double>(3.0, 1.0);
    constexpr auto x1 = Vectorcd<2>{a, b};
    constexpr auto x2 = Vectorcd<2>{std::conj(b), a};
    constexpr auto M = Matrixcd<2, 2>{x1, x2};
    STATIC_REQUIRE(is_hermitian(M));
  }
}
