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
using namespace unit_test;

TEST_CASE("Test field types", "[field]") {
  SECTION("Field check") {
    STATIC_REQUIRE(FieldLike<double>);
    STATIC_REQUIRE(FieldLike<float>);
    STATIC_REQUIRE(FieldLike<std::complex<double>>);
    STATIC_REQUIRE(!FieldLike<int>);
    STATIC_REQUIRE(!FieldLike<unsigned>);
  }
}

TEST_CASE("Test constexpr math", "[math]") {
  constexpr auto res1 = constexpr_sqrt(64.0) - 8.0;
  STATIC_REQUIRE(res1 < eps);

  STATIC_REQUIRE(constexpr_abs(-2.0) == 2.0);

  STATIC_REQUIRE(constexpr_abs(constexpr_cos(0.0) - 1.0) < eps);
  STATIC_REQUIRE(constexpr_sin(0.0) < eps);

  STATIC_REQUIRE(constexpr_cos(M_PI / 2) < eps);
  STATIC_REQUIRE(constexpr_abs(constexpr_sin(M_PI / 2) - 1.0) < eps);

  STATIC_REQUIRE(constexpr_sin(M_PI) < eps);

  STATIC_REQUIRE(constexpr_sin(M_PI / 4) - constexpr_cos(M_PI / 4) < eps);

  constexpr auto a = constexpr_expi(M_PI / 6);
  STATIC_REQUIRE(a.real() == constexpr_cos(M_PI / 6));
  STATIC_REQUIRE(a.imag() == constexpr_sin(M_PI / 6));

  constexpr auto b = std::complex<double>(1.0, -2.0);
  constexpr auto expected = 2.3607;
  STATIC_REQUIRE(L2<std::complex<double>, 1>()(b) - expected < eps);

  constexpr auto V = Vectord<5>{1.0, 2.0, 3.0, 4.0, 5.0};
  constexpr auto P = argsort_desc(V);
  STATIC_REQUIRE(P == std::array<size_t, 5>{4, 3, 2, 1, 0});
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
    STATIC_REQUIRE(L2<double, 2>()(G) > 0);
    STATIC_REQUIRE(L2<double, 2>()(G) == euclidean_distance(A, B));
    STATIC_REQUIRE(L1<double, 2>()(G) > 0);
    STATIC_REQUIRE(L1<double, 2>()(G) == manhattan_distance(A, B));
    STATIC_REQUIRE(Linf<double, 2>()(G) == 3.0);

    constexpr auto normalized_A = normalize(A, L2<double, 2>());
    constexpr auto norm_A = L2<double, 2>()(A);
    STATIC_REQUIRE((1 / norm_A) * A == normalized_A);
    constexpr auto normalized_B = normalize(B, Linf<double, 2>());
    constexpr auto norm_B = Linf<double, 2>()(B);
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
    STATIC_REQUIRE(cross(ey, ex) == -1.0 * ez);
    STATIC_REQUIRE(cross(ez, ex) == ey);
    STATIC_REQUIRE(cross(ex, ez) == -1.0 * ey);
    STATIC_REQUIRE(cross(ey, ez) == ex);
    STATIC_REQUIRE(cross(ez, ey) == -1.0 * ex);

    STATIC_REQUIRE(
        cross(Vectord<3>(1.0, 2.0, 3.0), Vectord<3>(4.0, 5.0, 6.0)) ==
        Vectord<3>(-3.0, 6.0, -3.0));
  }

  SECTION("Static matrix checks") {
    constexpr auto A = Vectord<2>{1.0, 2.0};
    constexpr auto B = Matrixd<2, 2>{A, A};
    constexpr auto C = B + B;
    constexpr auto D = add(B, B);

    constexpr auto B_inv = inverse(B);
    constexpr auto B_B_inv = B * B_inv;
    // (dakre) this case is acting like pseudoinverse since B is degenerate
    STATIC_REQUIRE(B_B_inv(0)(0) == 0.5);
    STATIC_REQUIRE(B_B_inv(0)(1) == 0.5);
    STATIC_REQUIRE(B_B_inv(1)(0) == 0.5);
    STATIC_REQUIRE(B_B_inv(1)(1) == 0.5);

    // (dakre) this case acts like true inverse where Z is unique
    constexpr auto Z =
        Matrixd<2, 2>{Vectord<2>{1.0, 2.0}, Vectord<2>{3.0, 4.0}};
    constexpr auto Z_inv = inverse(Z);
    constexpr auto Z_Z_inv = Z * Z_inv;
    STATIC_REQUIRE(all_equal(Z_Z_inv, make_identity<double, 2>()));

    constexpr auto B_11 = get_block<0, 1, 0, 1>(B);
    STATIC_REQUIRE(B_11(0)(0) == 1.0);

    constexpr auto B_new =
        set_block<0, 1, 0, 1>(B, Matrix<double, 1, 1>{Vector<double, 1>{-2.0}});
    STATIC_REQUIRE(B_new(0)(0) == -2.0);
    STATIC_REQUIRE(B_new(0)(1) == 2.0);
    STATIC_REQUIRE(B_new(1)(0) == 1.0);
    STATIC_REQUIRE(B_new(1)(1) == 2.0);

    STATIC_REQUIRE(all_equal(C, D));
    STATIC_REQUIRE(C(0)(0) == 2.0);
    STATIC_REQUIRE(C(0)(1) == 4.0);
    STATIC_REQUIRE(C(1)(0) == 2.0);
    STATIC_REQUIRE(C(1)(1) == 4.0);

    STATIC_REQUIRE(get_n_rows<0, 1>(B) == get_row(B, 0));
    STATIC_REQUIRE(all_equal(get_n_rows<0, 2>(B), B));
    STATIC_REQUIRE(get_n_cols<0, 1>(B) == transpose(get_col(B, 0)));
    STATIC_REQUIRE(all_equal(get_n_cols<0, 2>(B), B));

    constexpr auto BR = get_upper_right_triangular_matrix(B);
    constexpr auto BL = get_lower_left_triangular_matrix(B);
    STATIC_REQUIRE(BR(1)(0) == 0.0);
    STATIC_REQUIRE(BL(0)(1) == 0.0);
    STATIC_REQUIRE(get_diag(BR) == get_diag(BL));

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
    STATIC_REQUIRE(Fro<std::complex<double>, 2>()(K) ==
                   constexpr_sqrt(trace(L).real()));

    constexpr auto I = make_identity<double, 3>();
    constexpr auto I_diag = get_diag(I);
    STATIC_REQUIRE(I_diag == Vectord<3>{1.0, 1.0, 1.0});
    constexpr auto off_diag = I - make_diag(I_diag);
    constexpr auto Zeros = make_zeros<double, 3, 3>();
    STATIC_REQUIRE(all_equal(off_diag, Zeros));

    STATIC_REQUIRE(I * I == I);

    STATIC_REQUIRE(Fro<double, 3>()(I) == constexpr_sqrt<double>(3));

    STATIC_REQUIRE(is_symmetric(I));

    constexpr auto a = std::complex<double>(2.0, 0.0);
    constexpr auto b = std::complex<double>(3.0, 1.0);
    constexpr auto x1 = Vectorcd<2>{a, b};
    constexpr auto x2 = Vectorcd<2>{std::conj(b), a};
    constexpr auto M = Matrixcd<2, 2>{x1, x2};
    STATIC_REQUIRE(is_hermitian(M));
  }
}

TEST_CASE("Householder vectors", "[householder]") {
  SECTION("Real householder vector") {
    constexpr Vector<double, 4> x{1.0, 2.0, 3.0, 4.0};
    constexpr auto h = HouseholderVector(x);
    constexpr auto beta = h.get_beta();
    constexpr auto v = h.get_v();

    constexpr auto I = make_identity<double, 4>();
    constexpr auto P =
        I - beta * outer_product(as_col_vector(v), as_row_vector(v));
    STATIC_REQUIRE(all_equal(transpose(P) * P, I));

    constexpr auto Px = P * x;
    constexpr auto expected = Vectord<4>{5.47723, 0.0, 0.0, 0.0};
    STATIC_REQUIRE(euclidean_distance(Px, expected) <= eps);
  }

  SECTION("Complex householder vector (all real)") {
    constexpr std::complex<double> a(1.0, 0.0);
    constexpr std::complex<double> b(2.0, 0.0);
    constexpr std::complex<double> c(3.0, 0.0);
    constexpr std::complex<double> d(4.0, 0.0);
    constexpr Vector<std::complex<double>, 4> x{a, b, c, d};

    constexpr auto h = HouseholderVector(x);
    constexpr auto beta = h.get_beta();
    constexpr auto v = h.get_v();

    constexpr auto I = make_identity<std::complex<double>, 4>();
    constexpr auto P =
        I - beta * outer_product(as_col_vector(v), as_row_vector(conjugate(v)));
    STATIC_REQUIRE(all_equal(conjugate_transpose(P) * P, I));

    // (dakre) complex case reflects in opp dir as reals maybe monitor more as
    // time goes on
    constexpr auto Px = P * x;
    constexpr std::complex<double> e(-5.47723, 0.0);
    constexpr std::complex<double> f(0.0, 0.0);
    constexpr auto expected = Vectorcd<4>{e, f, f, f};
    STATIC_REQUIRE(euclidean_distance(Px, expected) <= eps);
  }

  SECTION("Complex householder vector") {
    constexpr std::complex<double> a(1.0, 2.0);
    constexpr std::complex<double> b(3.0, 4.0);
    constexpr Vector<std::complex<double>, 2> x{a, b};

    constexpr auto h = HouseholderVector(x);
    constexpr auto beta = h.get_beta();
    constexpr auto v = h.get_v();

    constexpr auto I = make_identity<std::complex<double>, 2>();
    constexpr auto P =
        I - beta * outer_product(as_col_vector(v), as_row_vector(conjugate(v)));
    STATIC_REQUIRE(all_equal(conjugate_transpose(P) * P, I));
  }
}

TEST_CASE("Hessenberg matrices", "[hessenberg]") {
  SECTION("Real hessenberg matrix") {
    constexpr auto a = Vectord<3>{1.0, 2.0, 3.0};
    constexpr auto b = Vectord<3>{4.0, 5.0, 6.0};
    constexpr auto c = Vectord<3>{7.0, 8.0, 9.0};
    constexpr auto A = Matrixd<3, 3>{a, b, c};
    constexpr auto H_mat = HessenbergMatrix(A);
    constexpr auto H = H_mat();
    constexpr auto beta = H_mat.get_beta();
    STATIC_REQUIRE(constexpr_abs(beta(0)) > 0.0);
    STATIC_REQUIRE(constexpr_abs(beta(1)) == 0.0);
    STATIC_REQUIRE(constexpr_abs(beta(2)) == 0.0);
    STATIC_REQUIRE(H(2)(0) == 0.0);

    constexpr auto U = H_mat.get_U();
    constexpr auto UTU = transpose(U) * U;
    STATIC_REQUIRE(all_equal(UTU, make_identity<double, 3>()));

    STATIC_REQUIRE(all_equal(H, transpose(U) * A * U));
  }
}

TEST_CASE("Matrix types", "[matrix_types]") {
  SECTION("Positive Definite Matrices") {
    constexpr auto A =
        Matrixd<2, 2>{Vectord<2>{2.0, 1.0}, Vectord<2>{1.0, 2.0}};
    auto A_smat = SymmetricMatrix(A);
    REQUIRE(A_smat.type == SymmetricMatrixType::PD);
    REQUIRE(A_smat(0, 0) == 2.0);
    REQUIRE(A_smat(0, 1) == 1.0);
    REQUIRE(A_smat(1, 0) == 1.0);
    REQUIRE(A_smat(1, 1) == 2.0);

    constexpr auto B =
        Matrixd<2, 2>{Vectord<2>{4.0, -1.0}, Vectord<2>{-1.0, 3.0}};
    auto B_smat = SymmetricMatrix(B);
    REQUIRE(B_smat.type == SymmetricMatrixType::PD);
    REQUIRE(B_smat(0, 0) == 4.0);
    REQUIRE(B_smat(0, 1) == -1.0);
    REQUIRE(B_smat(1, 0) == -1.0);
    REQUIRE(B_smat(1, 1) == 3.0);

    constexpr auto I = make_identity<double, 3>();
    auto I_smat = SymmetricMatrix(I);
    REQUIRE(I_smat.type == SymmetricMatrixType::PD);
    REQUIRE(I_smat(0, 0) == 1.0);
    REQUIRE(I_smat(0, 1) == 0.0);
    REQUIRE(I_smat(0, 2) == 0.0);
    REQUIRE(I_smat(1, 0) == 0.0);
    REQUIRE(I_smat(1, 1) == 1.0);
    REQUIRE(I_smat(1, 2) == 0.0);
    REQUIRE(I_smat(2, 0) == 0.0);
    REQUIRE(I_smat(2, 0) == 0.0);
    REQUIRE(I_smat(2, 2) == 1.0);

    Matrix<double, 2, 2> C{Vectord<2>{1.0, 1.0}, Vectord<2>{1.0, 1.0}};
    auto C_smat = SymmetricMatrix(C);
    REQUIRE(C_smat.type == SymmetricMatrixType::PD);
    REQUIRE(C_smat(0, 0) == 1.0);
    REQUIRE(C_smat(0, 1) == 1.0);
    REQUIRE(C_smat(1, 0) == 1.0);
    REQUIRE(C_smat(1, 1) == 1.0);
  }

  SECTION("Positive semidefinite Matrices") {
    auto Z = Matrixd<2, 2>{};
    auto Z_smat = SymmetricMatrix(Z);
    REQUIRE(Z_smat.type != SymmetricMatrixType::PD);
    REQUIRE(Z_smat.type == SymmetricMatrixType::PSD);
    REQUIRE(Z_smat(0, 0) == 0.0);
    REQUIRE(Z_smat(0, 1) == 0.0);
    REQUIRE(Z_smat(1, 0) == 0.0);
    REQUIRE(Z_smat(1, 1) == 0.0);

    Z[0][0] = 1.0;
    Z_smat = SymmetricMatrix(Z);
    REQUIRE(Z_smat.type != SymmetricMatrixType::PD);
    REQUIRE(Z_smat.type == SymmetricMatrixType::PSD);
    REQUIRE(Z_smat(0, 0) == 1.0);
  }

  SECTION("Negative semidefinite Matrices") {
    auto Z = Matrixd<2, 2>{};
    Z[0][0] = -1.0;
    auto Z_smat = SymmetricMatrix(Z);
    REQUIRE(Z_smat.type == SymmetricMatrixType::NSD);
    REQUIRE(Z_smat(0, 0) == -1.0);
  }

  SECTION("Negative definite Matrices") {
    auto I = -1.0 * make_identity<double, 3>();
    auto I_smat = SymmetricMatrix(I);
    REQUIRE(I_smat.type == SymmetricMatrixType::ND);
    REQUIRE(I_smat(0, 0) == -1.0);
    REQUIRE(I_smat(1, 1) == -1.0);
    REQUIRE(I_smat(2, 2) == -1.0);
  }

  SECTION("Indefinite matrix") {
    auto I = -1.0 * make_identity<double, 3>();
    I[0][1] = 2.0, I[1][0] = 2.0;
    auto I_smat = SymmetricMatrix(I);
    REQUIRE(I_smat.type == SymmetricMatrixType::IND);
    REQUIRE(I_smat(0, 0) == -1.0);
    REQUIRE(I_smat(0, 1) == 2.0);
    REQUIRE(I_smat(1, 0) == 2.0);
    REQUIRE(I_smat(1, 1) == -1.0);
    REQUIRE(I_smat(2, 2) == -1.0);
  }

  SECTION("Invalid matrix") {
    auto A = Matrix<double, 2, 2>{Vectord<2>{1.0, 2.0}, Vectord<2>{3.0, 4.0}};
    auto A_smat = SymmetricMatrix(A);
    REQUIRE(A_smat.type == SymmetricMatrixType::INV);
  }
}
