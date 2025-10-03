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

TEST_CASE("Test linear ODE", "[dot(x) = Ax]") {
  SECTION("Zero matrix -> Identity") {
    constexpr auto Z = Matrix<double, 2, 2>{};
    constexpr auto res = matrix_exp(Z);
    constexpr auto I = make_identity<std::complex<double>, 2>();
    STATIC_REQUIRE(all_equal(res, I));
  }

  SECTION("Identity matrix -> e * I") {
    constexpr auto I = make_identity<std::complex<double>, 2>();
    constexpr auto res = matrix_exp(I);
    constexpr auto expected = constexpr_exp(1.0) * I;
    STATIC_REQUIRE(all_equal(res, expected));
  }

  SECTION("Matrix exponential of Jordan block") {
    constexpr auto A =
        Matrix<double, 2, 2>{Vectord<2>{2.0, 1.0}, Vectord<2>{0.0, 2.0}};
    constexpr double e2 = constexpr_exp(2.0);
    constexpr auto expected = Matrix<std::complex<double>, 2, 2>{
        Vectorcd<2>{e2, e2}, Vectorcd<2>{0.0, e2}};
    constexpr auto res = matrix_exp(A);
    STATIC_REQUIRE(all_equal(res, expected));
  }
}

TEST_CASE("Test linear system", "[y=Ax]") {
  constexpr auto x = Vectord<3>{3.0, -2.0, 1.0};
  constexpr auto A = Vector<Vectord<3>, 3>{x, x, x};
  constexpr auto y = A * x;
  STATIC_REQUIRE(y(0) == 14.0);
  STATIC_REQUIRE(y(1) == 14.0);
  STATIC_REQUIRE(y(2) == 14.0);
}

TEST_CASE("Test QR Decomposition", "[A = QR]") {
  SECTION("Real A") {
    constexpr auto x1 = Vectord<3>{1.0, 1.0, 0.0};
    constexpr auto x2 = Vectord<3>{1.0, 0.0, 1.0};
    constexpr auto x3 = Vectord<3>{0.0, 1.0, 1.0};
    constexpr auto A = Matrix<double, 3, 3>{x1, x2, x3};
    constexpr auto qr = QR(A);
    constexpr auto Q = qr.get_Q();
    constexpr auto R = qr.get_R();
    constexpr auto A_prime = Q * R;
    STATIC_REQUIRE(all_equal(A_prime, A));

    STATIC_REQUIRE(R(1)(0) == 0);
    STATIC_REQUIRE(R(2)(0) == 0);
    STATIC_REQUIRE(R(2)(1) == 0);

    constexpr auto QTQ = transpose(Q) * Q;
    STATIC_REQUIRE(all_equal(QTQ, make_identity<double, 3>()));
  }

  SECTION("Real A m > n") {
    constexpr auto x1 = Vectord<2>{1.0, 1.0};
    constexpr auto x2 = Vectord<2>{1.0, 0.0};
    constexpr auto x3 = Vectord<2>{1.0, 0.0};
    constexpr auto A = Matrixd<3, 2>{x1, x2, x3};
    constexpr auto qr = QR(A);
    constexpr auto Q = qr.get_Q();
    constexpr auto R = qr.get_R();
    STATIC_REQUIRE(std::same_as<decltype(Q), const Matrixd<3, 3>>);
    STATIC_REQUIRE(std::same_as<decltype(R), const Matrixd<3, 2>>);
    constexpr auto A_prime = Q * R;
    STATIC_REQUIRE(all_equal(A_prime, A));

    constexpr auto rank = qr.get_rank();
    STATIC_REQUIRE(rank == 2);

    constexpr auto QTQ = transpose(Q) * Q;
    constexpr auto QTQr = get_n_cols<0, rank>(get_n_rows<0, rank>(QTQ));
    STATIC_REQUIRE(all_equal(QTQr, make_identity<double, 2>()));
  }

  SECTION("Real A m < n") {
    constexpr auto x1 = Vectord<3>{1.0, 1.0, 0.0};
    constexpr auto x2 = Vectord<3>{0.0, 1.0, 1.0};
    constexpr auto A = Matrixd<2, 3>{x1, x2};
    constexpr auto qr = QR(transpose(A));
    constexpr auto Q = transpose(qr.get_Q());
    constexpr auto L = transpose(qr.get_R());
    STATIC_REQUIRE(std::same_as<decltype(Q), const Matrixd<3, 3>>);
    STATIC_REQUIRE(std::same_as<decltype(L), const Matrixd<2, 3>>);
    constexpr auto A_prime = L * Q;
    STATIC_REQUIRE(all_equal(A_prime, A));

    constexpr auto rank = qr.get_rank();
    STATIC_REQUIRE(rank == 2);
  }

  SECTION("Complex A") {
    constexpr auto a = std::complex<double>(2.0, 0.0);
    constexpr auto b = std::complex<double>(3.0, 1.0);
    constexpr auto c = std::complex<double>(4.0, 0.0);
    constexpr auto x1 = Vectorcd<2>{a, b};
    constexpr auto x2 = Vectorcd<2>{std::conj(b), c};
    constexpr auto A = Matrixcd<2, 2>{x1, x2};
    constexpr auto qr = QR(A);
    constexpr auto Q = qr.get_Q();
    constexpr auto R = qr.get_R();
    constexpr auto A_prime = Q * R;
    STATIC_REQUIRE(all_equal(A_prime, A));

    constexpr auto zero = std::complex<double>(0.0, 0.0);
    STATIC_REQUIRE(R(1)(0) == zero);

    constexpr auto QTQ = conjugate_transpose(Q) * Q;
    STATIC_REQUIRE(all_equal(QTQ, make_identity<std::complex<double>, 2>()));
    STATIC_REQUIRE(2 == qr.get_rank());
  }
}

TEST_CASE("Least squares", "[R x = Q^T b]") {
  SECTION("Real A") {
    constexpr auto x1 = Vectord<2>{1.0, 1.0};
    constexpr auto x2 = Vectord<2>{1.0, 2.0};
    constexpr auto x3 = Vectord<2>{1.0, 3.0};
    constexpr auto A = Matrixd<3, 2>{x1, x2, x3};
    constexpr auto qr = QR(A);
    constexpr auto Q = qr.get_Q();
    constexpr auto R = qr.get_R();
    STATIC_REQUIRE(std::same_as<decltype(Q), const Matrixd<3, 3>>);
    STATIC_REQUIRE(std::same_as<decltype(R), const Matrixd<3, 2>>);

    constexpr auto rank = qr.get_rank();
    STATIC_REQUIRE(rank == 2);

    constexpr auto A_prime = Q * R;
    STATIC_REQUIRE(all_equal(A_prime, A));

    constexpr auto QTQ = transpose(Q) * Q;
    STATIC_REQUIRE(std::same_as<decltype(QTQ), const Matrixd<3, 3>>);
    constexpr auto QTQr = get_n_cols<0, rank>(get_n_rows<0, rank>(QTQ));
    STATIC_REQUIRE(all_equal(QTQr, make_identity<double, 2>()));

    constexpr auto b = Vectord<3>{3.0, 5.0, 7.0};
    constexpr auto x_tilde = qr.solve(b);

    STATIC_REQUIRE(euclidean_distance(x_tilde, Vectord<2>{1.0, 2.0}) <= eps);
  }

  SECTION("Complex A") {
    // (dakre) adding small eps to im part but otherwise same as before
    constexpr auto a = std::complex<double>(1.0, 1e-8);
    constexpr auto b = std::complex<double>(2.0, 0.0);
    constexpr auto c = std::complex<double>(3.0, 0.0);
    constexpr auto x1 = Vectorcd<2>{a, a};
    constexpr auto x2 = Vectorcd<2>{a, b};
    constexpr auto x3 = Vectorcd<2>{a, c};
    constexpr auto A = Matrixcd<3, 2>{x1, x2, x3};
    constexpr auto qr = QR(A);
    constexpr auto Q = qr.get_Q();
    constexpr auto R = qr.get_R();
    STATIC_REQUIRE(std::same_as<decltype(Q), const Matrixcd<3, 3>>);
    STATIC_REQUIRE(std::same_as<decltype(R), const Matrixcd<3, 2>>);

    constexpr auto rank = qr.get_rank();
    STATIC_REQUIRE(rank == 2);

    constexpr auto QTQ = transpose(Q) * Q;
    STATIC_REQUIRE(std::same_as<decltype(QTQ), const Matrixcd<3, 3>>);
    constexpr auto QTQr = get_n_cols<0, rank>(get_n_rows<0, rank>(QTQ));
    STATIC_REQUIRE(all_equal(QTQr, make_identity<std::complex<double>, 2>()));

    constexpr auto d = std::complex<double>(5.0, 0);
    constexpr auto e = std::complex<double>(7.0, 0);
    constexpr auto y = Vectorcd<3>{c, d, e};
    constexpr auto x_tilde = qr.solve(y);
    STATIC_REQUIRE(euclidean_distance(x_tilde, Vectorcd<2>{1.0, 2.0}) <= eps);
  }

  SECTION("1x1 case") {
    constexpr auto X = Vector<double, 1>{0.0};
    constexpr auto A = Matrix<double, 1, 1>{X};
    constexpr auto qr = QR(A);
    constexpr auto Q = qr.get_Q();
    constexpr auto R = qr.get_R();
    STATIC_REQUIRE(Q(0)(0) == 1.0);
    STATIC_REQUIRE(R(0)(0) == 0.0);
  }

  SECTION("Min norm case m < n (Reals)") {
    mat_traits<Matrix<double, 2, 3>> A_mat;
    constexpr auto A = A_mat(1.0, 0.0, 0.0, 0.0, 1.0, 0.0);

    constexpr Vector<double, 2> b{2.0, 3.0};
    constexpr auto qr = QR(transpose(A));
    constexpr auto Q = qr.get_Q();
    constexpr auto R = qr.get_R();
    constexpr auto x_min = min_norm(Q, R, b);
    STATIC_REQUIRE(euclidean_distance(A * x_min, b) <= eps);
  }

  SECTION("Min norm case m < n (Complex)") {
    mat_traits<Matrix<std::complex<double>, 2, 3>> A_mat;
    constexpr auto A = A_mat(1.0, 0.0, 1.0, 0.0, 1.0, 0.0);

    constexpr Vector<std::complex<double>, 2> b{2.0, -3.0};
    constexpr auto qr = QR(transpose(A));
    constexpr auto Q = qr.get_Q();
    constexpr auto R = qr.get_R();
    constexpr auto x_min = min_norm(Q, R, b);
    STATIC_REQUIRE(euclidean_distance(A * x_min, b) <= eps);
  }
}

TEST_CASE("Eigen Decomposition", "[A X = Lambda X]") {
  SECTION("Diagonal A") {
    constexpr auto diag = Vectord<3>{1.0, 2.0, 3.0};
    constexpr auto A = make_diag<double, 3>(diag);
    constexpr auto result = EigenDecomposition(A);
    constexpr auto converged = result.get_converged();
    STATIC_REQUIRE(converged);

    constexpr auto lambdas = result.get_eigenvalues();
    STATIC_REQUIRE(lambdas(0).real() == diag(0));
    STATIC_REQUIRE(lambdas(1).real() == diag(1));
    STATIC_REQUIRE(lambdas(2).real() == diag(2));

    constexpr auto V = result.get_eigenvectors();
    STATIC_REQUIRE(all_equal(V, make_identity<std::complex<double>, 3>()));

    constexpr auto rho = result.get_spectral_radius();
    STATIC_REQUIRE(rho == std::complex<double>(3.0));
  }

  SECTION("Symmetric A") {
    constexpr auto x1 = Vectord<2>{2.0, 1.0};
    constexpr auto x2 = Vectord<2>{1.0, 2.0};
    constexpr auto A = Matrixd<2, 2>{x1, x2};
    constexpr auto result = EigenDecomposition(A);
    constexpr auto expected =
        make_diag(Vectorcd<2>{std::complex(3.0, 0.0), std::complex(1.0, 0.0)});
    constexpr auto converged = result.get_converged();
    STATIC_REQUIRE(converged);

    constexpr auto lambdas = result.get_eigenvalues();
    STATIC_REQUIRE(all_equal(make_diag(lambdas), expected));

    constexpr auto V = result.get_eigenvectors();
    constexpr auto VTV = conjugate_transpose(V) * V;
    STATIC_REQUIRE(all_equal(VTV, make_identity<std::complex<double>, 2>()));

    constexpr auto VTAV = conjugate_transpose(V) * (as_complex_matrix(A) * V);
    STATIC_REQUIRE(all_equal(make_diag(lambdas), VTAV));

    constexpr auto VLVT = V * (make_diag(lambdas) * conjugate_transpose(V));
    STATIC_REQUIRE(all_equal(as_complex_matrix(A), VLVT));

    constexpr auto rho = result.get_spectral_radius();
    STATIC_REQUIRE(constexpr_absc(rho - expected(0)(0)) <= eps);
  }

  SECTION("Real A") {
    constexpr auto x1 = Vectord<3>{1.0, 2.0, 3.0};
    constexpr auto x2 = Vectord<3>{4.0, 5.0, 6.0};
    constexpr auto x3 = Vectord<3>{7.0, 8.0, 9.0};
    constexpr auto A = Matrixd<3, 3>{x1, x2, x3};
    constexpr auto result = EigenDecomposition(A);
    constexpr auto converged = result.get_converged();
    STATIC_REQUIRE(converged);

    constexpr auto expected = make_diag(
        Vectorcd<3>{std::complex(1.61168440e+01), std::complex(-1.11684397e+00),
                    std::complex(0.0)});
    constexpr auto lambdas = result.get_eigenvalues();
    STATIC_REQUIRE(all_equal(make_diag(lambdas), expected));

    constexpr auto rho = result.get_spectral_radius();
    STATIC_REQUIRE(rho == lambdas(0));
  }

  SECTION("Complex hermitian A") {
    constexpr auto a = std::complex<double>(2.0, 0.0);
    constexpr auto b = std::complex<double>(3.0, 1.0);
    constexpr auto c = std::complex<double>(4.0, 0.0);
    constexpr auto x1 = Vectorcd<2>{a, b};
    constexpr auto x2 = Vectorcd<2>{std::conj(b), c};
    constexpr auto A = Matrixcd<2, 2>{x1, x2};

    constexpr auto res = EigenDecomposition(A);
    constexpr auto lambdas = res.get_eigenvalues();
    constexpr auto converged = res.get_converged();
    STATIC_REQUIRE(converged);

    constexpr auto d = std::complex<double>(3 + constexpr_sqrt(11.0));
    constexpr auto e = std::complex<double>(3 - constexpr_sqrt(11.0));
    constexpr auto expected = make_diag(Vectorcd<2>{d, e});
    constexpr auto result = make_diag(lambdas);
    STATIC_REQUIRE(all_equal(result, expected));

    constexpr auto V = res.get_eigenvectors();
    constexpr auto V1 = as_col_vector(V(0));
    constexpr auto V2 = as_col_vector(V(1));
    constexpr auto V1TV2 = conjugate_transpose(V1) * V2;
    STATIC_REQUIRE((V1TV2 * V1TV2)(0)(0).real() < eps);
    STATIC_REQUIRE((V1TV2 * V1TV2)(0)(0).imag() < eps);

    constexpr auto VTV = conjugate_transpose(V) * V;
    STATIC_REQUIRE(all_equal(VTV, make_identity<std::complex<double>, 2>()));

    constexpr auto VTAV = conjugate_transpose(V) * (as_complex_matrix(A) * V);
    STATIC_REQUIRE(all_equal(make_diag(lambdas), VTAV));

    constexpr auto VLVT = V * (result * conjugate_transpose(V));
    STATIC_REQUIRE(all_equal(A, VLVT));

    constexpr auto rho = res.get_spectral_radius();
    STATIC_REQUIRE(rho == lambdas(0));
  }
}

TEST_CASE("SVD", "[A = U Sigma VT]") {
  SECTION("Known SVD") {
    constexpr auto x1 = Vectord<2>{1.0, 0.0};
    constexpr auto x2 = Vectord<2>{0.0, 2.0};
    constexpr auto x3 = Vectord<2>{0.0, 0.0};
    constexpr auto A = Matrixd<3, 2>{x1, x2, x3};
    constexpr auto result = SVD(A);

    constexpr auto U = result.get_left_singular_vectors();
    constexpr auto S = result.get_singular_values();
    constexpr auto VT = result.get_right_singular_vectors();
    STATIC_REQUIRE(std::same_as<decltype(U), const Matrixcd<3, 3>>);
    STATIC_REQUIRE(std::same_as<decltype(S), const Matrixd<3, 2>>);
    STATIC_REQUIRE(std::same_as<decltype(VT), const Matrixcd<2, 2>>);
    STATIC_REQUIRE(euclidean_distance(get_diag(S), Vectord<2>{2.0, 1.0}) <=
                   eps);

    constexpr auto UTU = conjugate_transpose(U) * U;
    STATIC_REQUIRE(all_equal(get_n_rows<0, 2>(get_n_cols<0, 2>(UTU)),
                             make_identity<std::complex<double>, 2>()));

    constexpr auto VTV = VT * conjugate_transpose(VT);
    STATIC_REQUIRE(all_equal(VTV, make_identity<std::complex<double>, 2>()));

    constexpr auto USVT = U * (as_complex_matrix(S) * VT);
    STATIC_REQUIRE(all_equal(as_complex_matrix(A), USVT));
  }

  SECTION("Zeros case") {
    constexpr auto Z1 = make_zeros<double, 2, 2>();
    constexpr auto all_zeros = [](const auto& Z) {
      bool rc = true;
      for (const auto& row : Z)
        for (const auto& elem : row) rc &= elem == 0;
      return rc;
    };
    STATIC_REQUIRE(all_zeros(Z1));
    constexpr auto result = SVD(Z1);
    constexpr auto U = result.get_left_singular_vectors();
    constexpr auto S = result.get_singular_values();
    constexpr auto VT = result.get_right_singular_vectors();
    STATIC_REQUIRE(std::same_as<decltype(U), const Matrixcd<2, 2>>);
    STATIC_REQUIRE(std::same_as<decltype(S), const Matrixd<2, 2>>);
    STATIC_REQUIRE(std::same_as<decltype(VT), const Matrixcd<2, 2>>);
    STATIC_REQUIRE(euclidean_distance(get_diag(S), Vectord<2>{0.0, 0.0}) <=
                   eps);

    constexpr auto V1 = as_col_vector(VT(0));
    constexpr auto V2 = as_col_vector(VT(1));
    constexpr auto V1TV2 = conjugate_transpose(V1) * V2;
    STATIC_REQUIRE((V1TV2 * V1TV2)(0)(0).real() < eps);
    STATIC_REQUIRE((V1TV2 * V1TV2)(0)(0).imag() < eps);

    constexpr auto U1 = as_col_vector(U(0));
    constexpr auto U2 = as_col_vector(U(1));
    constexpr auto U1TU2 = conjugate_transpose(U1) * U2;
    STATIC_REQUIRE((U1TU2 * U1TU2)(0)(0).real() < eps);
    STATIC_REQUIRE((U1TU2 * U1TU2)(0)(0).imag() < eps);
  }

  SECTION("Complex hermitian A") {
    constexpr auto a = std::complex<double>(2.0, 0.0);
    constexpr auto b = std::complex<double>(3.0, 1.0);
    constexpr auto c = std::complex<double>(4.0, 0.0);
    constexpr auto x1 = Vectorcd<2>{a, b};
    constexpr auto x2 = Vectorcd<2>{std::conj(b), c};
    constexpr auto A = Matrixcd<2, 2>{x1, x2};

    constexpr auto d = double(3 + constexpr_sqrt(11.0));
    constexpr auto e = constexpr_abs(double(3 - constexpr_sqrt(11.0)));
    constexpr auto expected = make_diag(Vectord<2>{d, e});

    constexpr auto result = SVD(A);
    constexpr auto U = result.get_left_singular_vectors();
    constexpr auto S = result.get_singular_values();
    constexpr auto VT = result.get_right_singular_vectors();
    STATIC_REQUIRE(std::same_as<decltype(U), const Matrixcd<2, 2>>);
    STATIC_REQUIRE(std::same_as<decltype(S), const Matrixd<2, 2>>);
    STATIC_REQUIRE(std::same_as<decltype(VT), const Matrixcd<2, 2>>);

    // (dakre) since A is hermitian its singular values are equivalent to the
    // abs value of the eigenvalues of A
    STATIC_REQUIRE(all_equal(S, expected));

    constexpr auto I = make_identity<std::complex<double>, 2>();

    constexpr auto UTU = conjugate_transpose(U) * U;
    STATIC_REQUIRE(all_equal(UTU, I));

    constexpr auto VTV = VT * conjugate_transpose(VT);
    STATIC_REQUIRE(all_equal(VTV, I));

    constexpr auto USVT = U * (as_complex_matrix(S) * VT);
    STATIC_REQUIRE(all_equal(A, USVT));
  }
}
