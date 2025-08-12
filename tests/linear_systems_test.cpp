/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <iostream>

#include "math.hpp"

using namespace topology;
using namespace algebra;
using namespace spaces;

template <typename T>
concept DefaultConstructible = requires { T{}; };

// (dakre) TODO please note that constexpr sqrt for 0 has precision of 1e-10
// need to fix more and look into but an improvement over newton step precision
// of 1e-7 w/ guess 1 before. Thus setting test eps to 1e-10
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
    constexpr auto QR = qr(A);
    constexpr auto Q = std::get<0>(QR);
    constexpr auto R = std::get<1>(QR);
    constexpr auto A_prime = Q * R;
    STATIC_REQUIRE(all_equal(A_prime, A));

    STATIC_REQUIRE(R(1)(0) == 0);
    STATIC_REQUIRE(R(2)(0) == 0);
    STATIC_REQUIRE(R(2)(1) == 0);

    constexpr auto QTQ = transpose(Q) * Q;
    STATIC_REQUIRE(all_equal(QTQ, make_identity<double, 3>()));
  }

  SECTION("Complex A") {
    constexpr auto a = std::complex<double>(2.0, 0.0);
    constexpr auto b = std::complex<double>(3.0, 1.0);
    constexpr auto c = std::complex<double>(4.0, 0.0);
    constexpr auto x1 = Vectorcd<2>{a, b};
    constexpr auto x2 = Vectorcd<2>{std::conj(b), c};
    constexpr auto A = Matrixcd<2, 2>{x1, x2};
    constexpr auto QR = qr(A);
    constexpr auto Q = std::get<0>(QR);
    constexpr auto R = std::get<1>(QR);
    constexpr auto A_prime = Q * R;
    STATIC_REQUIRE(all_equal(A_prime, A));

    constexpr auto zero = std::complex<double>(0.0, 0.0);
    STATIC_REQUIRE(R(1)(0) == zero);

    constexpr auto QTQ = conjugate_transpose(Q) * Q;
    STATIC_REQUIRE(all_equal(QTQ, make_identity<std::complex<double>, 2>()));
  }
}

TEST_CASE("Least squares", "[R x = Q^T b]") {
  SECTION("Real A") {
    constexpr auto x1 = Vectord<2>{1.0, 1.0};
    constexpr auto x2 = Vectord<2>{1.0, 2.0};
    constexpr auto x3 = Vectord<2>{1.0, 3.0};
    constexpr auto A = Matrixd<3, 2>{x1, x2, x3};
    constexpr auto QR = qr(A);
    constexpr auto Q = std::get<0>(QR);
    constexpr auto R = std::get<1>(QR);
    STATIC_REQUIRE(std::same_as<decltype(Q), const Matrixd<3, 2>>);
    STATIC_REQUIRE(std::same_as<decltype(R), const Matrixd<2, 2>>);

    constexpr auto QTQ = transpose(Q) * Q;
    STATIC_REQUIRE(std::same_as<decltype(QTQ), const Matrixd<2, 2>>);
    STATIC_REQUIRE(all_equal(QTQ, make_identity<double, 2>()));

    constexpr auto b = Vectord<3>{3.0, 5.0, 7.0};
    constexpr auto x_tilde = lsq(Q, R, b);
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
    constexpr auto QR = qr(A);
    constexpr auto Q = std::get<0>(QR);
    constexpr auto R = std::get<1>(QR);
    STATIC_REQUIRE(std::same_as<decltype(Q), const Matrixcd<3, 2>>);
    STATIC_REQUIRE(std::same_as<decltype(R), const Matrixcd<2, 2>>);

    constexpr auto QTQ = transpose(Q) * Q;
    STATIC_REQUIRE(std::same_as<decltype(QTQ), const Matrixcd<2, 2>>);
    STATIC_REQUIRE(all_equal(QTQ, make_identity<std::complex<double>, 2>()));

    constexpr auto d = std::complex<double>(5.0, 0);
    constexpr auto e = std::complex<double>(7.0, 0);
    constexpr auto y = Vectorcd<3>{c, d, e};
    constexpr auto x_tilde = lsq(Q, R, y);
    STATIC_REQUIRE(euclidean_distance(x_tilde, Vectorcd<2>{1.0, 2.0}) <= eps);
  }
}

TEST_CASE("Eigen Decomposition", "[A X = Lambda X]") {
  SECTION("Diagonal A") {
    constexpr auto diag = Vectord<3>{1.0, 2.0, 3.0};
    constexpr auto A = make_diag<double, 3>(diag);
    constexpr auto result = eigen_decomposition(A);
    constexpr auto Lambda = std::get<0>(result);
    STATIC_REQUIRE(get_eigenvalues(result) == get_diag(Lambda));
    STATIC_REQUIRE(all_equal(Lambda, make_diag(diag)));
  }

  SECTION("Symmetric A") {
    constexpr auto x1 = Vectord<2>{2.0, 1.0};
    constexpr auto x2 = Vectord<2>{1.0, 2.0};
    constexpr auto A = Matrixd<2, 2>{x1, x2};
    constexpr auto result = eigen_decomposition(A);
    constexpr auto Lambda = std::get<0>(result);
    constexpr auto expected = make_diag(Vectord<2>{3.0, 1.0});
    STATIC_REQUIRE(all_equal(Lambda, expected));
  }

  SECTION("Complex hermitian A") {
    constexpr auto a = std::complex<double>(2.0, 0.0);
    constexpr auto b = std::complex<double>(3.0, 1.0);
    constexpr auto c = std::complex<double>(4.0, 0.0);
    constexpr auto x1 = Vectorcd<2>{a, b};
    constexpr auto x2 = Vectorcd<2>{std::conj(b), c};
    constexpr auto A = Matrixcd<2, 2>{x1, x2};
    constexpr auto res = eigen_decomposition(A);
    constexpr auto lambdas = get_eigenvalues(res);
    constexpr auto d = std::complex<double>(3 + constexpr_sqrt(11.0));
    constexpr auto e = std::complex<double>(3 - constexpr_sqrt(11.0));
    constexpr auto expected = make_diag(Vectorcd<2>{d, e});
    constexpr auto result = make_diag(lambdas);
    STATIC_REQUIRE(all_equal(result, expected));
    STATIC_REQUIRE(is_hermitian(A));
  }
}

TEST_CASE("SVD", "[A = U Sigma VT]") {
  SECTION("Known SVD") {
    constexpr auto x1 = Vectord<2>{1.0, 0.0};
    constexpr auto x2 = Vectord<2>{0.0, 2.0};
    constexpr auto x3 = Vectord<2>{0.0, 0.0};
    constexpr auto A = Matrixd<3, 2>{x1, x2, x3};
    constexpr auto result = svd(A);

    constexpr auto U = std::get<0>(result);
    constexpr auto S = std::get<1>(result);
    constexpr auto VT = std::get<2>(result);
    STATIC_REQUIRE(std::same_as<decltype(U), const Matrixd<3, 3>>);
    STATIC_REQUIRE(std::same_as<decltype(S), const Vectord<2>>);
    STATIC_REQUIRE(std::same_as<decltype(VT), const Matrixd<2, 2>>);
    STATIC_REQUIRE(euclidean_distance(S, Vectord<2>{1.0, 2.0}) <= eps);

    constexpr auto I3 = make_identity<double, 3>();
    constexpr auto UTU = transpose(U) * U;
    STATIC_REQUIRE(all_equal(UTU, I3));

    constexpr auto VTV = VT * transpose(VT);
    STATIC_REQUIRE(all_equal(VTV, make_identity<double, 2>()));
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
    constexpr auto result = svd(Z1);
    constexpr auto U = std::get<0>(result);
    constexpr auto S = std::get<1>(result);
    constexpr auto VT = std::get<2>(result);
    STATIC_REQUIRE(std::same_as<decltype(U), const Matrixd<2, 2>>);
    STATIC_REQUIRE(std::same_as<decltype(S), const Vectord<2>>);
    STATIC_REQUIRE(std::same_as<decltype(VT), const Matrixd<2, 2>>);
    STATIC_REQUIRE(euclidean_distance(S, Vectord<2>{0.0, 0.0}) <= eps);

    constexpr auto I = make_identity<double, 2>();
    STATIC_REQUIRE(all_equal(VT, I));
    STATIC_REQUIRE(all_equal(U, I));
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

    constexpr auto result = svd(A);
    STATIC_REQUIRE(is_hermitian(A));
    constexpr auto U = std::get<0>(result);
    constexpr auto S = std::get<1>(result);
    constexpr auto VT = std::get<2>(result);
    STATIC_REQUIRE(std::same_as<decltype(U), const Matrixcd<2, 2>>);
    STATIC_REQUIRE(std::same_as<decltype(S), const Vectord<2>>);
    STATIC_REQUIRE(std::same_as<decltype(VT), const Matrixcd<2, 2>>);

    // (dakre) since A is hermitian its singular values are equivalent to the
    // abs value of the eigenvalues of A
    STATIC_REQUIRE(all_equal(make_diag(S), expected));

    constexpr auto I = make_identity<std::complex<double>, 2>();

    constexpr auto UTU = conjugate_transpose(U) * U;
    STATIC_REQUIRE(all_equal(UTU, I));

    constexpr auto VTV = VT * conjugate_transpose(VT);
    STATIC_REQUIRE(all_equal(VTV, I));
  }
}
