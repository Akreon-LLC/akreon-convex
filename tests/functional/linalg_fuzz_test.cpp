/*
 * Copyright © 2025 Akreon LLC. All rights reserved.
 *
 * Proprietary and Confidential.
 * This source code is the property of Akreon LLC and constitutes a trade secret
 * protected by U.S. and international law.
 *
 * Unauthorized copying, modification, distribution, or disclosure of this file,
 * via any medium, is strictly prohibited without express written permission
 * from Akreon LLC.
 */

#include <algorithm>
#include <cstdio>
#include <random>

#include "math.hpp"

#define TRIALS 10000

#define N_ROWS 8

using namespace linalg;

template <typename T>
void print_scalar(const T& x) {
  if constexpr (is_complex_v<T>)
    std::printf("(%g, %g) ", static_cast<double>(x.real()),
                static_cast<double>(x.imag()));
  else
    std::printf("%g ", static_cast<double>(x));
}

template <typename T, size_t Rows, size_t Cols>
void print_mat(const Matrix<T, Rows, Cols>& A) {
  for (const auto& row : A) {
    for (const auto& pt : row) print_scalar(pt);
    std::printf("\n");
  }
}

template <typename T, size_t N>
void print_vec(const Vector<T, N>& b) {
  for (const auto& x : b) print_scalar(x);
  std::printf("\n");
}

template <typename T, size_t Rows, size_t Cols>
auto make_A(std::mt19937& rng, std::uniform_real_distribution<T>& dist) {
  Matrix<T, Rows, Cols> A;
  for (size_t i = 0; i < Rows; ++i)
    for (size_t j = 0; j < Cols; ++j) A(i)(j) = dist(rng);
  return A;
}

template <typename T, size_t N>
auto make_vec(std::mt19937& rng, std::uniform_real_distribution<T>& dist) {
  Vector<T, N> b;
  for (size_t i = 0; i < N; ++i) b(i) = dist(rng);
  return b;
}

template <typename T, size_t Rows, size_t Cols>
  requires(FieldLike<T> && !is_complex_v<T>)
bool fuzz_test(const scalar_type_t<T>& tol) {
  std::mt19937 rng(42);
  std::uniform_real_distribution<T> dist(-10.0, 10.0);
  for (size_t i = 0; i < TRIALS; ++i) {
    const auto A = make_A<T, Rows, Cols>(rng, dist);
    const auto res = QR(A);
    const auto Q = res.Q();
    const auto R = res.R();
    const auto r = res.rank();
    static_assert(std::same_as<decltype(Q), const Matrix<T, Rows, Rows>>);
    static_assert(std::same_as<decltype(R), const Matrix<T, Rows, Cols>>);
    const auto A_prime = Q * R;
    if (!all_equal(A_prime, A, tol)) {
      std::fprintf(stderr, "QR test case failed for tolerance %g\n",
                   static_cast<double>(tol));
      std::fprintf(stderr, "Matrix rank %zu\n", r);
      print_mat(A_prime);
      print_mat(A);
      return false;
    }

    if constexpr (Rows == Cols) {
      const auto eig_decomp = EigenDecomposition(A);
      if (!eig_decomp.converged) {
        std::fprintf(stderr, "A\n");
        print_mat(A);
        std::fprintf(stderr, "Eigen values\n");
        print_vec(eig_decomp.lambda);
        return false;
      }
    }

    const auto svd = SVD(A);
    const auto U = svd.U;
    const auto S = svd.S;
    const auto VT = svd.VT;
    static_assert(
        std::same_as<decltype(U), const Matrix<std::complex<T>, Rows, Rows>>);
    static_assert(
        std::same_as<decltype(VT), const Matrix<std::complex<T>, Cols, Cols>>);
    static_assert(std::same_as<decltype(S), const Matrix<T, Rows, Cols>>);

    if (!svd.converged &&
        all_equal(as_complex_matrix(A), U * (as_complex_matrix(S) * VT))) {
      std::fprintf(stderr, "A\n");
      print_mat(A);
      std::fprintf(stderr, "Singular values\n");
      print_mat(S);
      std::fprintf(stderr, "U\n");
      print_mat(U);
      std::fprintf(stderr, "VT\n");
      print_mat(VT);
      return false;
    }

    T min_sv = 1000, max_sv = 0;
    for (const auto& row : S) {
      for (const auto& x : row) {
        min_sv = x < min_sv ? x : min_sv;
        max_sv = x > max_sv ? x : max_sv;
      }
    }
    T k = max_sv / min_sv;

    const auto x = make_vec<T, Cols>(rng, dist);
    const auto b = A * x;

    const auto x_tilde = res.solve(b);
    const auto lsq_fit = A * x_tilde;
    const auto residual = euclidean_distance(lsq_fit, b);
    if (residual > tol) {
      std::fprintf(stderr, "LSQ test case failed for tolerance %g\n",
                   static_cast<double>(tol));
      std::fprintf(stderr, "Residual %g\n", static_cast<double>(residual));
      std::fprintf(stderr, "Matrix rank %zu\n", r);
      std::fprintf(stderr, "Matrix condition number %g\n",
                   static_cast<double>(k));
      print_mat(A);
      std::fprintf(stderr, "Vector b\n");
      print_vec(b);
      std::fprintf(stderr, "Vector lsq fit\n");
      print_vec(lsq_fit);
      return false;
    }
  }

  return true;
}

int main() {
  bool ok = true;
  constexpr_for<2, N_ROWS>([&]<size_t i>(std::integral_constant<size_t, i>) {
    if (!ok) return;
    std::printf("Starting double symmetric test for %zu\n", i);
    if (!fuzz_test<double, i, i>(1e-10)) {
      std::fprintf(stderr, "double symmetric fuzz test failed for size %zu\n",
                   i);
      ok = false;
      return;
    }

    std::printf("Starting float symmetric test for %zu\n", i);
    if (!fuzz_test<float, i, i>(1e-3)) {
      std::fprintf(stderr, "float symmetric fuzz test failed for size %zu\n",
                   i);
      ok = false;
    }
  });

  constexpr_for<3, N_ROWS>([&]<size_t i>(std::integral_constant<size_t, i>) {
    if (!ok) return;
    std::printf("Starting double tall skinny test %zu rows by 2 cols\n", i);
    if (!fuzz_test<double, i, 2>(1e-10)) {
      std::fprintf(stderr,
                   "double tall skinny fuzz test failed for size %zux2\n", i);
      ok = false;
      return;
    }

    std::printf("Starting float tall skinny test %zu rows by 2 cols\n", i);
    if (!fuzz_test<float, i, 2>(1e-3)) {
      std::fprintf(stderr, "float tall skinny fuzz test failed for size %zux2\n",
                   i);
      ok = false;
    }
  });

  return ok ? 0 : -1;
}
