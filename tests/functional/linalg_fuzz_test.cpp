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
#include <charconv>
#include <cmath>
#include <string_view>
#include <cstdio>
#include <random>

#include "math.hpp"

namespace functional_test {

constexpr std::size_t DefaultTrials = 100;
constexpr std::size_t MaxOrder = 8;
constexpr unsigned Seed = 42;
constexpr std::size_t MaxDiagnosticsPerShape = 3;

using namespace linalg;

template <typename T>
void print_scalar(const T& x) {
  if constexpr (is_complex_v<T>)
    std::printf("(%g, %g) ", static_cast<double>(x.real()),
                static_cast<double>(x.imag()));
  else std::printf("%g ", static_cast<double>(x));
}

template <typename T, std::size_t Rows, std::size_t Cols>
void print_mat(const Matrix<T, Rows, Cols>& A) {
  for (std::size_t row = 0; row < Rows; ++row) {
    for (std::size_t col = 0; col < Cols; ++col) print_scalar(A(row, col));
    std::printf("\n");
  }
}

template <typename T, std::size_t Rows, std::size_t Cols>
auto make_A(std::mt19937& rng, std::uniform_real_distribution<T>& dist) {
  Matrix<T, Rows, Cols> A;
  for (std::size_t i = 0; i < Rows; ++i)
    for (std::size_t j = 0; j < Cols; ++j) A(i, j) = dist(rng);
  return A;
}

template <typename T, std::size_t N>
auto make_vec(std::mt19937& rng, std::uniform_real_distribution<T>& dist) {
  Vector<T, N> b;
  for (std::size_t i = 0; i < N; ++i) b(i) = dist(rng);
  return b;
}

template <typename T, std::size_t Rows, std::size_t Cols>
  requires(Field<T> && !is_complex_v<T>)
std::size_t check_decompositions(real_type_t<T> tol, std::size_t trials) {
  std::mt19937 rng(Seed);
  std::size_t failures = 0, failed_matrices = 0;
  std::uniform_real_distribution<T> dist(-10.0, 10.0);
  for (std::size_t i = 0; i < trials; ++i) {
    const auto A = make_A<T, Rows, Cols>(rng, dist);
    bool failed = false;
    auto fail = [&](const char* operation) {
      ++failures;
      failed = true;
      if (failed_matrices >= MaxDiagnosticsPerShape) return;
      std::fprintf(
          stderr,
          "seed=%u scalar=%s rows=%zu cols=%zu trial=%zu check=%s tol=%g\n",
          Seed, std::same_as<T, float> ? "float" : "double", Rows, Cols, i,
          operation, static_cast<double>(tol));
      print_mat(A);
    };
    const auto res = QR(A);
    const auto Q = res.Q();
    const auto R = res.R();
    static_assert(std::same_as<decltype(Q), const Matrix<T, Rows, Rows>>);
    static_assert(std::same_as<decltype(R), const Matrix<T, Rows, Cols>>);
    const auto A_prime = Q * R;
    if (!all_equal(A_prime, A, tol)) fail("QR reconstruction");

    if (!all_equal(transpose(Q) * Q, make_identity<T, Rows>(), tol))
      fail("QR orthogonality");

    if constexpr (Rows == Cols) {
      const auto eig_decomp = EigenDecomposition(A);
      if (!eig_decomp.converged) fail("eigen convergence");
      else {
        T scale = 1;
        for (const auto& value : A)
          scale = std::max(scale, constexpr_abs(value));
        if (!all_equal(as_complex_matrix(A) * eig_decomp.V,
                       eig_decomp.V * make_diag(eig_decomp.lambda),
                       tol * scale))
          fail("eigenpair residual");
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

    // Both successful convergence and reconstruction are required. The former
    // condition used && with a positive equality check, hiding bad SVD results.
    if (!svd.converged ||
        !all_equal(as_complex_matrix(A), U * (as_complex_matrix(S) * VT), tol))
      fail("SVD convergence or reconstruction");

    // S is rectangular: inspect its diagonal, not structural off-diagonal
    // zeros.
    for (std::size_t j = 0; j < std::min(Rows, Cols); ++j)
      if (!std::isfinite(S(j, j)) || S(j, j) < 0)
        fail("finite nonnegative singular values");

    const auto x = make_vec<T, Cols>(rng, dist);
    const auto b = A * x;

    const auto [x_tilde, flag] = res.solve(b);
    static_cast<void>(flag);
    const auto lsq_fit = A * x_tilde;
    const auto residual = euclidean_distance(lsq_fit, b);
    if (!std::isfinite(residual) || residual > tol)
      fail("least-squares residual");
    failed_matrices += failed;
  }
  std::printf("scalar=%s rows=%zu cols=%zu: %zu/%zu matrices passed, "
              "failed_checks=%zu\n",
              std::same_as<T, float> ? "float" : "double", Rows, Cols,
              trials - failed_matrices, trials, failures);

  return failures;
}

} // namespace functional_test

int main(int argc, char** argv) {
  using namespace functional_test;
  std::size_t trials = DefaultTrials;
  if (argc > 2) return 2;
  if (argc == 2) {
    const std::string_view argument(argv[1]);
    const auto parsed = std::from_chars(
        argument.data(), argument.data() + argument.size(), trials);
    if (parsed.ec != std::errc{} ||
        parsed.ptr != argument.data() + argument.size() || trials == 0) {
      std::fprintf(stderr, "usage: linalg_fuzz_test [positive trial count]\n");
      return 2;
    }
  }
  std::size_t failures = 0, matrices = 0;
  auto check = [&]<std::size_t Rows, std::size_t Cols>() {
    failures += check_decompositions<double, Rows, Cols>(1e-10, trials);
    failures += check_decompositions<float, Rows, Cols>(1e-3f, trials);
    matrices += 2 * trials;
  };
  // Dense square matrices exercise general eigen convergence as well as QR,
  // SVD and consistent least-squares solves. Tall matrices skip eigen analysis.
  constexpr_for<2, MaxOrder>(
      [&]<std::size_t N>(std::integral_constant<std::size_t, N>) {
        check.template operator()<N, N>();
      });
  constexpr_for<3, MaxOrder>(
      [&]<std::size_t N>(std::integral_constant<std::size_t, N>) {
        check.template operator()<N, 2>();
      });
  std::printf("linalg_fuzz_test: seed=%u matrices=%zu failed_checks=%zu "
              "(QR, eigen convergence, SVD, least squares)\n",
              Seed, matrices, failures);
  return failures == 0 ? 0 : 1;
}
