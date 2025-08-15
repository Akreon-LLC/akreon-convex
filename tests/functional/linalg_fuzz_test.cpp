/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#include <algorithm>
#include <iostream>
#include <random>

#include "math.hpp"

#define TRIALS 1000

using namespace algebra;

template <typename T, size_t Rows, size_t Cols>
void print_mat(const Matrix<T, Rows, Cols>& A) {
  for (const auto& row : A) {
    for (const auto& pt : row) {
      std::cout << pt << " ";
    }
    std::cout << std::endl;
  }
}

template <typename T, size_t N>
void print_vec(const Vector<T, N>& b) {
  for (const auto& x : b) std::cout << x << " ";
  std::cout << std::endl;
}

template <typename T, size_t Rows, size_t Cols>
auto make_A(std::mt19937& rng, std::uniform_real_distribution<T>& dist) {
  Matrix<T, Rows, Cols> A;
  for (size_t i = 0; i < Rows; ++i)
    for (size_t j = 0; j < Cols; ++j) A[i][j] = dist(rng);
  return A;
}

template <typename T, size_t N>
auto make_vec(std::mt19937& rng, std::uniform_real_distribution<T>& dist) {
  Vector<T, N> b;
  for (size_t i = 0; i < N; ++i) b[i] = dist(rng);
  return b;
}
#if 0
template <typename T, size_t Rows, size_t Cols>
  requires(FieldLike<T> && !is_complex_v<T>)
void fuzz_test(const scalar_type_t<T>& tol) {
  std::mt19937 rng(42);
  std::uniform_real_distribution<T> dist(-10.0, 10.0);
  for (size_t i = 0; i < TRIALS; ++i) {
    const auto A = make_A<T, Rows, Cols>(rng, dist);
    const auto res = QR(A);
    const auto Q = res.get_Q();
    const auto R = res.get_R();
    const auto r = res.get_rank();
    static_assert(std::same_as<decltype(Q), const Matrix<T, Rows, Rows>>);
    static_assert(std::same_as<decltype(R), const Matrix<T, Rows, Cols>>);
    const auto A_prime = Q * R;
    if (!all_equal(A_prime, A, tol)) {
      std::cerr << "QR test case failed for tolerance " << tol << std::endl;
      std::cerr << "Matrix rank " << r << std::endl;
      print_mat(A_prime);
      print_mat(A);
      throw std::runtime_error("A_prime != A");
    }

    const auto svd = SVD(A);
    const auto U = svd.get_left_singular_vectors();
    const auto sigmas = svd.get_singular_values();
    const auto VT = svd.get_right_singular_vectors();
    static_assert(std::same_as<decltype(U), const Matrix<T, Rows, Rows>>);
    static_assert(std::same_as<decltype(VT), const Matrix<T, Cols, Cols>>);
    static_assert(std::same_as<decltype(sigmas), const Vector<T, Cols>>);

    //const auto A_svd = U * (make_diag(sigmas) * VT);
    //std::cout << all_equal(A, A_svd) << std::endl;

    T min_sv = 1000, max_sv = 0;
    for (const auto& x : sigmas) {
      min_sv = x < min_sv ? x : min_sv;
      max_sv = x > max_sv ? x : max_sv;
    }
    T k = max_sv / min_sv;

    const auto UTU = U * transpose(U);
    const auto VTV = transpose(VT) * VT;
    if (!all_equal(UTU, make_identity<T, Rows>()) ||
        !all_equal(VTV, make_identity<T, Cols>())) {
      std::cerr << "SVD test case failed to form orthonormal basis vectors"
                << std::endl;
      std::cerr << "Matrix rank " << r << std::endl;
      std::cerr << "Matrix condition number " << k << std::endl;
      print_mat(A);
      std::cerr << "Left singular vectors" << std::endl;
      print_mat(U);
      std::cerr << "Right singular vectors" << std::endl;
      print_mat(VT);
      throw std::runtime_error("Failed to perform SVD on A");
    }

    const auto x = make_vec<T, Cols>(rng, dist);
    const auto b = A * x;
    const auto x_tilde = res.solve(b);
    const auto lsq_fit = A * x_tilde;
    const auto normalized_residual =
        euclidean_distance(lsq_fit, b) / L2<T, Rows>(b);
    if (normalized_residual > tol) {
      std::cerr << "LSQ test case failed for tolerance " << tol << std::endl;
      std::cerr << "Normalized residual " << normalized_residual << std::endl;
      std::cerr << "Matrix rank " << r << std::endl;
      std::cerr << "Matrix condition number " << k << std::endl;
      print_mat(A);
      std::cerr << "Vector b" << std::endl;
      print_vec(b);
      std::cerr << "Vector lsq fit" << std::endl;
      print_vec(lsq_fit);
      throw std::runtime_error("lsq_fit != b");
    }
  }
}
#endif

int main() {
  try {
#if 0
    std::cout << "Starting double test" << std::endl;
    fuzz_test<double, 3, 3>(1e-10);
    // (dakre) TODO float precision is not so good, partly to be expected but
    // householder reflection qr may help
    std::cout << "Starting float test" << std::endl;
    fuzz_test<float, 3, 3>(1e-2);
    std::cout << "Starting double 3x2 test" << std::endl;
    fuzz_test<double, 3, 2>(1e-10);
    std::cout << "Starting double 2x3 test" << std::endl;
    fuzz_test<double, 2, 3>(1e-10);
#endif
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    return -1;
  }

  return 0;
}
