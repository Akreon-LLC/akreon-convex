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
void print_A(const Matrix<T, Rows, Cols>& A) {
  std::cout << "Matrix A values" << std::endl;
  for (const auto& row : A) {
    for (const auto& pt : row) {
      std::cout << pt << " ";
    }
    std::cout << std::endl;
  }
}

template <typename T, size_t Rows, size_t Cols>
auto make_A(std::mt19937& rng, std::uniform_real_distribution<T>& dist) {
  Matrix<T, Rows, Cols> A;
  for (size_t i = 0; i < Rows; ++i)
    for (size_t j = 0; j < Cols; ++j) A[i][j] = dist(rng);
  return A;
}

template <typename T, size_t Rows, size_t Cols>
  requires(FieldLike<T> && !is_complex_v<T>)
void fuzz_test_qr(const scalar_type_t<T>& tol) {
  std::mt19937 rng(42);
  std::uniform_real_distribution<T> dist(-10.0, 10.0);
  for (size_t i = 0; i < TRIALS; ++i) {
    const auto A = make_A<T, Rows, Cols>(rng, dist);
    const auto res = QR(A);
    const auto Q = res.get_Q();
    const auto R = res.get_R();
    static_assert(std::same_as<decltype(Q), const Matrix<T, Rows, Rows>>);
    static_assert(std::same_as<decltype(R), const Matrix<T, Rows, Cols>>);
    const auto A_prime = Q * R;
    if (!all_equal(A_prime, A, tol)) {
      std::cerr << "Test case failed for tolerance " << tol << std::endl;
      print_A(A_prime);
      print_A(A);
      throw std::runtime_error("A_prime != A");
    }
  }
}

int main() {
  try {
    std::cout << "Starting qr double test" << std::endl;
    fuzz_test_qr<double, 3, 3>(1e-10);
    std::cout << "Starting qr float test" << std::endl;
    // (dakre) TODO float precision is not so good, partly to be expected but
    // householder reflection qr may help
    fuzz_test_qr<float, 3, 3>(1e-3);
    std::cout << "Starting qr double 3x2 test" << std::endl;
    fuzz_test_qr<double, 3, 2>(1e-10);
    std::cout << "Starting qr double 2x3 test" << std::endl;
    fuzz_test_qr<double, 2, 3>(1e-10);
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    return -1;
  }

  return 0;
}
