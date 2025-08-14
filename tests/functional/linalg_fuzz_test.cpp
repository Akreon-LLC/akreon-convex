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
    const auto A_prime = res.get_Q() * res.get_R();
    if (!all_equal(A_prime, A, tol)) throw std::runtime_error("A_prime != A");
  }
}

int main() {
  try {
    std::cout << "Starting double test" << std::endl;
    fuzz_test_qr<double, 3, 3>(1e-10);
    std::cout << "Starting float test" << std::endl;
    // (dakre) TODO float precision is not so good, partly to be expected but
    // householder reflection qr may help
    fuzz_test_qr<float, 3, 3>(1e-3);
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    return -1;
  }

  return 0;
}
