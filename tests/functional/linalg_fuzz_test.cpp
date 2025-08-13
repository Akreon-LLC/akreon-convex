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
  requires(FieldLike<T>)
void fuzz_test_qr(std::mt19937& rng, std::uniform_real_distribution<T>& dist) {
  for (size_t i = 0; i < TRIALS; ++i) {
    const auto A = make_A<T, Rows, Cols>(rng, dist);
    const auto res = QR(A);
    const auto A_prime = res.get_Q() * res.get_R();
    if (!all_equal(A_prime, A)) throw std::runtime_error("A_prime != A");
  }
}

int main() {
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist(-10.0, 10.0);

  try {
    fuzz_test_qr<double, 3, 3>(rng, dist);
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    return -1;
  }

  return 0;
}
