/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstring>
#include <iostream>

#include "convex/debug/pd_ipm_format.hpp"
#include "math.hpp"

namespace unit_test {

/// @brief eps for testing
constexpr double eps = 1e-8;

/**
 * @brief log
 *
 * @param res
 * @param trace
 *
 * @return void
 */
inline constexpr void log(const auto& res, const auto& trace) {
  static_cast<void>(trace);
  char buf[64000];
  convex::debug::result_to_cstr(res, buf, sizeof(buf));
  std::cout << buf << std::endl;
}

/**
 * @brief log mat
 *
 * @param A (mat)
 *
 * @return void
 */
template <linalg::MatrixContainer Mat>
inline constexpr void log_mat(const Mat& A) {
  constexpr std::size_t M = linalg::mat_rows_v<Mat>;
  constexpr std::size_t N = linalg::mat_cols_v<Mat>;
  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) std::cout << A(i, j) << " ";
    std::cout << std::endl;
  }
}

/**
 * @brief log vector
 *
 * @param b (vec)
 *
 * @return void
 */
template <linalg::VectorContainer Vec>
inline constexpr void log_vec(const Vec& b) {
  constexpr std::size_t N = linalg::vector_dim_v<Vec>;
  for (std::size_t i = 0; i < N; ++i) std::cout << b(i) << " ";
  std::cout << std::endl;
}

} // namespace unit_test
