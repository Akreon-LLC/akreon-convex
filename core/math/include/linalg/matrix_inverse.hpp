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

#pragma once

#include "linalg_util.hpp"

namespace linalg {

/**
 * @brief matrix inverse
 *
 * A A^{-1} = I
 *
 * @param matrix A of type T N x N elements
 *
 * @return matrix
 */
template <SquareMatrixLike Mat>
constexpr auto inverse(const Mat& A) {
  using T = mat_type_t<Mat>;
  constexpr auto N = mat_rows_v<Mat>;
  static_assert(N >= 1);

  const auto I = make_identity<T, N>();
  Matrix<T, N, N> res;
  const auto qr_A = QR(A);
  for (size_t i = 0; i < N; ++i) {
    const auto x = qr_A.solve(get_col(I, i));
    for (size_t j = 0; j < N; ++j) res[j][i] = x(j);
  }
  return res;
}

/**
 * @brief operator overload for matrix inverse
 *
 * @param matrix A of type T
 * @param matrix B of type T
 *
 * @return matrix
 */
template <MatrixLike MatA, MatrixLike MatB>
  requires(std::same_as<mat_type_t<MatA>, mat_type_t<MatB>>)
constexpr auto operator/(const MatA& A, const MatB& B) {
  static_assert(mat_cols_v<MatB> == mat_cols_v<MatA>);
  return A * inverse(B);
}

}  // namespace linalg
