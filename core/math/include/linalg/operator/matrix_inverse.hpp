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

#include "ldl.hpp"
#include "linalg/linalg_util.hpp"
#include "linalg/spectral/qr.hpp"

namespace linalg {

/**
 * @brief Matrix inverse via QR
 */
template <MatrixLike MatIn>
struct Inverse {
  using Mat = linalg::remove_cvref_t<MatIn>;
  using T = mat_type_t<Mat>;
  static constexpr size_t N = mat_rows_v<Mat>;
  static_assert(N >= 1, "inverse requires matrix dim of at least 1");

  /**
   * @brief operator
   *
   * I = A^{-1} A
   *
   * @param A (matrix)
   *
   * @return A^{-1}
   */
  constexpr auto operator()(const Mat& A) const { return apply(A); }

  /**
   * @brief apply
   *
   * @param A (matrix)
   *
   * @return A^{-1}
   */
  constexpr auto apply(const Mat& A) const {
    const auto qr = QR(A);
    Mat A_inv{};
    for (size_t i = 0; i < N; ++i) {
      Vector<T, N> I_vec{};
      I_vec(i) = T(1);
      const auto a_col = qr.solve(I_vec);
      for (size_t j = 0; j < N; ++j) A_inv(j)(i) = a_col(j);
    }
    return A_inv;
  }
};

/**
 * @brief Symmetric matrix inverse via LDL^T
 */
template <SquareMatrixLike MatIn>
struct SymmetricInverse {
  using Mat = linalg::remove_cvref_t<MatIn>;
  using T = mat_type_t<Mat>;
  static constexpr size_t N = mat_rows_v<Mat>;
  static_assert(N >= 1, "inverse requires matrix dim of at least 1");

  /**
   * @brief operator
   *
   * I = A^{-1} A
   *
   * @param A (matrix)
   *
   * @return A^{-1}
   */
  constexpr auto operator()(const Mat& A) const { return apply(A); }

  /**
   * @brief apply
   *
   * @param A (matrix)
   *
   * @return A^{-1}
   */
  constexpr auto apply(const Mat& A) const {
    const auto ldl = LDL(A);

    AK_ASSERT(ldl.is_valid() &&
              "Input matrix is not a valid symmetric matrix for LDL^T");

    Mat A_inv{};
    for (size_t i = 0; i < N; ++i) {
      Vector<T, N> I_vec{};
      I_vec(i) = T(1);
      const auto a_col = ldl.solve(I_vec);
      for (size_t j = 0; j < N; ++j) A_inv(j)(i) = a_col(j);
    }
    return A_inv;
  }
};

}  // namespace linalg
