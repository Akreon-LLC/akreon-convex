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

#include "linalg/matrix_ops.hpp"

namespace linalg {

/**
 * @brief compute the frobenius norm of a matrix
 *
 * sqrt(tr(A^T A)) if A \in R else sqrt(tr(A^* A)) where A \in C
 *
 * @param matrix A Rows x Cols elements of type T (must be square)
 *
 * @return scalar
 */
template <SquareMatrixLike Mat>
struct Fro {
  using T = mat_type_t<Mat>;
  static constexpr size_t N = mat_rows_v<Mat>;

  constexpr auto operator()(const Mat& A) {
    scalar_type_t<T> result = 0;
    if constexpr (RealLike<T>) {
      result = trace(transpose(A) * A);
    } else {
      const auto tmp = trace(conjugate_transpose(A) * A);
      assert(is_real(tmp));
      result = tmp.real();
    }
    return constexpr_sqrt(result);
  }
};

}  // namespace linalg
