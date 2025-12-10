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

#include "algebra/algebra_traits.hpp"

namespace linalg {

/// @brief forward declaration for a matrix
template <SquareMatrixLike Mat>
struct SymmetricMatrix;

/// @brief type traits definition for symmetric matrices (subspace of R^n)
template <SquareMatrixLike Mat>
struct space_traits<SymmetricMatrix<Mat>> {
  using value_type = mat_type_t<Mat>;
  using Point = Mat;

  static constexpr size_t N = mat_rows_v<Mat>;
  static constexpr size_t DIM = N * (N + 1) / 2;
};

/// @brief mat type traits definition for symmetric matrices
template <SquareMatrixLike Mat>
struct mat_traits<SymmetricMatrix<Mat>> {
  using value_type = mat_type_t<Mat>;
  static constexpr size_t ROWS = mat_rows_v<Mat>;
  static constexpr size_t COLS = mat_cols_v<Mat>;
};

/// @brief vec type traits for symmetric matrices
template <SquareMatrixLike Mat>
struct vec_traits<SymmetricMatrix<Mat>> {
  using value_type = mat_type_t<Mat>;
  static constexpr size_t N = mat_rows_v<Mat>;
  static constexpr size_t SIZE = N * (N + 1) / 2;
};

}  // namespace linalg
