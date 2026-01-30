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

#include "convex_traits.hpp"

namespace convex {

/**
 * @brief KKT Operator
 *
 * Establish conditions for optimality of general convex (differentiable)
 * problem:
 * \nabla f_0(x^*) + \sum_i^m \lambda_i \nabla f_i(x^*) + \sum_i^p \nu_i \nabla
 * h_i(x^*) = 0
 *
 * (i)   f_i(x^*) <= 0
 * (ii)  h_i(x^*) = 0
 * (iii) \lambda_i^* >= 0
 * (iv)  \lambda_i^* f_i(x^*) = 0
 * (v)   \nabla f_0(x^*) + \sum_i^m \lambda_i^* \nabla f_i(x^*) + \sum_i^p \nu_i
 * \nabla h_i(x^*) = 0
 *
 * KKT System
 * [ H A^T ] [ x; nu ] = [ -q; b ]
 * [ A 0   ]
 */
template <linalg::SquareMatrixLike Mat>
struct KKTSystem : linalg::SymmetricOperator<Mat, KKTSystem<Mat>> {
  using SymmetricBase = linalg::SymmetricOperator<Mat, KKTSystem<Mat>>;
  /// @brief alias to bring in from symmetric base
  using typename SymmetricBase::ScalarT;
  using typename SymmetricBase::T;

  /// @brief helper for N and DIM
  using SymmetricBase::DIM;
  using SymmetricBase::N;

  /**
   * @brief CTOR
   *
   * @param A (mat)
   * @param tolerance
   */
  constexpr explicit KKTSystem(const Mat& A, const ScalarT& tol)
      : SymmetricBase(A, tol) {}

  /**
   * @brief factor via LDL^T with tikhonov regularization
   *
   * @param Mat A
   * @param tol
   *
   * @return void
   */
  constexpr void factor(Mat A, const ScalarT& tol) {
    /// (dakre) apply tikhonov regularization up front
    // A = A + tol * make_identity<T, N>();

    /// (dakre) lambda function for computing LDL^T
    auto ldl_t = [&]<size_t j = 1>(const bool& status) -> bool {
      if (j >= N) return status;

      linalg::Vector<T, j> v{};
      for (size_t i = 0; i < j - 1; ++i) v[i] = A(j)(i) * A(i)(i);

      const auto A_row = linalg::get_block<j, j + 1, 0, j>(A);
      A[j][j] = A[j][j] - (A_row * v)(0);

      if (A(j)(j) < tol) {
        return false;
      }

      const auto A_col = linalg::get_block<j, N, j, j + 1>(A);
      const auto A_blk = linalg::get_block<j, N, 0, j>(A);
      const auto col_up =
          A_col - (T(1) / A(j)(j)) * linalg::as_col_vector(A_blk * v);
      A = linalg::set_block<j, N, j, j + 1>(A, col_up);

      return status;
    };

    /// (dakre) check if LDL^T completed
    if (!ldl_t(true)) {
      this->type = linalg::SymmetricMatrixType::INV;
      return;
    }

    // (dakre) store lower triangle
    this->L = lower_triangle_to_vec(A);

    // (dakre) KKT system is indefinite by nature
    this->type = linalg::SymmetricMatrixType::IND;
  }
};

}  // namespace convex
