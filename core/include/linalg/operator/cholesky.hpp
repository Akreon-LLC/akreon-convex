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

#include "config/assert.hpp"
#include "linalg/linalg_util.hpp"
#include "symmetric_operator.hpp"

namespace linalg {

/**
 * @brief A = GG^T cholesky decomposition where A in S^n_++
 */
template <SquareMatrixLike Mat>
struct Cholesky : SymmetricOperator<Mat, Cholesky<Mat>> {
  /// @brief alias for symmetric operators
  using SymmetricBase = SymmetricOperator<Mat, Cholesky<Mat>>;
  using typename SymmetricBase::ComplexT;
  using typename SymmetricBase::ScalarT;
  using typename SymmetricBase::T;

  /// @brief helpers for symmetric matrix
  using SymmetricBase::DIM;
  using SymmetricBase::N;

  static_assert(N > 1, "GG^T requires as least 2x2 matrix");

  /**
   * @brief CTOR
   *
   * @param A (matrix)
   */
  constexpr explicit Cholesky(const Mat& A) : SymmetricBase(A) { factor(A); }

  /**
   * @brief CTOR
   *
   * @param A (matrix)
   * @param config (FactorConfig)
   */
  constexpr explicit Cholesky(const Mat& A,
                              const config::FactorConfig<T>& config_in)
      : SymmetricBase(A), config(config_in) {
    factor(A);
  }

  /**
   * @brief G
   *
   * @return G
   */
  constexpr auto G() const { return make_lower_triangle<Mat>(G_vec); }

private:
  /**
   * @brief GG^T of A in S^n_++
   *
   * @param A (matrix)
   *
   * @return void
   */
  constexpr void factor(Mat A) {
    auto chol = [&]<size_t j = 0>(this auto&& self) {
      if constexpr (j >= SymmetricBase::N) {
        return;
      } else {
        constexpr size_t next = j + 1;
        if constexpr (j > 0) {
          const auto A_jn_j = get_block<j, SymmetricBase::N, j, next>(A);
          const auto A_jn_0j = get_block<j, SymmetricBase::N, 0, j>(A);
          const auto A_j_0j = get_block<j, next, 0, j>(A);
          const auto lhs = A_jn_j - A_jn_0j * transpose(A_j_0j);
          A = set_block<j, SymmetricBase::N, j, next>(A, lhs);
        }
        const auto A_jn_j = get_block<j, SymmetricBase::N, j, next>(A);
        const auto a_jj = constexpr_sqrt_safe(A(j)(j));
        A = set_block<j, SymmetricBase::N, j, next>(A, T(1) / a_jj * A_jn_j);
        return self.template operator()<next>();
      }
    };

    chol();

    G_vec = get_lower_triangle_with_diag(A);

    this->compute_definiteness(get_diag(A), config.tol);
  }

  /** @brief member vars */
  config::FactorConfig<T> config; ///< factor config
  Vector<T, DIM> G_vec;           ///< lower left triangle of chol
};

} // namespace linalg
