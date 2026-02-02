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

#include "linalg/linalg_util.hpp"
#include "symmetric_operator.hpp"

namespace linalg {

/**
 * @brief LDL^T factorization of a square / symmetric matrix A
 */
template <SquareMatrixLike Mat>
struct LDL : SymmetricOperator<Mat, LDL<Mat>> {
  /// @brief alias for symmetric matrix
  using SymmetricBase = SymmetricOperator<Mat, LDL<Mat>>;
  using typename SymmetricBase::ComplexT;
  using typename SymmetricBase::ScalarT;
  using typename SymmetricBase::T;

  /// @brief helpers for symmetric matrix
  using SymmetricBase::DIM;
  using SymmetricBase::N;

  static_assert(N > 1, "LDL^T requires at least a 2x2 matrix");

  /**
   * @brief CTOR
   *
   * @param Matrix A
   * @param Scalar tolerance
   *
   * @param matrix A
   */
  constexpr explicit LDL(const Mat& A, ScalarT tol = 1e-8) : SymmetricBase(A) {
    factor(A, tol);
  }

  /**
   * @brief D
   *
   * @return D diag mat
   */
  constexpr auto D() const { return make_diag(D_vec); }

  /**
   * @brief L
   *
   * @return L mat
   */
  constexpr auto L() const { return make_unit_lower_triangle<Mat>(L_vec); }

 private:
  /**
   * @brief LDL^T of a square matrix
   *
   * @param Matrix A
   * @param Scalar tolerance
   *
   * @return void
   */
  constexpr void factor(Mat A, const ScalarT& tol) {
    auto ldl_t = [&]<size_t j = 0>(this auto&& self,
                                   const bool& status) -> bool {
      /// (dakre) base case for recursion
      if constexpr (j >= N)
        return status;
      else {
        /// (dakre) handle j = 0 iteration separately (don't have prior intertia
        /// build up, only need to scale first col by first diag elem)
        constexpr size_t next = j + 1;
        if constexpr (j == 0) {
          if (linalg::constexpr_abs(A(0)(0)) < tol) return false;

          const auto A_col = linalg::get_block<next, N, 0, next>(A);
          const auto col_up = (T(1) / A(0)(0)) * A_col;

          A = linalg::set_block<next, N, 0, next>(A, col_up);
        } else {
          /// (dakre) nominal LDL^T algo denoted in Golub & Van Load p. 158
          linalg::Vector<T, j> v{};
          for (size_t k = 0; k < j; ++k) v(k) = A(j)(k) * A(k)(k);

          const auto A_row = linalg::get_block<j, next, 0, j>(A);
          A(j)(j) = A(j)(j) - (A_row * v)(0);

          if (linalg::constexpr_abs(A(j)(j)) < tol) return false;

          if constexpr (next < N) {
            const auto A_col = linalg::get_block<next, N, j, next>(A);
            const auto A_blk = linalg::get_block<next, N, 0, j>(A);

            const auto col_up =
                T(1) / A(j)(j) * (A_col - linalg::as_col_vector(A_blk * v));

            A = linalg::set_block<next, N, j, next>(A, col_up);
          }
        }
        return self.template operator()<next>(status);
      }
    };

    if (!ldl_t(true)) {
      this->type = SymmetricOperatorType::INV;
      return;
    }

    L_vec = get_lower_triangle(A);
    D_vec = get_diag(A);

    this->compute_definiteness(D_vec, tol);
  }

  /** @brief member variables */
  Vector<T, DIM - N> L_vec;  ///< lower left triangle of factorization
  Vector<T, N> D_vec;        ///< diag of factorization
};

}  // namespace linalg
