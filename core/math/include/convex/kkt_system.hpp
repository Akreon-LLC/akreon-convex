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

#include <iostream>

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
  constexpr explicit KKTSystem(Mat A, ScalarT tol = 1e-8) : SymmetricBase(A) {
    (void)tol;
    //factor(A, tol);
  }

  /**
   * @brief factor via LDL^T with tikhonov regularization
   *
   * @param Mat A
   * @param tol
   *
   * @return void
   */
  constexpr void factor(const Mat& A, const ScalarT& tol) {
    /// (dakre) apply tikhonov regularization up front
    // A = A + tol * make_identity<T, N>();

    /// (dakre) lambda function for computing LDL^T (in-place in A)
    bool res = ldl_t(A, tol, true);
#if 0
    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < N; ++j) std::cout << A(i)(j) << " ";
      std::cout << std::endl;
    }
#endif

    /// (dakre) check if LDL^T completed
    if (res) {
      this->type = linalg::SymmetricOperatorType::INV;
      return;
    }

    // L = lower_triangle_to_vec(A);

    // (dakre) KKT system is indefinite by nature
    this->type = linalg::SymmetricOperatorType::IND;
  }

  /**
   * @brief solve
   *
   * L^T x = b via foward solve
   *
   * @param vec b
   *
   * @return vec x
   */
  constexpr linalg::Vector<T, N> solve(const linalg::Vector<T, N>& b) const {
    linalg::Vector<T, N> x;
    for (size_t i = 0; i < N; ++i) {
      T sum = 0;
      for (size_t j = 0; j < i; ++j) sum += (L(this->index(i, j)) * x(j));

      assert(L(this->index(i, i)) != T(0));
      x(i) = (b(i) - sum) / L(this->index(i, i));
    }
    return x;
  }

  /** @brief member var */
  linalg::Vector<T, DIM> L;  ///< L from LDL^T factorization
};

}  // namespace convex
