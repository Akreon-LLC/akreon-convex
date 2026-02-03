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

#include "linalg/vector/algebra_traits.hpp"

namespace linalg {

/// @brief type traits for kkt_mat
template <typename H, typename A>
struct kkt_traits;

/**
 * @brief KKT Type traits
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
template <linalg::SquareMatrixLike Hessian, linalg::MatrixLike Equality>
  requires(std::same_as<mat_type_t<Hessian>, mat_type_t<Equality>>)
struct kkt_traits<Hessian, Equality> {
  using T = mat_type_t<Hessian>;
  static constexpr size_t HRows = mat_rows_v<Hessian>;
  static constexpr size_t ERows = mat_rows_v<Equality>;
  static constexpr size_t ECols = mat_cols_v<Equality>;
  static constexpr size_t N = HRows + ERows;
  using Mat = linalg::Matrix<T, N, N>;

  /**
   * @brief make kkt mat
   *
   * @param Hessian H
   * @param Equality A
   *
   * @return KKT mat
   */
  constexpr auto make(const Hessian& H, const Equality& A) const {
    Mat kkt;
    kkt = set_block<0, HRows, 0, HRows>(kkt, H);
    kkt = set_block<0, ECols, HRows, N>(kkt, transpose(A));
    kkt = set_block<HRows, N, 0, ECols>(kkt, A);
    return kkt;
  }

  /**
   * @brief tikhonov regularization
   *
   * @param KKT mat
   * @param delta applied to H
   * @param delta lower block -- adding damping to Ax + c nu = 0
   *
   * @return updated KKT
   */
  constexpr auto regularize(Mat kkt, const T& delta, const T& c) const {
    for (size_t i = 0; i < N / 2; ++i) kkt(i)(i) += delta;
    for (size_t i = N / 2; i < N; ++i) kkt(i)(i) += c;
    return kkt;
  }

  /**
   * @brief H block
   *
   * @param KKT mat
   *
   * @return H from kkt
   */
  constexpr auto H_block(const Mat& kkt) const {
    return get_block<0, HRows, 0, HRows>(kkt);
  }

  /**
   * @brief A block
   *
   * @param KKT mat
   *
   * @return A from kkt
   */
  constexpr auto A_block(const Mat& kkt) const {
    return get_block<HRows, N, 0, ECols>(kkt);
  }
};

}  // namespace linalg
