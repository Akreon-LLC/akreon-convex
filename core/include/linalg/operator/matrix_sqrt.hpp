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

#include "config/solver_config.hpp"
#include "linalg/linalg_util.hpp"
#include "linalg/operator/matrix_inverse.hpp"

namespace linalg {

/**
 * @brief Matrix sqrt (X^{1/2})
 */
template <SquareMatrixLike MatIn>
struct MatrixSqrt {
  /// @brief alias & types
  using Mat = linalg::remove_cvref_t<MatIn>;
  using T = mat_type_t<Mat>;
  static constexpr size_t N = mat_rows_v<Mat>;

  static_assert(RealLike<T>, "Sqrt assumes reals domain");
  static_assert(N >= 1, "Sqrt requires matrix dim of at least 1");

  /**
   * @brief operator
   *
   * @param A (matrix)
   *
   * @return (A^{1/2}, A^{-1/2}, event)
   */
  constexpr auto operator()(const Mat& A) const { return apply(A); }

  /**
   * @brief apply
   *
   * Denman-Beavers iteration (quadratic convergence)
   *
   * @param A (matrix)
   *
   * @return (A^{1/2}, A^{-1/2}, event)
   */
  constexpr auto apply(const Mat& A) const {
    Mat Y{A};
    Mat Z{make_identity<T, N>()};

    for (size_t i = 0; i < config.steps; ++i) {
      const auto [Y_inv, Y_event] = MatrixInverse<Mat>()(Y);
      if (Y_event != linalg::NumericalEvent::NONE)
        return std::tuple(Mat{}, Mat{}, Y_event);

      const auto [Z_inv, Z_event] = MatrixInverse<Mat>()(Z);
      if (Z_event != linalg::NumericalEvent::NONE)
        return std::tuple(Mat{}, Mat{}, Z_event);

      const auto Y_next = T(0.5) * (Y + Z_inv);
      const auto Z_next = T(0.5) * (Z + Y_inv);

      if (all_equal(Y_next * Y_next, A, config.tol))
        return std::tuple(Y_next, Z_next, linalg::NumericalEvent::NONE);

      Y = Y_next;
      Z = Z_next;
    }

    return std::tuple(Y, Z, linalg::NumericalEvent::DB_MAX_ITERS_REACHED);
  }

  /** @brief member vars **/
  config::FactorConfig<T> config{
      .steps = 128, .tol = T(1e-12), .floor_eps = T(1e-14)}; ///< factor config
};

} // namespace linalg
