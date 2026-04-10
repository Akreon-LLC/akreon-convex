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

namespace linalg {

template <SquareMatrixLike Mat, size_t Dim>
struct MatrixPencil {
  /// @brief extracting alias and types
  using T = mat_type_t<Mat>;
  using Fs = Vector<Mat, Dim>;

  /**
   * @brief apply_full
   *
   * Fs = { F_i }
   *
   * @param op (Fn)
   *
   * @return Fs
   */
  template <MatrixLike FullMat, typename Fn>
  constexpr auto apply_full(const Fn& op) const {
    static constexpr size_t Rows = mat_rows_v<FullMat>;
    static constexpr size_t Cols = mat_cols_v<FullMat>;
    Fs F_blks{};
    for (size_t i = 0, idx = 0; i < Rows; ++i) {
      for (size_t j = 0; j < Cols; ++j, ++idx) {
        FullMat E{};
        E(i)(j) = T(1);
        F_blks(idx) = op(E);
      }
    }
    return F_blks;
  }

  /**
   * @brief apply symmetric matrix basis
   *
   * Fs = { F_i }
   *
   * @param op (Fn)
   *
   * @return Fs
   */
  template <SquareMatrixLike SymMat, typename Fn>
  constexpr auto apply_sym_mat(const Fn& op) const {
    static constexpr size_t N = mat_rows_v<SymMat>;

    Fs F_blks{};
    for (size_t i = 0, idx = 0; i < N; ++i) {
      for (size_t j = 0; j < N; ++j) {
        SymMat E{};
        if (i == j)
          E(i)(j) = T(1);
        else if (i < j)
          E(i)(j) = E(j)(i) = T(1) / linalg::constexpr_sqrt_safe(T(2));
        else
          continue;

        F_blks(idx++) = op(E);
      }
    }
    return F_blks;
  }
};

}  // namespace linalg
