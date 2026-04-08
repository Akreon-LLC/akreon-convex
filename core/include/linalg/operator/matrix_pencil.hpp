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
   * @brief operator overload (apply matrix pencil)
   *
   * Fs = { F_i }
   *
   * @return Fs
   */
  template <MatrixLike InnerMat, typename Fn>
  constexpr auto operator()(const Fn& op) const {
    return this->template apply<InnerMat>(op);
  }

  /**
   * @brief apply
   *
   * Fs = { F_i }
   *
   * @return Fs
   */
  template <MatrixLike InnerMat, typename Fn>
  constexpr auto apply(const Fn& op) const {
    static constexpr size_t N = mat_rows_v<InnerMat>;
    Fs F_blks{};
    for (size_t i = 0, idx = 0; i < N; ++i) {
      for (size_t j = 0; j < N; ++j) {
        InnerMat E{};
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
