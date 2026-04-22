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

#include <functional>

#include "linalg/linalg_util.hpp"

namespace linalg {

/**
 * @brief P(x) : J -> J
 *
 * Quadratic rep of Jordan Algebra as a hessian
 *
 * P(x)^{-1} = P(x^{-1})... valid automorphism
 *
 * P(x) -> 2L(x)^2 - L(x)^2
 * P(x) : J -> J... (linear op / automorphism of the cone)
 * L(x) : J -> J... (linear op)
 * x \circ y -> J x J -> J... (bilinear map)
 */
template <SquareMatrixLike JxJ, typename Circ>
struct QuadraticRep {
  /// @brief extract type & sizes
  using T = mat_type_t<JxJ>;
  static constexpr size_t N = mat_rows_v<JxJ>;
  using J = Vector<T, N>;

  /**
   * @brief CTOR
   *
   * @param x \in J
   */
  constexpr explicit QuadraticRep(const J& x_in) : x(x_in) {}

  /**
   * @brief apply
   *
   * P(x)y = 2L(x)^2y - L(x^2)y
   * = 2 L(x) L(x) y - L(x^2) y
   * = 2 x circ (x circ y) - (x circ x) circ y
   *
   * @param y \in J
   *
   * @return P(x)y
   */
  constexpr auto apply(const J& y) const {
    const Circ circ;
    return T(2) * circ(x, circ(x, y)) - circ(circ(x, x), y);
  }

  /**
   * @brief apply column-wise to a matrix
   *
   * P(x)Y = [P(x)y_1 ... P(x)y_m]
   *
   * @param Y matrix with columns in J
   *
   * @return P(x)Y
   */
  template <MatrixLike MatX>
    requires(N == mat_rows_v<MatX>)
  constexpr auto apply(const MatX& Y) const {
    constexpr size_t Cols = mat_cols_v<MatX>;
    Matrix<T, N, Cols> result{};
    for (size_t j = 0; j < Cols; ++j) {
      const auto y_col = get_col(Y, j);
      const auto p_y_col = apply(y_col);
      for (size_t i = 0; i < N; ++i) result(i)(j) = p_y_col(i);
    }
    return result;
  }

  /** @brief member var */
  J x; ///< x in domain of P (i.e., P(x))
};

} // namespace linalg
