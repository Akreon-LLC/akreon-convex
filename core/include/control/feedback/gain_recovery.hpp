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

#include "linalg/operator/matrix_inverse.hpp"

namespace control {

/**
 * @brief recover state-feedback gain from a lifted gain variable
 *
 * For standard state feedback and H2 state feedback, the lifted variable has
 * the same form:
 *
 *   Y = K Q
 *   K = Y Q^{-1}
 *
 * @param Q symmetric positive-definite Lyapunov/Gramian block
 * @param Y lifted feedback product
 *
 * @return tuple(K, numerical event)
 */
template <linalg::SquareMatrixLike QMat, linalg::MatrixLike YMat>
constexpr auto recover_state_feedback_gain(const QMat& Q, const YMat& Y) {
  using T = linalg::mat_type_t<QMat>;
  static constexpr size_t InputDim = linalg::mat_rows_v<YMat>;
  static constexpr size_t StateDim = linalg::mat_cols_v<YMat>;
  static_assert(linalg::mat_rows_v<QMat> == StateDim,
                "Q row dimension must match Y column dimension");
  static_assert(linalg::mat_cols_v<QMat> == StateDim,
                "Q column dimension must match Y column dimension");

  using KMat = linalg::Matrix<T, InputDim, StateDim>;
  const auto [Q_inv, event] = linalg::SymmetricMatrixInverse<QMat>()(Q);
  if (event != linalg::NumericalEvent::NONE) return std::tuple(KMat{}, event);
  const auto K = Y * Q_inv;
  return std::tuple(K, linalg::NumericalEvent::NONE);
}

} // namespace control
