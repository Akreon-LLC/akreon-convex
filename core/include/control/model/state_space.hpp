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

#include "../control_traits.hpp"

#pragma once

namespace control {

/**
 * @brief StateSpace
 *
 * \dot{x} = A x(t) + B u(t)
 * y = C x(t) + D u(t)
 */
template <linalg::SquareMatrixLike Plant, linalg::MatrixLike Control,
          linalg::MatrixLike Output, linalg::MatrixLike Feedthrough,
          TimeDomainType Type>
struct StateSpace {
  /// @brief extract type and alias
  using T = linalg::mat_type_t<Plant>;
  static constexpr size_t PlantRows = linalg::mat_rows_v<Plant>;
  static constexpr size_t PlantCols = linalg::mat_cols_v<Plant>;
  static constexpr size_t ControlRows = linalg::mat_rows_v<Control>;
  static constexpr size_t ControlCols = linalg::mat_cols_v<Control>;
  static constexpr size_t OutputRows = linalg::mat_rows_v<Output>;
  static constexpr size_t OutputCols = linalg::mat_cols_v<Output>;
  static constexpr size_t FeedthroughRows = linalg::mat_rows_v<Feedthrough>;
  static constexpr size_t FeedthroughCols = linalg::mat_cols_v<Feedthrough>;

  static_assert(PlantRows == ControlRows,
                "state dim is not aligned between plant and control matrices");

  static_assert(
      OutputRows == FeedthroughRows,
      "output dim is not aligned between output and feedthrough matrices");

  /// @brief state vector
  using State = linalg::Vector<T, PlantCols>;
  using Input = linalg::Vector<T, ControlCols>;

  /**
   * @brief CTOR
   *
   * @param A (Plant)
   * @param B (Control)
   * @param C (Output)
   */
  constexpr explicit StateSpace(const Plant& A_in, const Control& B_in,
                                const Output& C_in)
      : A(A_in), B(B_in), C(C_in) {}

  /**
   * @brief CTOR
   *
   * @param A (Plant)
   * @param B (Control)
   * @param C (Output)
   * @param D (Feedthrough)
   */
  constexpr explicit StateSpace(const Plant& A_in, const Control& B_in,
                                const Output& C_in, const Feedthrough& D_in)
      : A(A_in), B(B_in), C(C_in), D(D_in) {}

  /** @brief member var */
  Plant A;                       ///< Plant matrix
  Control B;                     ///< Control matrix
  Output C;                      ///< Output matrix
  std::optional<Feedthrough> D;  ///< Feedthrough matrix
};

}  // namespace control
