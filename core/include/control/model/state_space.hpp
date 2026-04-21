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
 * x_next = x + dt (A x + B u)
 * y = C x + D u
 */
template <linalg::SquareMatrixLike PlantIn, linalg::MatrixLike ControlIn,
          linalg::MatrixLike OutputIn>
struct StateSpace {
  /// @brief extract type and alias
  using Plant = linalg::remove_cvref_t<PlantIn>;
  using Control = linalg::remove_cvref_t<ControlIn>;
  using Output = linalg::remove_cvref_t<OutputIn>;

  using T = linalg::mat_type_t<Plant>;
  static constexpr size_t PlantRows = linalg::mat_rows_v<Plant>;
  static constexpr size_t PlantCols = linalg::mat_cols_v<Plant>;
  static constexpr size_t ControlRows = linalg::mat_rows_v<Control>;
  static constexpr size_t ControlCols = linalg::mat_cols_v<Control>;
  static constexpr size_t OutputRows = linalg::mat_rows_v<Output>;
  static constexpr size_t OutputCols = linalg::mat_cols_v<Output>;

  using Feedthrough = linalg::Matrix<T, OutputRows, ControlCols>;

  static_assert(PlantRows == ControlRows,
                "state dim is not aligned between plant and control matrices");

  /// @brief state vector
  using State = linalg::Vector<T, PlantCols>;
  using Input = linalg::Vector<T, ControlCols>;
  using OutVec = linalg::Vector<T, OutputRows>;

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

  /**
   * @brief x_next
   *
   * Euler-discretized propagation with sample time dt.
   *
   * @param x (State)
   * @param u (Input)
   * @param dt (time step)
   *
   * @return xdot
   */
  constexpr auto x_next(const State& x, const Input& u, const T& dt) const {
    return x + dt * (A * x + B * u);
  }

  /**
   * @brief y
   *
   * y = C x + D u
   *
   * @param x (State)
   * @param u (Input)
   *
   * @return y
   */
  constexpr auto y(const State& x, const Input& u) const {
    return D.has_value() ? C * x + (*D) * u : C * x;
  }

  /** @brief member var */
  Plant A;                       ///< Plant matrix
  Control B;                     ///< Control matrix
  Output C;                      ///< Output matrix
  std::optional<Feedthrough> D;  ///< Feedthrough matrix
};

}  // namespace control
