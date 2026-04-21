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

#include "control/control_traits.hpp"

namespace control {

/**
 * @brief OutputIntegrator
 *
 * e = r - y
 * z_next = z + dt e
 *
 * u_i = -Ki z
 */
template <StateSpaceLike SS, size_t ErrorDim>
struct OutputIntegratorBlock {
  /// @brief alias & type extractions from SS
  using StateSpace = linalg::remove_cvref_t<SS>;
  using T = ss_type_t<StateSpace>;
  using Input = input_vec_t<StateSpace>;
  using Output = output_vec_t<StateSpace>;
  using IntState = linalg::Vector<T, ErrorDim>;
  using SelectMat = linalg::Matrix<T, ErrorDim, linalg::vec_size_v<Output>>;
  using KiMat = linalg::Matrix<T, linalg::vec_size_v<Input>, ErrorDim>;

  static_assert(ErrorDim <= linalg::vec_size_v<Output> && ErrorDim > 0,
                "Invalid ErrorDim set on the integrator block");

  /**
   * @brief z_delta
   *
   * z_delta = e = r - y
   *
   * @param r (Reference)
   * @param y (Output)
   *
   * @return z_delta
   */
  constexpr auto z_delta(const Output& r, const Output& y) const {
    return S * (r - y);
  }

  /**
   * @brief u_i
   *
   * u_i = -K_i z
   *
   * @param z (Integrator State)
   *
   * @return u_i
   */
  constexpr auto u_i(const IntState& z) const { return T(-1) * Ki * z; }

  /** @brief member vars **/
  SelectMat S;  ///< Output channels to integrate
  KiMat Ki;     ///< Integrator gain
};

/**
 * @brief output_integrator_block
 *
 * @param S (Channel Selector)
 * @param Ki (Integral Gain)
 *
 * @return OutputIntegratorBlock
 */
template <StateSpaceLike SS, size_t ErrorDim>
constexpr auto output_integrator(
    const linalg::Matrix<ss_type_t<SS>, ErrorDim,
                         linalg::vec_size_v<output_vec_t<SS>>>& S,
    const linalg::Matrix<ss_type_t<SS>, linalg::vec_size_v<input_vec_t<SS>>,
                         ErrorDim>& Ki) {
  return OutputIntegratorBlock<SS, ErrorDim>{S, Ki};
}

}  // namespace control
