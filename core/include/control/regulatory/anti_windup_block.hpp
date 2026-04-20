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
 * @brief AntiWindupBlock
 *
 * z_corr = Kaw (u_sat - u_cmd)
 */
template <StateSpaceLike SS, size_t ErrorDim>
struct AntiWindupBlock {
  /// @brief alias & type extraction
  using StateSpace = linalg::remove_cvref_t<SS>;
  using T = ss_type_t<StateSpace>;
  using Input = input_vec_t<StateSpace>;
  using IntState = linalg::Vector<T, ErrorDim>;
  using KawMat = linalg::Matrix<T, ErrorDim, linalg::vec_size_v<Input>>;

  /**
   * @brief z correction (Integrator Corrector)
   *
   * @param u_cmd (Commanded Input)
   * @param u_sat (Saturated Input)
   *
   * @return Kaw (u_sat - u_cmd)
   */
  constexpr auto z_correction(const Input& u_cmd, const Input& u_sat) const {
    return Kaw * (u_sat - u_cmd);
  }

  /** @brief member vars **/
  KawMat Kaw;  ///< Anti-windup gain
};

/**
 * @brief anti_windup_block
 *
 * @param Kaw (Anti-windup gain)
 *
 * @return AntiWindupBlock
 */
template <StateSpaceLike SS, size_t ErrorDim>
constexpr auto anti_windup_block(
    const linalg::Matrix<ss_type_t<SS>, ErrorDim,
                         linalg::vec_size_v<input_vec_t<SS>>>& Kaw) {
  return AntiWindupBlock<SS, ErrorDim>{Kaw};
}

}  // namespace control
