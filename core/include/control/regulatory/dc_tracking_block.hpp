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
 * @brief DCTrackingBlock
 */
template <StateSpaceLike SS>
struct DCTrackingBlock {
  /// @brief alias & type extractions from SS
  using StateSpace = linalg::remove_cvref_t<SS>;
  using T = ss_type_t<StateSpace>;
  using Input = input_vec_t<StateSpace>;
  using Output = output_vec_t<StateSpace>;
  using KrMat =
      linalg::Matrix<T, linalg::vec_size_v<Input>, linalg::vec_size_v<Output>>;

  /**
   * @brief u_ff
   *
   * u_ff = Kr r
   *
   * @param r (Output)
   *
   * @return u_ff
   */
  constexpr auto u_ff(const Output& r) const { return Kr * r; }

  /** @brief member vars **/
  KrMat Kr;
};

/**
 * @brief dc_tracking_block
 *
 * @param Kr (DC Tracking Gain)
 *
 * @return DCTrackingBlcok
 */
template <StateSpaceLike SS>
constexpr auto dc_tracking_block(
    const linalg::Matrix<ss_type_t<SS>, linalg::vec_size_v<input_vec_t<SS>>,
                         linalg::vec_size_v<output_vec_t<SS>>>& Kr) {
  return DCTrackingBlock<SS>{Kr};
}

}  // namespace control
