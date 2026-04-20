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

#include <algorithm>

#include "control/control_traits.hpp"

namespace control {

/**
 * @brief InputSaturationBlock
 *
 * u_sat = clamp(u_cmd, lower, upper)
 */
template <StateSpaceLike SS>
struct InputSaturationBlock {
  /// @brief alias & type extraction
  using StateSpace = linalg::remove_cvref_t<SS>;
  using T = ss_type_t<StateSpace>;
  using Input = input_vec_t<StateSpace>;

  /**
   * @brief u (Saturated)
   *
   * @brief u (Commanded Input)
   *
   * @return u (Saturated)
   */
  constexpr auto u(const Input& u_cmd) const {
    Input u_sat{};
    for (size_t i = 0; i < linalg::vec_size_v<Input>; ++i)
      u_sat(i) = std::max(lower(i), std::min(upper(i), u_cmd(i)));
    return u_sat;
  }

  /** @brief member vars **/
  Input lower;  ///< lower bounds
  Input upper;  ///< upper bounds
};

/**
 * @brief input_saturation_block (helper)
 *
 * @param lower (lower bounds vector)
 * @param higher (higher bounds vector)
 *
 * @return InputSaturationBlock
 */
template <StateSpaceLike SS>
constexpr auto input_saturation_block(const input_vec_t<SS>& lower,
                                      const input_vec_t<SS>& upper) {
  return InputSaturationBlock<SS>{lower, upper};
}

}  // namespace control
