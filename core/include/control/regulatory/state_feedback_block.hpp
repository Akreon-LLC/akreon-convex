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
 * @brief StateFeedbackBlock
 *
 * u = -K x
 */
template <StateSpaceLike SS>
struct StateFeedbackBlock {
  /// @brief alias & type extractions from SS
  using StateSpace = linalg::remove_cvref_t<SS>;
  using T = ss_type_t<StateSpace>;
  using State = state_vec_t<StateSpace>;
  using Input = input_vec_t<StateSpace>;
  using KMat =
      linalg::Matrix<T, linalg::vec_size_v<Input>, linalg::vec_size_v<State>>;

  /**
   * @brief u_fb
   *
   * u = - K x
   *
   * @param x (State)
   *
   * return u
   */
  constexpr auto u_fb(const State& x) const { return T(-1) * K * x; }

  /** @brief member vars **/
  KMat K;  ///< State feedback gain
};

/**
 * @brief make_block
 *
 * @param K (state feedback gain)
 *
 * @return StateFeedbackBlock
 */
template <StateSpaceLike SS>
constexpr auto state_feedback_block(
    const linalg::Matrix<ss_type_t<SS>, linalg::vec_size_v<input_vec_t<SS>>,
                         linalg::vec_size_v<state_vec_t<SS>>>& K) {
  return StateFeedbackBlock<SS>{K};
}

}  // namespace control
