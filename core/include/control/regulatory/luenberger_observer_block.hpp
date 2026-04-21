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
 * @brief LuenbergerObserverBlock
 *
 * yhat = C x + D u
 * e = y - yhat
 * xhat_next = xhat + dt (A xhat + B u + L e)
 */
template <StateSpaceLike SS>
struct LuenbergerObserverBlock {
  /// @brief alias & type extractions from SS
  using StateSpace = linalg::remove_cvref_t<SS>;
  using T = ss_type_t<StateSpace>;
  using State = state_vec_t<StateSpace>;
  using Input = input_vec_t<StateSpace>;
  using Output = output_vec_t<StateSpace>;
  using LMat =
      linalg::Matrix<T, linalg::vec_size_v<State>, linalg::vec_size_v<Output>>;

  /**
   * @brief xhat_delta
   *
   * @param ss (StateSpace)
   * @param xhat (Estimated State)
   * @param u (Input)
   * @param y (Output)
   *
   * @return xhat_delta
   */
  constexpr auto xhat_delta(const StateSpace& ss, const State& xhat,
                            const Input& u, const Output& y) const {
    const auto yhat =
        ss.D.has_value() ? ss.C * xhat + (*ss.D) * u : ss.C * xhat;
    return ss.A * xhat + ss.B * u + L * (y - yhat);
  }

  /** @brief member vars **/
  LMat L;  ///< Observer gain
};

/**
 * @brief luenberger_observer_block
 *
 * @param L (Observer Gain)
 *
 * @return LuenbergerObserverBlock
 */
template <StateSpaceLike SS>
constexpr auto luenberger_observer_block(
    const linalg::Matrix<ss_type_t<SS>, linalg::vec_size_v<state_vec_t<SS>>,
                         linalg::vec_size_v<output_vec_t<SS>>>& L) {
  return LuenbergerObserverBlock<SS>{L};
}

}  // namespace control
