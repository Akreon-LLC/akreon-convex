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
 * @brief Controller
 *
 * Runtime block for a discrete-time controller.
 *
 * The controller assumes the provided state-space model has already been
 * discretized at the intended sample period. In other words, the time-step
 * dependence is already encoded in the discrete state-space matrices, so
 * stepping the controller applies the discrete update directly rather than
 * performing any runtime integration or online discretization.
 *
 * x_next = Ad x_k + Bd u_k
 * y = Cd x_k + Dd u_k
 */
template <DiscreteStateSpaceLike SS>
struct Controller {
  /// @brief extract type and alias
  using StateSpace = linalg::remove_cvref_t<SS>;
  using State = state_vec_t<StateSpace>;
  using Input = input_vec_t<StateSpace>;
  using T = ss_type_t<StateSpace>;

  /**
   * @brief CTOR
   *
   * @param ss_in (StateSpace)
   * @param dt_in (Sample period metadata associated with the discrete model)
   */
  constexpr explicit Controller(const SS& ss_in, const T& dt_in)
      : ss(ss_in), dt(dt_in) {}

  /**
   * @brief step
   *
   * @param x (State)
   * @param u (Input)
   *
   * @return (x_next, y)
   */
  constexpr auto step(const State& x, const Input& u) const {
    const auto x_next = ss.plant_dynamics(x, u);
    const auto y = ss.plant_output(x, u);
    return std::tuple(x_next, y);
  }

  /** @brief member vars **/
  SS ss; ///< Dynamic controller as SS
  T dt;  ///< Time step
};

} // namespace control
