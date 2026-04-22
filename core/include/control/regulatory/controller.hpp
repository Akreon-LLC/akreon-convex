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
 * Runtime block for a continuous-time controller, discretized online with a
 * forward-Euler step under a zero-order-hold input assumption.
 *
 * \dot{x} = Ac x(t) + Bc u(t)
 * y = Cc x(t) + Dc u(t)
 */
template <ContinuousStateSpaceLike SS>
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
   * @param dt (Discretization dt)
   */
  constexpr explicit Controller(const SS& ss_in, const T& dt_in)
      : ss(ss_in), dt(dt_in) {}

  /**
   * @brief step
   *
   * TODO - make the discretization capable w/ non-zoh like foh + turin later on
   *
   * @param x (State)
   * @param u (Input)
   *
   * @return (x_next, y)
   */
  constexpr auto step(const State& x, const Input& u) const {
    auto x_next = ss.plant_dynamics(x, u);
    x_next = x + dt * x_next;
    const auto y = ss.plant_output(x, u);
    return std::tuple(x_next, y);
  }

  /** @brief member vars **/
  SS ss; ///< Dynamic controller as SS
  T dt;  ///< Time step
};

} // namespace control
