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
 * Runtime block for a discrete-time controller/compensator.
 *
 * z_{k+1} = A_c z_k + B_c eta_k
 * u_k     = C_c z_k + D_c eta_k
 *
 * where:
 *   z     = controller internal state
 *   eta   = external controller input vector, e.g. [r; y_meas]
 *   u_k   = controller command output
 *
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
   * @param z (Controller State)... [xhat; x]
   * @param eta (Exogeneous Input)... [r; y_meas]
   *
   * @return (z_next, u_cmd)
   */
  constexpr auto step(const State& z, const Input& eta) const {
    const auto z_next = ss.state_update(z, eta);
    const auto u_cmd = ss.output_map(z, eta);
    return std::tuple(z_next, u_cmd);
  }

  /** @brief member vars **/
  SS ss; ///< Dynamic controller as SS
  T dt;  ///< Time step
};

} // namespace control
