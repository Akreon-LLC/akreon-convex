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

#include "config/assert.hpp"
#include "control/model/state_space.hpp"
#include "convex/convex_problem.hpp"
#include "convex/sets/lmi.hpp"
#include "convex/variable.hpp"
#include "linalg/operator/matrix_inverse.hpp"
#include "linalg/operator/matrix_pencil.hpp"

namespace control {

/**
 * @brief H2 State Feedback
 *
 * LDI governed by H2 norm
 *
 * Given the LTI System with additive disturbance
 *   xdot = A x + B_u u + B_w w
 *   y = C x + D_u u
 *
 * We seek a static feedback controller u = Kx s.t.
 *   xdot = (A + B_u K) x + B_w w... where (A + B_u K) is the cled loop Plant
 *   y = (C + D_u K) x... where (C + D_u K) is the closed loop performance
 * output
 *
 * The controllability gramian (Wc) requirement
 *   A_cl Wc + Wc A_cl^T + Bw Bw^T < 0
 *
 * The H2 norm squared (total impulse response energy)
 *   ||T_{wz}||_2^2 = Tr(C_cl Wc C_cl^T)
 *
 * We have a bilinear term in K Wc, so let Y = K Wc where K = Y Wc^-1
 *
 * H2 objective can be lifted via epigraph with Z
 *   Z >= (C Wc + D_u Y) Wc^{-1} (C Wc + Du Y)^T
 *
 * Convex Problem is now
 *   min  Tr(Z)
 *   s.t. A Wc + Wc A^T + B_u Y + Y^T B_u^T + B_w B_w^T < 0
 *        [ Z                C Wc + Du Y ] >= 0
 *        [ (C Wc + Du Y)^T  Wc          ]
 *  Dec Var = [ svec(Wc); vec(Y); svec(Z) ]
 */
template <ContinuousStateSpaceLike SS>
struct H2StateFeedback {
  /// @brief extracting alias and types
  using StateSpace = linalg::remove_cvref_t<SS>;
  using T = ss_type_t<StateSpace>;
  using State = state_vec_t<StateSpace>;
  using Input = input_vec_t<StateSpace>;
  using Output = output_vec_t<StateSpace>;

  static constexpr size_t StateDim = linalg::vec_size_v<State>;
  static constexpr size_t InputDim = linalg::vec_size_v<Input>;
  static constexpr size_t OutputDim = linalg::vec_size_v<Output>;
};

} // namespace control
