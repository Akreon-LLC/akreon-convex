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

#include "control/model/state_space.hpp"
#include "convex/variable.hpp"

namespace control {
/**
 * @brief LQR
 *
 * Performance of the system V(x,t) = min_u ∫ [x(t)^T Q x(t) + u(t)^T R u(t)] dt
 * Q \in S^n_+ penalizes state deviations
 * R \in S^n_+ penalizes control effort
 *
 * V(x) = x^T P x
 * 0 = min_u[ x^T Q x + u^T R u + \nabla V^T(A x + B u) ]
 * \nabla V = 2 P x
 * so we have... u^T R u + 2 x^T P B u
 * derivative w.r.t u and set to zero... u^* = -R^{-1} B^T P x
 *
 * Goal find u(t) that minimizes J via Ricatti EQ
 * u^*(t) = -K x(t)... K = R^{-1} B^T P
 * where K is the feedback gain controller
 *
 * sub back into HJB: 0 = x^T(Q + A^T P + P A - P B R^{-1} B^T P) x
 *
 * Therefore CARE: A^T P + P A - P B R^{-1} B^T P + Q = 0
 *
 * LMI form: [ -(A^T P + P A + Q) P B ]
 *           [ B^T P              R   ]
 */
template <ContinuousStateSpaceLike SS, linalg::SquareMatrixLike StateCostMat,
          linalg::SquareMatrixLike InputCostMat>
struct LQR {
  /// @brief extracting alias and types
  using T = ss_type_t<SS>;
  using State = state_vec_t<SS>;
  using Input = input_vec_t<SS>;

  static constexpr size_t StateDim = linalg::vec_size_v<State>;
  static constexpr size_t InputDim = linalg::vec_size_v<Input>;

  static_assert(linalg::mat_rows_v<StateCostMat> == StateDim,
                "Expected Q mat row/col dim != State dim from SS");

  static_assert(linalg::mat_rows_v<InputCostMat> == InputDim,
                "Expected R mat row/col dim != Input dim from SS");

  /**
   * @brief operator overload (form LQR problem)
   *
   * @param ss (StateSpace)
   * @param Q (State Cost)
   * @param R (Control Cost)
   * @param x (dec var child)
   *
   * @return composed atom
   */
  template <convex::AtomLike Var>
  constexpr auto operator()(const SS& ss, const StateCostMat& Q,
                            const InputCostMat& R, const Var& P) {
    static_assert(convex::is_var_type_v<Var>, "Input atom is not a var");
    static_assert(
        convex::dom_vartype_v<convex::meta_dom_t<convex::atom_meta_t<Var>>> ==
            convex::VarType::PSD,
        "Var type != PSD");

    (void)ss;
    (void)Q;
    (void)R;
    (void)P;
    return T(0);
  }
};

}  // namespace control
