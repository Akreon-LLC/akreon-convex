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
#include "convex/sets/lmi.hpp"
#include "convex/variable.hpp"
#include "linalg/operator/matrix_inverse.hpp"

namespace control {

/**
 * @brief PolytopicStateFeedback
 *
 * Polytopic State Feedback (PLDI)
 * Given the LTV system
 *   xdot = A(t) x + B_u(t) u
 *
 * We seek a static state feedback controller
 *   u = K x
 *
 * s.t. the CL system is quadratically stable.
 *
 * There exists Q > 0, Q = P^{-1} so
 *   Q A_i^T + A_i Q + B_{ui} Y + Y^T B_{ui}^T < 0
 */
template <ContinuousStateSpaceLike SS, size_t N>
struct PolytopicStateFeedback {
  /// @brief extracting alias and types
  using StateSpace = linalg::remove_cvref_t<SS>;

  using T = ss_type_t<StateSpace>;
  using State = state_vec_t<StateSpace>;
  using Input = input_vec_t<StateSpace>;

  static constexpr size_t StateDim = linalg::vec_size_v<State>;
  static constexpr size_t InputDim = linalg::vec_size_v<Input>;

  using QMat = linalg::Matrix<T, StateDim, StateDim>;
  using KMat = linalg::Matrix<T, InputDim, StateDim>;
  using KTMat = linalg::Matrix<T, StateDim, InputDim>;

  static constexpr size_t QDim = StateDim * (StateDim + 1) / 2;
  using FMat = QMat;
  using Fs = linalg::Vector<FMat, QDim>;

  using Polytope = linalg::Vector<SS, N>;

  /**
   * @brief operator overload (polytope state feedback)
   *
   * @param polytope (Vector of StateSpace)
   * @param Q (var PSD)
   * @param Y (var)
   *
   * @return PLDI as LMI
   */
  template <convex::AtomLike Var>
  constexpr auto operator()(const Polytope& poly, const Var& Q,
                            const Var& Y) const {
    return apply(poly, Q, Y);
  }

  /**
   * @brief apply
   *
   * @param polytope (Vector of StateSpace)
   * @param Q (var)
   *
   * @return PLDI as LMI
   */
  template <convex::AtomLike Var>
  constexpr auto apply(const Polytope& poly, const Var& Q, const Var& Y) const {
    static_assert(convex::is_var_type_v<Var>, "Input atom is not a var");
    using QDom = convex::meta_dom_t<convex::atom_meta_t<Var>>;
    static_assert(convex::dom_vartype_v<QDom> == convex::VarType::PSD,
                  "Var type != PSD");
    using QSpace = convex::dom_space_t<QDom>;
    static_assert(std::same_as<QSpace, linalg::SymmetricMatrix<QMat>>,
                  "Expected Q var size != Q mat size");

    /// (dakre) note sign change to make this psd
    auto M = [&](const QMat& E) -> FMat {

    };
  }
};

}  // namespace control
