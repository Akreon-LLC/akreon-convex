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

#include "control/model/state_space.hpp"
#include "convex/composed/weighted_sum.hpp"
#include "convex/convex_problem.hpp"
#include "convex/sets/lmi.hpp"
#include "convex/variable.hpp"
#include "linalg/operator/matrix_inverse.hpp"
#include "linalg/operator/matrix_pencil.hpp"
#include "math.hpp"

namespace control {
/**
 * @brief LyapunovCertificate
 *
 * Performance of the system V(x,t) = min_u ∫ [x(t)^T Q x(t) + u(t)^T R u(t)] dt
 * Q \in S^n_+ penalizes state deviations
 * R \in S^n_++ penalizes control effort
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
 * LMI (convex relaxation) -> A^T P + P A - P B R^{-1} B^T P + Q \preceq 0
 * along with P \succeq 0
 * 0 \preceq -(A^T P + P A + Q) + P B R^{-1} B^T P
 *
 * Target Schur-complement LMI:
 * [ -(A^T P + P A + Q)  -P B ] \succeq 0 & P \succeq 0
 * [ -B^T P               R   ]
 */
template <ContinuousStateSpaceLike SS, linalg::SquareMatrixLike StateCostMat,
          linalg::SquareMatrixLike InputCostMat>
struct LyapunovCertificate {
  /// @brief extracting alias and types
  using StateSpace = linalg::remove_cvref_t<SS>;
  using StateCost = linalg::remove_cvref_t<StateCostMat>;
  using InputCost = linalg::remove_cvref_t<InputCostMat>;

  using T = ss_type_t<StateSpace>;
  using State = state_vec_t<StateSpace>;
  using Input = input_vec_t<StateSpace>;

  static constexpr size_t StateDim = linalg::vec_size_v<State>;
  static constexpr size_t InputDim = linalg::vec_size_v<Input>;

  using PMat = linalg::Matrix<T, StateDim, StateDim>;
  using KMat = linalg::Matrix<T, InputDim, StateDim>;

  static_assert(linalg::mat_rows_v<StateCost> == StateDim,
                "Expected Q mat row/col dim != State dim from SS");

  static_assert(linalg::mat_rows_v<InputCost> == InputDim,
                "Expected R mat row/col dim != Input dim from SS");

  static constexpr size_t PDim = StateDim * (StateDim + 1) / 2;
  static constexpr size_t TotalDim = StateDim + InputDim;
  using FMat = linalg::Matrix<T, TotalDim, TotalDim>;
  using Fs = linalg::Vector<FMat, PDim>;

  /**
   * @brief operator overload (form Lyapunov certificate)
   *
   * @param ss (StateSpace)
   * @param Q (State Cost)
   * @param R (Control Cost)
   * @param P (dec var child)
   *
   * @return Lyapunov certificate as LMI
   */
  template <convex::AtomLike Var>
  constexpr auto operator()(const SS& ss, const StateCostMat& Q,
                            const InputCostMat& R, const Var& P) const {
    return apply(ss, Q, R, P);
  }

  /**
   * @brief apply (form Lyapunov certificate)
   *
   * @param ss (StateSpace)
   * @param Q (State Cost)
   * @param R (Control Cost)
   * @param P (dec var child)
   *
   * @return Lyapunov certificate as LMI
   */
  template <convex::AtomLike Var>
  constexpr auto apply(const SS& ss, const StateCostMat& Q,
                       const InputCostMat& R, const Var& P) const {
    static_assert(convex::is_var_type_v<Var>, "Input atom is not a var");
    using PDom = convex::meta_dom_t<convex::atom_meta_t<Var>>;
    static_assert(convex::dom_vartype_v<PDom> == convex::VarType::PSD,
                  "Var type != PSD");
    using PSpace = convex::dom_space_t<PDom>;
    static_assert(std::same_as<PSpace, linalg::SymmetricMatrix<PMat>>,
                  "Expected P var size != P mat size");

    auto M = [&](const PMat& E) -> FMat {
      const auto lyap = T(-1) * (linalg::transpose(ss.A) * E + E * ss.A);
      const auto pb = T(-1) * E * ss.B;
      const auto pTb = T(-1) * linalg::transpose(ss.B) * E;

      FMat F{};
      F = linalg::set_block<0, StateDim, 0, StateDim>(F, lyap);
      F = linalg::set_block<0, StateDim, StateDim, TotalDim>(F, pb);
      F = linalg::set_block<StateDim, TotalDim, 0, StateDim>(F, pTb);

      return F;
    };

    FMat B{};
    B = linalg::set_block<0, StateDim, 0, StateDim>(B, T(-1) * Q);
    B = linalg::set_block<StateDim, TotalDim, StateDim, TotalDim>(B, R);

    linalg::MatrixPencil<FMat, PDim> pencil;
    const auto F_blks = pencil.template apply_sym_mat<PMat>(M);

    return convex::LMI(F_blks, B, P);
  }

  /**
   * @brief K (optimal controller)
   *
   * K = R^{-1} B^T P
   *
   * @param P (lyapnuov P)
   * @param ss (StateSpace)
   * @param R (Input Cost)
   *
   * @return K
   */
  constexpr auto K(const PMat& P, const SS& ss, const InputCostMat& R) const {
    const auto [R_inv, event] = linalg::SymmetricMatrixInverse<InputCost>()(R);
    if (event != linalg::NumericalEvent::NONE) return std::tuple(KMat{}, event);
    const auto K = R_inv * linalg::transpose(ss.B) * P;
    return std::tuple(K, linalg::NumericalEvent::NONE);
  }
};

/**
 * @brief make lqr problem
 *
 * @param ss (StateSpace)
 * @param Q (State Cost)
 * @param R (Control Cost)
 * @param P (var \in PSD)
 * @param margin (tol to make P PD)
 *
 * @return lqr problem
 */
template <ContinuousStateSpaceLike SS, linalg::SquareMatrixLike StateCostMat,
          linalg::SquareMatrixLike InputCostMat, convex::AtomLike Var>
constexpr auto make_lqr_problem(const SS& ss, const StateCostMat& Q,
                                const InputCostMat& R, const Var& P,
                                ss_type_t<SS> margin = 1e-6) {
  using T = ss_type_t<SS>;
  using PDom = convex::meta_dom_t<convex::atom_meta_t<Var>>;
  using PSpace = convex::dom_space_t<PDom>;
  constexpr size_t N = linalg::mat_rows_v<ss_plant_t<SS>>;
  constexpr size_t K = N * (N + 1) / 2;

  const auto lyap =
      LyapunovCertificate<decltype(ss), decltype(Q), decltype(R)>();
  const auto lmi = lyap(ss, Q, R, P);

  const auto I = linalg::make_identity<T, N>();

  const auto P_geq =
      convex::Inequality<convex::InequalityType::GEQ, decltype(P), decltype(I)>(
          P, margin * I);
  const auto prob_constraints =
      convex::make_problem(P).add_constraint(P_geq).in_set(lmi);

  linalg::Vector<T, K> c{};
  for (size_t i = 0, idx = 0; i < N; ++i)
    for (size_t j = 0; j < i + 1; ++j, ++idx)
      if (i == j) c(idx) = T(1);

  const auto obj = convex::WeightedSum<PSpace>()(c, P);
  return prob_constraints.minimize(obj);
}

} // namespace control
