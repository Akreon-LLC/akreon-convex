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
 * @brief StateFeedback
 *
 * Linear Differential Inclusion (LDI) State Feedback
 * Given the LTI system
 *   xdot = A x + B_u u
 *
 * We seek a static state-feedback
 *   u = K x
 *
 * such that the closed-loop system is quadratically stable.
 *
 * There exists P > 0 such that
 *   (A + B_u K)^T P + P (A + B_u K) < 0
 *
 * Equivalently, letting Q = P^{-1} > 0,
 *   Q (A + B_u K)^T + (A + B_u K) Q < 0
 *
 * This is bilinear in Q and K, so define
 *   Y = K Q
 * hence
 *   K = Y Q^{-1}
 *
 * The problem is represented over the merged decision variable
 *   z = [ svec(Q); vec(Y) ].
 */
template <ContinuousStateSpaceLike SS>
struct StateFeedback {
  /// @brief extracting alias and types
  using StateSpace = linalg::remove_cvref_t<SS>;

  using T = ss_type_t<StateSpace>;
  using State = state_vec_t<StateSpace>;
  using Input = input_vec_t<StateSpace>;

  static constexpr size_t StateDim = linalg::vec_size_v<State>;
  static constexpr size_t InputDim = linalg::vec_size_v<Input>;

  using QMat = linalg::Matrix<T, StateDim, StateDim>;
  using YMat = linalg::Matrix<T, InputDim, StateDim>;
  using KMat = linalg::Matrix<T, InputDim, StateDim>;
  using FMat = QMat;

  static constexpr size_t QDim = StateDim * (StateDim + 1) / 2;
  static constexpr size_t YDim = InputDim * StateDim;
  static constexpr size_t ZDim = QDim + YDim;

  using ZSpace = linalg::RowVector<T, ZDim>;
  using Fs = linalg::Vector<FMat, ZDim>;

  /**
   * @brief operator overload (form state feedback)
   *
   * @param ss (StateSpace)
   * @param Z (dec var [svec(Q); vec(Y)])
   *
   * @return LDI as LMI
   */
  template <convex::AtomLike ZVar>
  constexpr auto operator()(const SS& ss, const ZVar& Z) const {
    return apply(ss, Z);
  }

  /**
   * @brief apply
   *
   * @param ss (StateSpace)
   * @param Z (dec var [svec(Q); vec(Y)])
   *
   * @return LDI as LMI
   */
  template <convex::AtomLike ZVar>
  constexpr auto apply(const SS& ss, const ZVar& Z) const {
    static_assert(convex::is_var_type_v<ZVar>, "Input atom is not a var");
    using ZDom = convex::meta_dom_t<convex::atom_meta_t<ZVar>>;
    static_assert(convex::dom_vartype_v<ZDom> == convex::VarType::FREE,
                  "Merged decision var must be free");
    using ZVarSpace = convex::dom_space_t<ZDom>;
    static_assert(std::same_as<ZVarSpace, ZSpace>,
                  "Expected merged decision size [svec(Q); vec(Y)]");

    auto M = [&](const QMat& E) -> FMat {
      return T(-1) * (ss.A * E + E * linalg::transpose(ss.A));
    };

    auto M_y = [&](const YMat& E) -> FMat {
      return T(-1) *
             (ss.B * E + linalg::transpose(E) * linalg::transpose(ss.B));
    };

    const auto Fq_local =
        linalg::MatrixPencil<FMat, QDim>{}.template apply_sym_mat<QMat>(M);
    const auto Fy_local =
        linalg::MatrixPencil<FMat, YDim>{}.template apply_full<YMat>(M_y);

    Fs F_blks{};
    for (size_t i = 0; i < QDim; ++i) F_blks(i) = Fq_local(i);
    for (size_t i = 0; i < YDim; ++i) F_blks(QDim + i) = Fy_local(i);

    return convex::LMI(F_blks, FMat{}, Z);
  }

  /**
   * @brief K
   *
   * K = Y Q^{-1}
   *
   * @param Q (QMat)
   * @param Y (YMat)
   *
   * @return K
   */
  constexpr auto K(const QMat& Q, const YMat& Y) const {
    const auto [Q_inv, event] = linalg::SymmetricMatrixInverse<QMat>()(Q);
    if (event != linalg::NumericalEvent::NONE) return std::tuple(KMat{}, event);
    const auto K = Y * Q_inv;
    return std::tuple(K, linalg::NumericalEvent::NONE);
  }
};

template <ContinuousStateSpaceLike SS, convex::AtomLike ZVar>
constexpr auto make_state_feedback_q_constraint(const ZVar& Z,
                                                ss_type_t<SS> margin = 1e-6) {
  using T = ss_type_t<SS>;
  using SF = StateFeedback<SS>;
  using QMat = typename SF::QMat;
  using Fs = typename SF::Fs;

  const auto I = linalg::make_identity<T, SF::StateDim>();
  const auto Fq =
      linalg::MatrixPencil<QMat, SF::QDim>{}.template apply_sym_mat<QMat>(
          [](const QMat& E) -> QMat { return E; });

  Fs F_blks{};
  for (size_t i = 0; i < SF::QDim; ++i) F_blks(i) = Fq(i);
  for (size_t i = 0; i < SF::YDim; ++i) F_blks(SF::QDim + i) = QMat{};

  return convex::LMI(F_blks, T(-1) * margin * I, Z);
}

/**
 * @brief make state feedback problem
 *
 * @param ss (StateSpace)
 * @param Z (merged decision var [svec(Q); vec(Y)])
 * @param obj (objective)
 * @param margin (PSD margin)
 *
 * @return state feedback problem
 */
template <ContinuousStateSpaceLike SS, convex::AtomLike ZVar>
constexpr auto make_state_feedback_problem(const SS& ss, const ZVar& Z,
                                           const auto& obj,
                                           ss_type_t<SS> margin = 1e-6) {
  const auto Q_geq = make_state_feedback_q_constraint<SS>(Z, margin);
  const auto sf = StateFeedback<decltype(ss)>();
  const auto lmi = sf(ss, Z);
  const auto prob_constraints =
      convex::make_problem(Z).in_set(Q_geq).in_set(lmi);
  return prob_constraints.minimize(obj);
}

/**
 * @brief make state feedback feasibility problem
 *
 * @param ss (StateSpace)
 * @param Z (merged decision var [svec(Q); vec(Y)])
 * @param margin (PSD margin)
 *
 * @return state feedback feasibility problem
 */
template <ContinuousStateSpaceLike SS, convex::AtomLike ZVar>
constexpr auto make_state_feedback_problem(const SS& ss, const ZVar& Z,
                                           ss_type_t<SS> margin = 1e-6) {
  const auto Q_geq = make_state_feedback_q_constraint<SS>(Z, margin);
  const auto sf = StateFeedback<decltype(ss)>();
  const auto lmi = sf(ss, Z);
  return convex::make_problem(Z).in_set(Q_geq).in_set(lmi).feasibility();
}

} // namespace control
