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
#include "control/feedback/gain_recovery.hpp"
#include "control/model/plant_model.hpp"
#include "control/model/state_space.hpp"
#include "convex/convex_problem.hpp"
#include "convex/intersection.hpp"
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
template <ContinuousPlantModelLike SFModel>
struct StateFeedback {
  /// @brief extracting alias and types
  using Model = linalg::remove_cvref_t<SFModel>;

  using T = plant_type_t<Model>;
  using State = plant_state_vec_t<Model>;

  static constexpr size_t StateDim = plant_traits<Model>::StateDim;
  static constexpr size_t InputDim = plant_traits<Model>::ControlDim;

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
   * @param model (StateSpace)
   * @param Z (dec var [svec(Q); vec(Y)])
   *
   * @return state-feedback synthesis constraints
   */
  template <convex::AtomLike ZVar>
  constexpr auto operator()(const SFModel& model, const ZVar& Z,
                            T margin = 1e-6) const {
    return apply(model, Z, margin);
  }

  /**
   * @brief apply
   *
   * @param model (StateSpace)
   * @param Z (dec var [svec(Q); vec(Y)])
   * @param margin (for PD)
   *
   * @return state-feedback synthesis constraints
   */
  template <convex::AtomLike ZVar>
  constexpr auto apply(const SFModel& model, const ZVar& Z,
                       T margin = 1e-6) const {
    static_assert(convex::is_var_type_v<ZVar>, "Input atom is not a var");
    using ZDom = convex::meta_dom_t<convex::atom_meta_t<ZVar>>;
    static_assert(convex::dom_vartype_v<ZDom> == convex::VarType::FREE,
                  "Merged decision var must be free");
    using ZVarSpace = convex::dom_space_t<ZDom>;
    static_assert(std::same_as<ZVarSpace, ZSpace>,
                  "Expected merged decision size [svec(Q); vec(Y)]");

    const auto stability_lmi = stability_constraint(model, Z);
    const auto q_lmi = q_constraint(Z, margin);
    return convex::Intersection(q_lmi, stability_lmi);
  }

  /**
   * @brief q_constraint
   *
   * @param Z (dec var)
   * @param margin (PD)
   *
   * @return Q constraint as LMI
   */
  template <convex::AtomLike ZVar>
  constexpr auto q_constraint(const ZVar& Z, T margin = 1e-6) const {
    const auto I = linalg::make_identity<T, StateDim>();
    const auto Fq_psd =
        linalg::MatrixPencil<QMat, QDim>{}.template apply_sym_mat<QMat>(
            [](const QMat& E) -> QMat { return E; });

    Fs Q_blks{};
    for (size_t i = 0; i < QDim; ++i) Q_blks(i) = Fq_psd(i);
    for (size_t i = 0; i < YDim; ++i) Q_blks(QDim + i) = QMat{};

    return convex::LMI(Q_blks, T(-1) * margin * I, Z);
  }

  /**
   * @brief stability_constraint
   *
   * @param model (PlantModel)
   * @param Z (Dec Var)
   *
   * @return stability constraint as LMI
   */
  template <convex::AtomLike ZVar>
  constexpr auto stability_constraint(const SFModel& model,
                                      const ZVar& Z) const {
    auto M = [&](const QMat& E) -> FMat {
      const auto A = model.dynamics_matrix();
      return T(-1) * (A * E + E * linalg::transpose(A));
    };

    auto M_y = [&](const YMat& E) -> FMat {
      const auto B_u = model.control_matrix();
      return T(-1) * (B_u * E + linalg::transpose(E) * linalg::transpose(B_u));
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
    return recover_state_feedback_gain(Q, Y);
  }
};

/**
 * @brief make state feedback problem
 *
 * @param model (StateSpace)
 * @param Z (merged decision var [svec(Q); vec(Y)])
 * @param obj (objective)
 * @param margin (PSD margin)
 *
 * @return state feedback problem
 */
template <ContinuousPlantModelLike SFModel, convex::AtomLike ZVar>
constexpr auto make_state_feedback_problem(
    const SFModel& model, const ZVar& Z, const auto& obj,
    plant_type_t<SFModel> margin = 1e-6) {
  const auto sf = StateFeedback<decltype(model)>();
  const auto constraints = sf(model, Z, margin);
  const auto prob_constraints = convex::make_problem(Z).in_set(constraints);
  return prob_constraints.minimize(obj);
}

/**
 * @brief make state feedback feasibility problem
 *
 * @param model (StateSpace)
 * @param Z (merged decision var [svec(Q); vec(Y)])
 * @param margin (PSD margin)
 *
 * @return state feedback feasibility problem
 */
template <ContinuousPlantModelLike SFModel, convex::AtomLike ZVar>
constexpr auto make_state_feedback_problem(
    const SFModel& model, const ZVar& Z, plant_type_t<SFModel> margin = 1e-6) {
  const auto sf = StateFeedback<decltype(model)>();
  const auto constraints = sf(model, Z, margin);
  return convex::make_problem(Z).in_set(constraints).feasibility();
}

} // namespace control
