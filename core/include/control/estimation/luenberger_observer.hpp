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
 * @brief Luenberger Observer
 *
 * Given
 *   xdot = A x + B u
 *   y    = C x
 *
 * design a state observer
 *   xhatdot = A xhat + B u + L (y - C xhat)
 *
 * such that the estimation error
 *   e = x - xhat
 * satisfies
 *   edot = (A - L C) e
 * and the error dynamics are asymptotically stable
 * (or satisfy the chosen performance certificate).
 *
 * There exists Q > 0 s.t.
 *   A^T Q + Q A + W C + C^T W^T < 0
 *
 * where W = Q L.
 *
 * we can elim W same as state feedback via introducing \sigma = 1
 *   A^T Q + Q A - \sigma C^T C < 0
 *
 * observer gain calculation
 *   L = (\sigma / 2) Q^{-1} C^T
 */
template <ContinuousPlantModelLike ObsModel>
struct LuenbergerObserver {
  /// @brief extracting alias and types
  using Model = linalg::remove_cvref_t<ObsModel>;

  using T = plant_type_t<Model>;
  using State = plant_state_vec_t<Model>;

  static constexpr size_t StateDim = plant_traits<Model>::StateDim;
  static constexpr size_t OutputDim = plant_traits<Model>::MeasurementDim;

  using QMat = linalg::Matrix<T, StateDim, StateDim>;
  using LMat = linalg::Matrix<T, StateDim, OutputDim>;

  static constexpr size_t QDim = StateDim * (StateDim + 1) / 2;
  using FMat = QMat;
  using Fs = linalg::Vector<FMat, QDim>;

  /**
   * @brief operator overload (form luenberger observer)
   *
   * @param model (StateSpace)
   * @param Q (Var)
   *
   * @return observer synthesis constraints
   */
  template <convex::AtomLike Var>
  constexpr auto operator()(const ObsModel& model, const Var& Q,
                            T margin = 1e-6) const {
    return apply(model, Q, margin);
  }

  /**
   * @brief apply
   *
   * @param model (StateSpace)
   * @param Q (Var)
   *
   * @return observer synthesis constraints
   */
  template <convex::AtomLike Var>
  constexpr auto apply(const ObsModel& model, const Var& Q,
                       T margin = 1e-6) const {
    static_assert(convex::is_var_type_v<Var>, "Input atom is not a var");
    using QDom = convex::meta_dom_t<convex::atom_meta_t<Var>>;
    static_assert(convex::dom_vartype_v<QDom> == convex::VarType::PSD,
                  "Var type != PSD");
    using QSpace = convex::dom_space_t<QDom>;
    static_assert(std::same_as<QSpace, linalg::SymmetricMatrix<QMat>>,
                  "Expected Q var size != Q mat size");

    /// (dakre) note sign change to make this psd
    auto M = [&](const QMat& E) -> FMat {
      const auto& A = model.dynamics_matrix();
      return T(-1) * (linalg::transpose(A) * E + E * A);
    };

    const auto C_y = model.measurement_matrix();
    FMat B = linalg::transpose(C_y) * C_y;

    linalg::MatrixPencil<FMat, QDim> pencil;
    const auto F_blks = pencil.template apply_sym_mat<QMat>(M);

    const auto observer_lmi = convex::LMI(F_blks, B, Q);

    const auto I = linalg::make_identity<T, StateDim>();
    const auto Fq =
        linalg::MatrixPencil<QMat, QDim>{}.template apply_sym_mat<QMat>(
            [](const QMat& E) -> QMat { return E; });
    const auto q_lmi = convex::LMI(Fq, T(-1) * margin * I, Q);

    return convex::Intersection(q_lmi, observer_lmi);
  }

  /**
   * @brief L
   *
   * L = 1/2 Q^{-1} C^T
   *
   * @param model (StateSpace)
   * @param Q (QMat)
   *
   * @return L
   */
  constexpr auto L(const ObsModel& model, const QMat& Q) const {
    const auto [Q_inv, event] = linalg::SymmetricMatrixInverse<QMat>()(Q);
    if (event != linalg::NumericalEvent::NONE) return std::tuple(LMat{}, event);
    const auto C_y = model.measurement_matrix();
    const auto L = T(0.5) * Q_inv * linalg::transpose(C_y);
    return std::tuple(L, linalg::NumericalEvent::NONE);
  }
};

/**
 * @brief make observer problem
 *
 * @param model (StateSpace)
 * @param Q (Var)
 * @param obj (Objective)
 * @param margin (PSD margin)
 *
 * @return observer problem
 */
template <ContinuousPlantModelLike ObsModel, convex::AtomLike Var>
constexpr auto make_observer_problem(const ObsModel& model, const Var& Q,
                                     const auto& obj,
                                     plant_type_t<ObsModel> margin = 1e-6) {
  const auto obs = LuenbergerObserver<decltype(model)>();
  const auto constraints = obs(model, Q, margin);

  const auto prob_constraints = convex::make_problem(Q).in_set(constraints);

  return prob_constraints.minimize(obj);
}

/**
 * @brief make observer problem
 *
 * @param model (StateSpace)
 * @param Q (Var)
 * @param margin (PSD margin)
 *
 * @return observer problem
 */
template <ContinuousPlantModelLike ObsModel, convex::AtomLike Var>
constexpr auto make_observer_problem(const ObsModel& model, const Var& Q,
                                     plant_type_t<ObsModel> margin = 1e-6) {
  const auto obs = LuenbergerObserver<decltype(model)>();
  const auto constraints = obs(model, Q, margin);

  const auto prob_constraints = convex::make_problem(Q).in_set(constraints);

  return prob_constraints.feasibility();
}

} // namespace control
