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
 * There exists P > 0 s.t.
 *   A^T P + P A + W C + C^T W^T < 0
 *
 * where W = P L.
 *
 * we can elim W same as state feedback via introducing \sigma = 1
 *   A^T P + P A - \sigma C^T C < 0
 *
 * observer gain calculation
 *   L = (\sigma / 2) P^{-1} C^T
 */
template <ContinuousPlantModelLike ObsModel>
struct LuenbergerObserver {
  /// @brief extracting alias and types
  using Model = linalg::remove_cvref_t<ObsModel>;

  using T = plant_type_t<Model>;
  using State = plant_state_vec_t<Model>;

  static constexpr size_t StateDim = plant_traits<Model>::StateDim;
  static constexpr size_t OutputDim = plant_traits<Model>::MeasurementDim;

  using PMat = linalg::Matrix<T, StateDim, StateDim>;
  using LMat = linalg::Matrix<T, StateDim, OutputDim>;

  static constexpr size_t PDim = StateDim * (StateDim + 1) / 2;
  using FMat = PMat;
  using Fs = linalg::Vector<FMat, PDim>;

  /**
   * @brief operator overload (form luenberger observer)
   *
   * @param model (StateSpace)
   * @param P (Var)
   *
   * @return observer synthesis constraints
   */
  template <convex::AtomLike Var>
  constexpr auto operator()(const ObsModel& model, const Var& P,
                            T margin = 1e-6) const {
    return apply(model, P, margin);
  }

  /**
   * @brief apply
   *
   * @param model (StateSpace)
   * @param P (Var)
   *
   * @return observer synthesis constraints
   */
  template <convex::AtomLike Var>
  constexpr auto apply(const ObsModel& model, const Var& P,
                       T margin = 1e-6) const {
    static_assert(convex::is_var_type_v<Var>, "Input atom is not a var");
    using PDom = convex::meta_dom_t<convex::atom_meta_t<Var>>;
    static_assert(convex::dom_vartype_v<PDom> == convex::VarType::PSD,
                  "Var type != PSD");
    using PSpace = convex::dom_space_t<PDom>;
    static_assert(std::same_as<PSpace, linalg::SymmetricMatrix<PMat>>,
                  "Expected P var size != P mat size");

    /// (dakre) note sign change to make this psd
    auto M = [&](const PMat& E) -> FMat {
      const auto& A = model.dynamics_matrix();
      return T(-1) * (linalg::transpose(A) * E + E * A);
    };

    const auto C_y = model.measurement_matrix();
    FMat B = linalg::transpose(C_y) * C_y;

    linalg::MatrixPencil<FMat, PDim> pencil;
    const auto F_blks = pencil.template apply_sym_mat<PMat>(M);

    const auto observer_lmi = convex::LMI(F_blks, B, P);

    const auto I = linalg::make_identity<T, StateDim>();
    const auto Fp =
        linalg::MatrixPencil<PMat, PDim>{}.template apply_sym_mat<PMat>(
            [](const PMat& E) -> PMat { return E; });
    const auto p_lmi = convex::LMI(Fp, T(-1) * margin * I, P);

    return convex::Intersection(p_lmi, observer_lmi);
  }

  /**
   * @brief L
   *
   * L = 1/2 P^{-1} C^T
   *
   * @param model (StateSpace)
   * @param P (PMat)
   *
   * @return L
   */
  constexpr auto L(const ObsModel& model, const PMat& P) const {
    const auto [P_inv, event] = linalg::SymmetricMatrixInverse<PMat>()(P);
    if (event != linalg::NumericalEvent::NONE) return std::tuple(LMat{}, event);
    const auto C_y = model.measurement_matrix();
    const auto L = T(0.5) * P_inv * linalg::transpose(C_y);
    return std::tuple(L, linalg::NumericalEvent::NONE);
  }
};

/**
 * @brief make observer problem
 *
 * @param model (StateSpace)
 * @param P (Var)
 * @param obj (Objective)
 * @param margin (PSD margin)
 *
 * @return observer problem
 */
template <ContinuousPlantModelLike ObsModel, convex::AtomLike Var>
constexpr auto make_observer_problem(const ObsModel& model, const Var& P,
                                     const auto& obj,
                                     plant_type_t<ObsModel> margin = 1e-6) {
  const auto obs = LuenbergerObserver<decltype(model)>();
  const auto constraints = obs(model, P, margin);

  const auto prob_constraints = convex::make_problem(P).in_set(constraints);

  return prob_constraints.minimize(obj);
}

/**
 * @brief make observer problem
 *
 * @param model (StateSpace)
 * @param P (Var)
 * @param margin (PSD margin)
 *
 * @return observer problem
 */
template <ContinuousPlantModelLike ObsModel, convex::AtomLike Var>
constexpr auto make_observer_problem(const ObsModel& model, const Var& P,
                                     plant_type_t<ObsModel> margin = 1e-6) {
  const auto obs = LuenbergerObserver<decltype(model)>();
  const auto constraints = obs(model, P, margin);

  const auto prob_constraints = convex::make_problem(P).in_set(constraints);

  return prob_constraints.feasibility();
}

} // namespace control
