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
#include "control/control_util.hpp"
#include "control/model/plant_model.hpp"
#include "control/model/state_space.hpp"
#include "convex/convex_problem.hpp"
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
template <ContinuousObserverModelLike SS>
struct LuenbergerObserver {
  /// @brief extracting alias and types
  using Model = linalg::remove_cvref_t<SS>;

  using T = observer_type_t<Model>;
  using State = observer_state_vec_t<Model>;

  static constexpr size_t StateDim = observer_model_traits<Model>::StateDim;
  static constexpr size_t OutputDim = observer_model_traits<Model>::OutputDim;

  using PMat = linalg::Matrix<T, StateDim, StateDim>;
  using LMat = linalg::Matrix<T, StateDim, OutputDim>;

  static constexpr size_t PDim = StateDim * (StateDim + 1) / 2;
  using FMat = PMat;
  using Fs = linalg::Vector<FMat, PDim>;

  /**
   * @brief operator overload (form luenberger observer)
   *
   * @param ss (StateSpace)
   * @param P (Var)
   *
   * @return LDI as LMI
   */
  template <convex::AtomLike Var>
  constexpr auto operator()(const SS& ss, const Var& P) const {
    return apply(ss, P);
  }

  /**
   * @brief apply
   *
   * @param ss (StateSpace)
   * @param P (Var)
   *
   * @return LDI as LMI
   */
  template <convex::AtomLike Var>
  constexpr auto apply(const SS& ss, const Var& P) const {
    static_assert(convex::is_var_type_v<Var>, "Input atom is not a var");
    using PDom = convex::meta_dom_t<convex::atom_meta_t<Var>>;
    static_assert(convex::dom_vartype_v<PDom> == convex::VarType::PSD,
                  "Var type != PSD");
    using PSpace = convex::dom_space_t<PDom>;
    static_assert(std::same_as<PSpace, linalg::SymmetricMatrix<PMat>>,
                  "Expected P var size != P mat size");

    /// (dakre) note sign change to make this psd
    auto M = [&](const PMat& E) -> FMat {
      const auto& A = plant_A(ss);
      return T(-1) * (linalg::transpose(A) * E + E * A);
    };

    const auto C_y = observer_Cy(ss);
    FMat B = linalg::transpose(C_y) * C_y;

    linalg::MatrixPencil<FMat, PDim> pencil;
    const auto F_blks = pencil.template apply_sym_mat<PMat>(M);

    return convex::LMI(F_blks, B, P);
  }

  /**
   * @brief L
   *
   * L = 1/2 P^{-1} C^T
   *
   * @param ss (StateSpace)
   * @param P (PMat)
   *
   * @return L
   */
  constexpr auto L(const SS& ss, const PMat& P) const {
    const auto [P_inv, event] = linalg::SymmetricMatrixInverse<PMat>()(P);
    if (event != linalg::NumericalEvent::NONE) return std::tuple(LMat{}, event);
    const auto C_y = observer_Cy(ss);
    const auto L = T(0.5) * P_inv * linalg::transpose(C_y);
    return std::tuple(L, linalg::NumericalEvent::NONE);
  }
};

/**
 * @brief make observer problem
 *
 * @param ss (StateSpace)
 * @param P (Var)
 * @param obj (Objective)
 * @param margin (PSD margin)
 *
 * @return observer problem
 */
template <ContinuousObserverModelLike SS, convex::AtomLike Var>
constexpr auto make_observer_problem(const SS& ss, const Var& P,
                                     const auto& obj,
                                     observer_type_t<SS> margin = 1e-6) {
  using T = observer_type_t<SS>;
  constexpr size_t N = LuenbergerObserver<SS>::StateDim;
  const auto I = linalg::make_identity<T, N>();

  const auto P_geq =
      convex::Inequality<convex::InequalityType::GEQ, decltype(P), decltype(I)>(
          P, margin * I);
  const auto obs = LuenbergerObserver<decltype(ss)>();
  const auto lmi = obs(ss, P);

  const auto prob_constraints =
      convex::make_problem(P).add_constraint(P_geq).in_set(lmi);

  return prob_constraints.minimize(obj);
}

/**
 * @brief make observer problem
 *
 * @param ss (StateSpace)
 * @param P (Var)
 * @param margin (PSD margin)
 *
 * @return observer problem
 */
template <ContinuousObserverModelLike SS, convex::AtomLike Var>
constexpr auto make_observer_problem(const SS& ss, const Var& P,
                                     observer_type_t<SS> margin = 1e-6) {
  using T = observer_type_t<SS>;
  constexpr size_t N = LuenbergerObserver<SS>::StateDim;
  const auto I = linalg::make_identity<T, N>();

  const auto P_geq =
      convex::Inequality<convex::InequalityType::GEQ, decltype(P), decltype(I)>(
          P, margin * I);
  const auto obs = LuenbergerObserver<decltype(ss)>();
  const auto lmi = obs(ss, P);

  const auto prob_constraints =
      convex::make_problem(P).add_constraint(P_geq).in_set(lmi);

  return prob_constraints.feasibility();
}

} // namespace control
