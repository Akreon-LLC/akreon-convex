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
#include "convex/composed/weighted_sum.hpp"
#include "convex/convex_problem.hpp"
#include "convex/sets/lmi.hpp"
#include "convex/variable.hpp"
#include "linalg/operator/matrix_inverse.hpp"
#include "linalg/operator/matrix_pencil.hpp"

namespace control {

/**
 * @brief KalmanFilter
 *
 * Very similar to LQR with Q = process uncertainty (StateCostMat) & R = sensor
 * noise (OutputCostMat)
 *
 * LTI system with gaussian noise is as follows:
 *   xdot = A x + B u + G w
 *   y = Cx + v
 *
 * w = process noise, and v = measurement noise
 *
 * covariance weights:
 *   E[w w^T] = Q
 *   E[v v^T] = R
 *
 * steady state kalman gain is:
 *   L = P C^T R^{-1}
 *
 * standard ricatti eq:
 *   A P + P A^T + G Q G^T - P C^T R^{-1} C P < 0
 *
 * assume G = I to begin (TODO perhaps include this in modeling later)
 *
 * larger Q > R... trust measurements more and converse is also true,
 * analogous to LQR w/ large state penalty means regulate state harder and
 * conversely large control effort means to use less control effort
 */
template <ContinuousStateSpaceLike SS, linalg::SquareMatrixLike StateCostMat,
          linalg::SquareMatrixLike OutputCostMat>
struct KalmanFilter {
  /// @brief extracting alias and types
  using StateSpace = linalg::remove_cvref_t<SS>;
  using StateCost = linalg::remove_cvref_t<StateCostMat>;
  using OutputCost = linalg::remove_cvref_t<OutputCostMat>;

  using T = ss_type_t<StateSpace>;
  using State = state_vec_t<StateSpace>;
  using OutputMap = ss_output_t<StateSpace>;

  static constexpr size_t StateDim = linalg::vec_size_v<State>;
  static constexpr size_t OutputDim = linalg::mat_rows_v<OutputMap>;

  using PMat = linalg::Matrix<T, StateDim, StateDim>;
  using LMat = linalg::Matrix<T, StateDim, OutputDim>;

  static_assert(linalg::mat_rows_v<StateCost> == StateDim,
                "Expected Q mat row/col dim != State dim from SS");

  static_assert(linalg::mat_rows_v<OutputCost> == OutputDim,
                "Expected R mat row/col dim != Output dim from SS");

  static constexpr size_t PDim = StateDim * (StateDim + 1) / 2;
  static constexpr size_t TotalDim = StateDim + OutputDim;
  using FMat = linalg::Matrix<T, TotalDim, TotalDim>;
  using Fs = linalg::Vector<FMat, PDim>;

  /**
   * @brief operator overload (form kalman filter)
   *
   * @param ss (StateSpace)
   * @param Q (State Cost)
   * @param R (Output Cost)
   * @param P (dec var child)
   *
   * @return KalmanFilter as LMI
   */
  template <convex::AtomLike Var>
  constexpr auto operator()(const SS& ss, const StateCostMat& Q,
                            const OutputCostMat& R, const Var& P) const {
    return apply(ss, Q, R, P);
  }

  /**
   * @brief operator overload (form kalman filter)
   *
   * @param ss (StateSpace)
   * @param Q (State Cost)
   * @param R (Output Cost)
   * @param P (dec var child)
   *
   * @return KalmanFilter as LMI
   */
  template <convex::AtomLike Var>
  constexpr auto apply(const SS& ss, const StateCostMat& Q,
                       const OutputCostMat& R, const Var& P) const {
    static_assert(convex::is_var_type_v<Var>, "Input atom is not a var");
    using PDom = convex::meta_dom_t<convex::atom_meta_t<Var>>;
    static_assert(convex::dom_vartype_v<PDom> == convex::VarType::PSD,
                  "Var type != PSD");
    using PSpace = convex::dom_space_t<PDom>;
    static_assert(std::same_as<PSpace, linalg::SymmetricMatrix<PMat>>,
                  "Expected P var size != P mat size");

    auto M = [&](const PMat& E) -> FMat {
      const auto lyap = T(-1) * (ss.A * E + E * linalg::transpose(ss.A));
      const auto pc = T(-1) * ss.C * E;
      const auto pcT = T(-1) * E * linalg::transpose(ss.C);

      FMat F{};
      F = linalg::set_block<0, StateDim, 0, StateDim>(F, lyap);
      F = linalg::set_block<0, StateDim, StateDim, TotalDim>(F, pcT);
      F = linalg::set_block<StateDim, TotalDim, 0, StateDim>(F, pc);

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
   * @brief L (steady-state estimator gain)
   *
   * L = P C^T R^{-1}
   *
   * @param P (error covariance)
   * @param ss (StateSpace)
   * @param R (Output Cost)
   *
   * @return L
   */
  constexpr auto L(const PMat& P, const SS& ss, const OutputCostMat& R) const {
    const auto [R_inv, event] = linalg::SymmetricMatrixInverse<OutputCost>()(R);
    if (event != linalg::NumericalEvent::NONE) return std::tuple(LMat{}, event);
    const auto L = P * linalg::transpose(ss.C) * R_inv;
    return std::tuple(L, linalg::NumericalEvent::NONE);
  }
};

/**
 * @brief make lqe problem
 *
 * @param ss (StateSpace)
 * @param Q (State Cost)
 * @param R (Output Cost)
 * @param P (var \in PSD)
 * @param margin (tol to make P PD)
 *
 * @return lqe problem
 */
template <ContinuousStateSpaceLike SS, linalg::SquareMatrixLike StateCostMat,
          linalg::SquareMatrixLike OutputCostMat, convex::AtomLike Var>
constexpr auto make_lqe_problem(const SS& ss, const StateCostMat& Q,
                                const OutputCostMat& R, const Var& P,
                                ss_type_t<SS> margin = 1e-6) {
  using T = ss_type_t<SS>;
  using PDom = convex::meta_dom_t<convex::atom_meta_t<Var>>;
  using PSpace = convex::dom_space_t<PDom>;
  constexpr size_t N = linalg::mat_rows_v<ss_plant_t<SS>>;
  constexpr size_t K = N * (N + 1) / 2;

  const auto kf = KalmanFilter<decltype(ss), decltype(Q), decltype(R)>();
  const auto lmi = kf(ss, Q, R, P);

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
