/*
 * Copyright © 2026 Akreon LLC. All rights reserved.
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

#include "control/feedback/hinf_mixed_sensitivity.hpp"
#include "control/feedback/hinf_polytopic_state_feedback.hpp"
#include "convex/convex_problem.hpp"
#include "convex/variable.hpp"

namespace control {

/**
 * @brief Polytopic Hinf mixed-sensitivity state-feedback synthesis.
 *
 * Builds the same mixed-sensitivity generalized plant as
 * HinfMixedSensitivity at every original-plant vertex. The weights are shared
 * by all vertices. A common Q, Y, gamma decision vector certifies every
 * augmented vertex and recovers one full augmented-state gain K.
 */
template <ContinuousPlantModelLike Plant, size_t N,
          ContinuousPlantModelLike WpPlant, ContinuousPlantModelLike WtPlant,
          ContinuousPlantModelLike WuPlant>
struct HinfPolytopicMixedSensitivity {
  using Model = linalg::remove_cvref_t<Plant>;
  using T = plant_type_t<Model>;
  using VertexMixed = HinfMixedSensitivity<Model, WpPlant, WtPlant, WuPlant>;
  using MixedPlant = typename VertexMixed::MixedPlant;
  using Synthesis = HinfPolytopicStateFeedback<MixedPlant, N>;
  using PlantPolytope = linalg::Vector<Plant, N>;
  using MixedPolytope = linalg::Vector<MixedPlant, N>;

  static constexpr size_t StateDim = VertexMixed::StateDim;
  static constexpr size_t ControlDim = VertexMixed::ControlDim;
  static constexpr size_t ExogenousDim = VertexMixed::ExogenousDim;
  static constexpr size_t OutputDim = VertexMixed::OutputDim;
  static constexpr size_t QDim = Synthesis::QDim;
  static constexpr size_t YDim = Synthesis::YDim;
  static constexpr size_t GammaDim = Synthesis::GammaDim;
  static constexpr size_t DecVarDim = Synthesis::DecVarDim;

  using QMat = typename Synthesis::QMat;
  using YMat = typename Synthesis::YMat;
  using KMat = typename Synthesis::KMat;
  using DecVarSpace = typename Synthesis::DecVarSpace;

  /** @brief Build one mixed-sensitivity generalized plant per vertex. */
  constexpr auto model(const PlantPolytope& poly, const WpPlant& wp,
                       const WtPlant& wt, const WuPlant& wu) const {
    auto make_vertex = [&]<size_t idx>() {
      return VertexMixed().model(poly(idx), wp, wt, wu);
    };
    auto make_polytope = [&]<size_t... Is>(std::index_sequence<Is...>) {
      return MixedPolytope{make_vertex.template operator()<Is>()...};
    };
    return make_polytope(std::make_index_sequence<N>{});
  }

  /** @brief Form the common-certificate polytopic Hinf constraints. */
  template <convex::AtomLike DecVar>
  constexpr auto operator()(const PlantPolytope& poly, const WpPlant& wp,
                            const WtPlant& wt, const WuPlant& wu,
                            const DecVar& Z_hat, T margin = 1e-6) const {
    return apply(poly, wp, wt, wu, Z_hat, margin);
  }

  /** @brief Form the common-certificate polytopic Hinf constraints. */
  template <convex::AtomLike DecVar>
  constexpr auto apply(const PlantPolytope& poly, const WpPlant& wp,
                       const WtPlant& wt, const WuPlant& wu,
                       const DecVar& Z_hat, T margin = 1e-6) const {
    return Synthesis()(model(poly, wp, wt, wu), Z_hat, margin);
  }

  /** @brief Return the shared gamma objective. */
  template <convex::AtomLike DecVar>
  constexpr auto objective(const DecVar& Z_hat) const {
    return Synthesis().objective(Z_hat);
  }

  /** @brief Recover the common augmented-state feedback gain K. */
  constexpr auto K(const QMat& Q, const YMat& Y) const {
    return Synthesis().K(Q, Y);
  }
};

/** @brief Make a polytopic Hinf mixed-sensitivity synthesis problem. */
template <ContinuousPlantModelLike Plant, size_t N,
          ContinuousPlantModelLike WpPlant, ContinuousPlantModelLike WtPlant,
          ContinuousPlantModelLike WuPlant, convex::AtomLike DecVar>
constexpr auto make_hinf_polytopic_mixed_sensitivity_problem(
    const linalg::Vector<Plant, N>& poly, const WpPlant& wp, const WtPlant& wt,
    const WuPlant& wu, const DecVar& Z_hat, plant_type_t<Plant> margin = 1e-6) {
  const auto mixed =
      HinfPolytopicMixedSensitivity<Plant, N, WpPlant, WtPlant, WuPlant>();
  const auto lmis = mixed(poly, wp, wt, wu, Z_hat, margin);
  const auto prob = convex::make_problem(Z_hat).in_set(lmis);
  return prob.minimize(mixed.objective(Z_hat));
}

} // namespace control
