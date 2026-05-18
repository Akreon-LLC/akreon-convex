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
#include "control/feedback/hinf_state_feedback.hpp"
#include "control/model/plant_model.hpp"
#include "convex/convex_problem.hpp"
#include "convex/variable.hpp"

namespace control {

/**
 * @brief HinfMixedSensitivity
 *
 * Reference -> Output for the mixed-sensitivity feedback plant
 *   z = [ Wp S; Wt T; Wu KS ] r
 *
 * The CL TF from r to z is N(K) = F_l(P, K) the LFT in r -> N -> z
 *
 * Mixed Sensitivity Hinf asks to find K s.t.
 *   min_k || F_l(P, K) ||_inf
 *   min_k || [ Wp S; Wt T; Wu KS ] ||_inf
 *
 * The above can be extended to formal BRL -> Hinf LMI where the mixed
 * weighting under Hinf norm is bounded by gamma.
 *
 * (i) Lower frequency: Wp S desired to be small at low frequencies which
 * enables good tracking and disturbance rejection
 * (ii) Mid/High frequency: Wt T desired to roll-off at cut-off frequency to
 * attenuate noise and not amplify unmodeled noise / dynamics
 * (iii) Control effort: Wu KS desired to be low to impose control actuator
 * limits
 *
 * This implementation is the feedback form of mixed-sensitivity synthesis. It
 * solves for a full-state feedback gain K over the augmented state and does not
 * synthesize a dynamic compensator.
 *
 * z = [ weighted error ; weighted output; weighted control ] r
 *
 * Internal Signals
 *   e = r - y
 *   u = K e
 *   y = G u + d
 *
 *   y = G K (r - y) + d -> (I + GK) y = G K r + d
 *
 *   T = GK (I + GK)^{-1}
 *   S = (I + GK)^{-1}
 *
 *   y = T r + S d
 *
 * T: r -> y TF for reference tracking
 * S: d -> y TF for output disturbance rejection OR r -> e shown below
 *
 * So I = T + S... low freq S = 0 & T = I and vice versa at higher freq
 *
 * (a) let r = 0 we have y = S d, S = 0 then disturbance gets attenuated
 * (b) let d = 0 we have y = T r, e = r - y so e = (I - T) r, e = S r
 * for good tracking
 *
 * Original Plant
 *   xdot = A x + B u
 *   y = C x
 *
 * Weights
 *   Wp (state space)
 *     xdot_p = A_p x_p + B_p e
 *     z_p = C_p x_p + D_p e
 *
 *     z_p = Wp(s) e
 *
 *   Wt (state space)
 *     xdot_t = A_t x_t + B_t y
 *     z_t = C_t x_t + D_t y
 *
 *     z_t = Wt(s) y
 *
 *   Wu (state space)
 *     xdot_u = A_u x_u + B_u u
 *     z_u = C_u x_u + D_u u
 *
 *     z_u = Wu(s) u
 *
 * V1 synthesis uses state feedback over the mixed-sensitivity augmented state:
 *   u = K x_ms
 *
 * Augmented state
 *   x_ms = [ x; x_p; x_t; x_u ]
 *
 * Signals and systems
 *   y = C x + D_yu u
 *   e = r - y = r - C x - D_yu u
 *
 *   xdot_ms = A_ms x_ms + B_ms [r; u]
 *
 *   A_ms = [ A        0   0   0   ]
 *          [ -B_p C   A_p 0   0   ]
 *          [ B_t C    0   A_t 0   ]
 *          [ 0        0   0   A_u ]
 *
 *   B_ms = [ 0    B        ]
 *          [ B_p -B_p D_yu ]
 *          [ 0    B_t D_yu ]
 *          [ 0    B_u      ]
 *
 *  z_p = C_p x_p + D_p(r - C x - D_yu u)
 *  z_t = C_t x_t + D_t C x
 *        + D_t D_yu u
 *  z_u = C_u x_u + D_u u
 *
 *  z = C_ms x_ms + D_ms [r; u]
 *
 *  C_ms = [ (-D_p C) C_p 0   0   ]
 *         [  D_t C   0   C_t 0   ]
 *         [  0       0   0   C_u ]
 *
 *  D_ms = [ D_p -D_p D_yu ]
 *         [ 0    D_t D_yu ]
 *         [ 0    D_u      ]
 *
 * This StateSpace realization is passed to HinfStateFeedback.
 */
template <ContinuousPlantModelLike Plant, ContinuousPlantModelLike WpPlant,
          ContinuousPlantModelLike WtPlant, ContinuousPlantModelLike WuPlant>
struct HinfMixedSensitivity {
  /// @brief extracting alias and types
  using Model = linalg::remove_cvref_t<Plant>;
  using WpModel = linalg::remove_cvref_t<WpPlant>;
  using WtModel = linalg::remove_cvref_t<WtPlant>;
  using WuModel = linalg::remove_cvref_t<WuPlant>;
  using T = plant_type_t<Model>;

  static constexpr size_t PlantStateDim = plant_traits<Model>::StateDim;
  static constexpr size_t WpStateDim = plant_traits<WpModel>::StateDim;
  static constexpr size_t WtStateDim = plant_traits<WtModel>::StateDim;
  static constexpr size_t WuStateDim = plant_traits<WuModel>::StateDim;
  static constexpr size_t StateDim =
      PlantStateDim + WpStateDim + WtStateDim + WuStateDim;

  static constexpr size_t ControlDim = plant_traits<Model>::ControlDim;
  static constexpr size_t MeasurementDim = plant_traits<Model>::MeasurementDim;
  static constexpr size_t ExogenousDim = MeasurementDim;

  static constexpr size_t WpOutputDim = plant_traits<WpModel>::PerformanceDim;
  static constexpr size_t WtOutputDim = plant_traits<WtModel>::PerformanceDim;
  static constexpr size_t WuOutputDim = plant_traits<WuModel>::PerformanceDim;
  static constexpr size_t OutputDim = WpOutputDim + WtOutputDim + WuOutputDim;

  static constexpr size_t WpInputDim = plant_traits<WpModel>::ControlDim;
  static constexpr size_t WtInputDim = plant_traits<WtModel>::ControlDim;
  static constexpr size_t WuInputDim = plant_traits<WuModel>::ControlDim;

  static_assert(WpInputDim == MeasurementDim,
                "Wp control input must match plant measurement dimension");
  static_assert(WtInputDim == MeasurementDim,
                "Wt control input must match plant measurement dimension");
  static_assert(WuInputDim == ControlDim,
                "Wu control input must match plant control dimension");

  static constexpr size_t XEnd = PlantStateDim;
  static constexpr size_t WpEnd = PlantStateDim + WpStateDim;
  static constexpr size_t WtEnd = WpEnd + WtStateDim;
  static constexpr size_t WuEnd = WtEnd + WuStateDim;
  static constexpr size_t ZpEnd = WpOutputDim;
  static constexpr size_t ZtEnd = WpOutputDim + WtOutputDim;
  static constexpr size_t ZuEnd = OutputDim;
  static constexpr size_t InputDim = ExogenousDim + ControlDim;

  using AMat = linalg::Matrix<T, StateDim, StateDim>;
  using BMat = linalg::Matrix<T, StateDim, InputDim>;
  using CMat = linalg::Matrix<T, OutputDim, StateDim>;
  using DMat = linalg::Matrix<T, OutputDim, InputDim>;
  using SS = StateSpace<TimeDomainType::CONTINUOUS, AMat, BMat, CMat>;
  using Inputs = PortLayout<Port<PortRole::EXOGENOUS, 0, ExogenousDim>,
                            Port<PortRole::CONTROL, ExogenousDim, ControlDim>>;
  using Outputs = PortLayout<Port<PortRole::PERFORMANCE, 0, OutputDim>>;
  using MixedPlant = PlantModel<SS, Inputs, Outputs>;
  using Synthesis = HinfStateFeedback<MixedPlant>;

  using QMat = typename Synthesis::QMat;
  using YMat = typename Synthesis::YMat;
  using KMat = typename Synthesis::KMat;
  using DecVarSpace = typename Synthesis::DecVarSpace;

  static constexpr size_t QDim = Synthesis::QDim;
  static constexpr size_t YDim = Synthesis::YDim;
  static constexpr size_t GammaDim = Synthesis::GammaDim;
  static constexpr size_t DecVarDim = Synthesis::DecVarDim;

  /**
   * @brief form the mixed-sensitivity augmented generalized plant
   *
   * @param plant original plant with CONTROL input and MEASUREMENT output
   * @param wp sensitivity weight from e to z_p
   * @param wt complementary-sensitivity weight from y to z_t
   * @param wu control-sensitivity weight from u to z_u
   *
   * @return generalized PlantModel for HinfStateFeedback
   */
  constexpr auto model(const Plant& plant, const WpPlant& wp, const WtPlant& wt,
                       const WuPlant& wu) const {
    AMat A_ms{};
    BMat B_ms{};
    CMat C_ms{};
    DMat D_ms{};

    const auto A = plant.dynamics_matrix();
    const auto B = plant.control_matrix();
    const auto C = plant.measurement_matrix();
    const auto D_yu =
        plant.template feedthrough<PortRole::MEASUREMENT, PortRole::CONTROL>();

    const auto A_p = wp.dynamics_matrix();
    const auto B_p = wp.control_matrix();
    const auto C_p = wp.performance_matrix();
    const auto D_p =
        wp.template feedthrough<PortRole::PERFORMANCE, PortRole::CONTROL>();

    const auto A_t = wt.dynamics_matrix();
    const auto B_t = wt.control_matrix();
    const auto C_t = wt.performance_matrix();
    const auto D_t =
        wt.template feedthrough<PortRole::PERFORMANCE, PortRole::CONTROL>();

    const auto A_u = wu.dynamics_matrix();
    const auto B_u = wu.control_matrix();
    const auto C_u = wu.performance_matrix();
    const auto D_u =
        wu.template feedthrough<PortRole::PERFORMANCE, PortRole::CONTROL>();

    A_ms = linalg::set_block<0, XEnd, 0, XEnd>(A_ms, A);
    A_ms = linalg::set_block<XEnd, WpEnd, 0, XEnd>(A_ms, T(-1) * B_p * C);
    A_ms = linalg::set_block<XEnd, WpEnd, XEnd, WpEnd>(A_ms, A_p);
    A_ms = linalg::set_block<WpEnd, WtEnd, 0, XEnd>(A_ms, B_t * C);
    A_ms = linalg::set_block<WpEnd, WtEnd, WpEnd, WtEnd>(A_ms, A_t);
    A_ms = linalg::set_block<WtEnd, WuEnd, WtEnd, WuEnd>(A_ms, A_u);

    B_ms = linalg::set_block<XEnd, WpEnd, 0, ExogenousDim>(B_ms, B_p);
    B_ms = linalg::set_block<0, XEnd, ExogenousDim, InputDim>(B_ms, B);
    B_ms = linalg::set_block<XEnd, WpEnd, ExogenousDim, InputDim>(
        B_ms, T(-1) * B_p * D_yu);
    B_ms = linalg::set_block<WpEnd, WtEnd, ExogenousDim, InputDim>(B_ms,
                                                                   B_t * D_yu);
    B_ms = linalg::set_block<WtEnd, WuEnd, ExogenousDim, InputDim>(B_ms, B_u);

    C_ms = linalg::set_block<0, ZpEnd, 0, XEnd>(C_ms, T(-1) * D_p * C);
    C_ms = linalg::set_block<0, ZpEnd, XEnd, WpEnd>(C_ms, C_p);
    C_ms = linalg::set_block<ZpEnd, ZtEnd, 0, XEnd>(C_ms, D_t * C);
    C_ms = linalg::set_block<ZpEnd, ZtEnd, WpEnd, WtEnd>(C_ms, C_t);
    C_ms = linalg::set_block<ZtEnd, ZuEnd, WtEnd, WuEnd>(C_ms, C_u);
    D_ms = linalg::set_block<0, ZpEnd, 0, ExogenousDim>(D_ms, D_p);
    D_ms = linalg::set_block<0, ZpEnd, ExogenousDim, InputDim>(
        D_ms, T(-1) * D_p * D_yu);
    D_ms = linalg::set_block<ZpEnd, ZtEnd, ExogenousDim, InputDim>(D_ms,
                                                                   D_t * D_yu);
    D_ms = linalg::set_block<ZtEnd, ZuEnd, ExogenousDim, InputDim>(D_ms, D_u);
    return MixedPlant(SS(A_ms, B_ms, C_ms, D_ms));
  }

  /**
   * @brief operator overload to form mixed-sensitivity Hinf state feedback
   */
  template <convex::AtomLike DecVar>
  constexpr auto operator()(const Plant& plant, const WpPlant& wp,
                            const WtPlant& wt, const WuPlant& wu,
                            const DecVar& Z_hat, T margin = 1e-6) const {
    return apply(plant, wp, wt, wu, Z_hat, margin);
  }

  /**
   * @brief apply
   *
   * @return Hinf synthesis constraints for the mixed-sensitivity plant
   */
  template <convex::AtomLike DecVar>
  constexpr auto apply(const Plant& plant, const WpPlant& wp, const WtPlant& wt,
                       const WuPlant& wu, const DecVar& Z_hat,
                       T margin = 1e-6) const {
    return Synthesis()(model(plant, wp, wt, wu), Z_hat, margin);
  }

  /**
   * @brief objective
   *
   * @return gamma
   */
  template <convex::AtomLike DecVar>
  constexpr auto objective(const DecVar& Z_hat) const {
    return Synthesis().objective(Z_hat);
  }

  /**
   * @brief K (mixed-sensitivity full-state feedback gain)
   *
   * @param Q inverse BRL Lyapunov variable
   * @param Y lifted feedback product Y = K Q
   *
   * @return K
   */
  constexpr auto K(const QMat& Q, const YMat& Y) const {
    return Synthesis().K(Q, Y);
  }
};

/**
 * @brief make Hinf mixed-sensitivity problem
 */
template <ContinuousPlantModelLike Plant, ContinuousPlantModelLike WpPlant,
          ContinuousPlantModelLike WtPlant, ContinuousPlantModelLike WuPlant,
          convex::AtomLike DecVar>
constexpr auto make_hinf_mixed_sensitivity_problem(
    const Plant& plant, const WpPlant& wp, const WtPlant& wt, const WuPlant& wu,
    const DecVar& Z_hat, plant_type_t<Plant> margin = 1e-6) {
  const auto mixed = HinfMixedSensitivity<Plant, WpPlant, WtPlant, WuPlant>();
  const auto lmis = mixed(plant, wp, wt, wu, Z_hat, margin);
  const auto prob = convex::make_problem(Z_hat).in_set(lmis);
  const auto obj = mixed.objective(Z_hat);
  return prob.minimize(obj);
}

} // namespace control
