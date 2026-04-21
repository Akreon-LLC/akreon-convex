/*
 * copyright © 2025 akreon llc. all rights reserved.
 *
 * proprietary and confidential.
 * this source code is the property of akreon llc and constitutes a trade secret
 * protected by u.s. and international law.
 *
 * unauthorized copying, modification, distribution, or disclosure of this file,
 * via any medium, is strictly prohibited without express written permission
 * from akreon llc.
 */

#pragma once

#include "closed_loop.hpp"

namespace control {

/**
 * @brief ClosedLoopBuilder
 */
template <StateSpaceLike SS, typename Controller = NoController,
          typename Observer = NoObserver, typename Reference = NoReference,
          typename Integrator = NoIntegrator, typename Saturator = NoSaturator,
          typename AntiWindup = NoAntiWindup>
struct ClosedLoopBuilder {
  /// @brief alias & type extraction
  using StateSpace = linalg::remove_cvref_t<SS>;

  /**
   * @brief with_controller
   *
   * @param c (Controller)
   *
   * @return ClosedLoopBuilder
   */
  constexpr auto with_controller(const auto& c) const {
    using ControllerT = linalg::remove_cvref_t<decltype(c)>;
    return ClosedLoopBuilder<SS, ControllerT, Observer, Reference, Integrator,
                             Saturator, AntiWindup>{
        ss, c, observer, reference, integrator, saturator, anti_windup};
  }

  /**
   * @brief with_observer
   *
   * @param o (Observer)
   *
   * @return ClosedLoopBuilder
   */
  constexpr auto with_observer(const auto& o) const {
    using ObserverT = linalg::remove_cvref_t<decltype(o)>;
    return ClosedLoopBuilder<SS, Controller, ObserverT, Reference, Integrator,
                             Saturator, AntiWindup>{
        ss, controller, o, reference, integrator, saturator, anti_windup};
  }

  /**
   * @brief with_reference
   *
   * @param r (DC Tracking)
   *
   * @return ClosedLoopBuilder
   */
  constexpr auto with_reference(const auto& r) const {
    using ReferenceT = linalg::remove_cvref_t<decltype(r)>;
    return ClosedLoopBuilder<SS, Controller, Observer, ReferenceT, Integrator,
                             Saturator, AntiWindup>{
        ss, controller, observer, r, integrator, saturator, anti_windup};
  }

  /**
   * @brief with_integrator
   *
   * @param i (Integrator)
   *
   * @return ClosedLoopBuilder
   */
  constexpr auto with_integrator(const auto& i) const {
    using IntegratorT = linalg::remove_cvref_t<decltype(i)>;
    return ClosedLoopBuilder<SS, Controller, Observer, Reference, IntegratorT,
                             Saturator, AntiWindup>{
        ss, controller, observer, reference, i, saturator, anti_windup};
  }

  /**
   * @brief with_saturator
   *
   * @param s (Saturator)
   *
   * @return ClosedLoopBuilder
   */
  constexpr auto with_saturator(const auto& s) const {
    using SaturatorT = linalg::remove_cvref_t<decltype(s)>;
    return ClosedLoopBuilder<SS, Controller, Observer, Reference, Integrator,
                             SaturatorT, AntiWindup>{
        ss, controller, observer, reference, integrator, s, anti_windup};
  }

  /**
   * @brief with_anti_windup
   *
   * @param aw (AntiWindup)
   *
   * @return ClosedLoopBuilder
   */
  constexpr auto with_anti_windup(const auto& aw) const {
    using AntiWindupT = linalg::remove_cvref_t<decltype(aw)>;
    return ClosedLoopBuilder<SS, Controller, Observer, Reference, Integrator,
                             Saturator, AntiWindupT>{
        ss, controller, observer, reference, integrator, saturator, aw};
  }

  /**
   * @brief build
   *
   * @return ClosedLoop
   */
  constexpr auto build() const {
    return ClosedLoop<StateSpace, Controller, Observer, Reference, Integrator,
                      Saturator, AntiWindup>{ss,         controller, observer,
                                             reference,  integrator, saturator,
                                             anti_windup};
  }

  /** @brief member vars **/
  StateSpace ss;
  Controller controller{};
  Observer observer{};
  Reference reference{};
  Integrator integrator{};
  Saturator saturator{};
  AntiWindup anti_windup{};
};

/**
 * @brief loop
 *
 * @param ss (StateSpace)
 *
 * @return ClosedLoopBuilder
 */
template <StateSpaceLike SS>
constexpr auto loop(const SS& ss) {
  return ClosedLoopBuilder<linalg::remove_cvref_t<SS>>{ss};
}

}  // namespace control
