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

#include <tuple>

#include "control/control_traits.hpp"

namespace control {

/// @brief empty types
struct NoController {};
struct NoObserver {};
struct NoReference {};
struct NoIntegrator {};
struct NoSaturator {};
struct NoAntiWindup {};

/**
 * @brief ClosedLoop
 */
template <ContinuousStateSpaceLike SS, typename Controller = NoController,
          typename Observer = NoObserver, typename Reference = NoReference,
          typename Integrator = NoIntegrator, typename Saturator = NoSaturator,
          typename AntiWindup = NoAntiWindup>
struct ClosedLoop {
  /// @brief alias & extract
  using StateSpace = linalg::remove_cvref_t<SS>;
  using T = ss_type_t<StateSpace>;
  using State = state_vec_t<StateSpace>;
  using Input = input_vec_t<StateSpace>;
  using Output = output_vec_t<StateSpace>;

  /**
   * @brief ObserverState
   */
  template <typename O = Observer>
  struct ObserverState {
    State xhat;  ///< Estimated state
  };

  /**
   * @brief IntegratorState
   */
  template <typename I = Integrator>
  struct IntegratorState {
    typename I::IntState z;  ///< Integrator State
  };

  /**
   * @brief RuntimeState
   */
  template <typename O = Observer, typename I = Integrator>
  struct RuntimeState {
    State xhat;              ///< Estimated state
    typename I::IntState z;  ///< Integrator State
  };

  /**
   * @brief step (Based CL Feedback)
   *
   * xdot = A x + B u
   * u = -K x
   *
   * @param x (State)
   * @param dt (time step)
   *
   * @return (x)
   */
  constexpr auto step(const State& x, const T& dt) const {
    const auto u_cmd = this->control_from_state(x);
    const auto u = this->saturate_input(u_cmd);
    const auto xdot = this->plant_dynamics(x, u);
    return std::tuple(x + dt * xdot);
  }

  /**
   * @brief step (Reference Tracking CL Feedback)
   *
   * xdot = A x + B u
   * u = -K x + Kr r
   *
   * @param x (State)
   * @param r (Reference as Output)
   * @param dt (time step)
   *
   * @return (x)
   */
  constexpr auto step(const State& x, const Output& r, const T& dt) const {
    const auto u_cmd = this->control_from_state(x, r);
    const auto u = this->saturate_input(u_cmd);
    const auto xdot = this->plant_dynamics(x, u);
    return std::tuple(x + dt * xdot);
  }

  /**
   * @brief step (Reference Tracking + Integrate SS Error CL Feedback)
   *
   * xdot = A x + B u
   * zdot = S (r - y)
   * u = -K x + Kr r - Ki z
   *
   * @param x (State)
   * @param s (Integrator State)
   * @param r (Reference as Output)
   * @param dt (time step)
   *
   * @return (x, z)
   */
  template <typename I = Integrator>
  constexpr auto step(const State& x, const IntegratorState<I>& s,
                      const Output& r, const T& dt) const {
    const auto u_cmd = this->control_from_state(x, s.z, r);
    const auto u = this->saturate_input(u_cmd);
    const auto y = this->plant_output(x, u);
    const auto xdot = this->plant_dynamics(x, u);
    const auto zdot = this->integrator_dynamics(r, y, u_cmd, u);
    return std::tuple(x + dt * xdot, IntegratorState<I>{s.z + dt * zdot});
  }

  /**
   * @brief step (Observer Feedback)
   *
   * xdot = A x + B u
   * xhatdot = A xhat + B u + L (y - yhat)
   *
   * @param x (State)
   * @param s (ObserverState)
   * @param dt (time step)
   *
   * @return (x, xhat)
   */
  template <typename O = Observer>
  constexpr auto step(const State& x, const ObserverState<O>& s,
                      const T& dt) const {
    static_assert(!std::same_as<O, NoObserver>,
                  "Observer must be allocated for this step overload");

    const auto u_cmd = this->control_from_state(s.xhat);
    const auto u = this->saturate_input(u_cmd);
    const auto y = this->plant_output(x, u);
    const auto xdot = this->plant_dynamics(x, u);
    const auto xhatdot = this->observer_dynamics(s.xhat, u, y);
    return std::tuple(x + dt * xdot, ObserverState<O>{s.xhat + dt * xhatdot});
  }

  /**
   * @brief step (Reference Tracking w/ Observer Feedback)
   *
   * xdot = A x + B u
   * xhatdot = A xhat + B u + L (y - yhat)
   *
   * @param x (State)
   * @param s (ObserverState)
   * @param r (Reference as Output)
   * @param dt (time step)
   *
   * @return (x, xhat)
   */
  template <typename O = Observer>
  constexpr auto step(const State& x, const ObserverState<O>& s,
                      const Output& r, const T& dt) const {
    static_assert(!std::same_as<O, NoObserver>,
                  "Observer must be allocated for this step overload");

    const auto u_cmd = this->control_from_state(s.xhat, r);
    const auto u = this->saturate_input(u_cmd);
    const auto y = this->plant_output(x, u);
    const auto xdot = this->plant_dynamics(x, u);
    const auto xhatdot = this->observer_dynamics(s.xhat, u, y);
    return std::tuple(x + dt * xdot, ObserverState<O>{s.xhat + dt * xhatdot});
  }

  /**
   * @brief step (Reference Tracking w/ Observer Feedback & Integrator on SS
   * Error)
   *
   * xdot = A x + B u
   * xhatdot = A xhat + B u + L (y - yhat)
   * zdot = S (r - y)
   *
   * @param x (State)
   * @param s (RuntimeState - Includes observer & integrator states)
   * @param r (Reference as Output)
   * @param dt (time step)
   *
   * @return (x, xhat, z)
   */
  template <typename O = Observer, typename I = Integrator>
  constexpr auto step(const State& x, const RuntimeState<O, I>& s,
                      const Output& r, const T& dt) const {
    static_assert(!std::same_as<O, NoObserver>,
                  "Observer must be allocated for this step overload");
    static_assert(!std::same_as<I, NoIntegrator>,
                  "Integrator must be allocated for this step overload");

    const auto u_cmd = this->control_from_state(s.xhat, s.z, r);
    const auto u = this->saturate_input(u_cmd);
    const auto y = this->plant_output(x, u);
    const auto xdot = this->plant_dynamics(x, u);
    const auto xhatdot = this->observer_dynamics(s.xhat, u, y);
    const auto zdot = this->integrator_dynamics(r, y, u_cmd, u);
    return std::tuple(x + dt * xdot, RuntimeState<O, I>{s.xhat + dt * xhatdot,
                                                        s.z + dt * zdot});
  }

  /**
   * @brief x_dot (plant dynamics)
   *
   * @param x (State)
   * @param u (Input)
   *
   * @return xdot
   */
  constexpr auto plant_dynamics(const State& x, const Input& u) const {
    return ss.A * x + ss.B * u;
  }

  /**
   * @brief y (plant output)
   *
   * @param x (State)
   * @param u (Input)
   *
   * @return y
   */
  constexpr auto plant_output(const State& x, const Input& u) const {
    return ss.D.has_value() ? ss.C * x + (*ss.D) * u : ss.C * x;
  }

  /**
   * @brief u
   *
   * @param x_ctrl (State input for control)
   *
   * @return u = u_fb
   */
  constexpr auto control_from_state(const State& x_ctrl) const {
    static_assert(!std::same_as<Controller, NoController>,
                  "Controller must be allocated for state feedback");
    return controller.u_fb(x_ctrl);
  }

  /**
   * @brief u
   *
   * @param x_ctrl (State input for control)
   * @param r (Output)
   *
   * @return u = u_fb + u_ff
   */
  constexpr auto control_from_state(const State& x_ctrl,
                                    const Output& r) const {
    static_assert(!std::same_as<Reference, NoReference>,
                  "DC Tracking must be initialized for reference tracking");
    const auto u = this->control_from_state(x_ctrl);
    return u + reference.u_ff(r);
  }

  /**
   * @brief u
   *
   * @param x_ctrl (State input for control)
   * @param z (Integrator State)
   * @param r (Reference as Output)
   *
   * @return u = u_fb + u_ff + u_i
   */
  template <typename I = Integrator>
  constexpr auto control_from_state(const State& x_ctrl,
                                    const typename I::IntState& z,
                                    const Output& r) const {
    static_assert(
        !std::same_as<I, NoIntegrator>,
        "Integrator must be allocated for managing steady state error");
    const auto u = this->control_from_state(x_ctrl, r);
    return u + integrator.u_i(z);
  }

  /**
   * @brief saturate_input
   *
   * @param u_cmd (Input commanded)
   *
   * @return u = sat(u_cmd)
   */
  constexpr auto saturate_input(const Input& u_cmd) const {
    if constexpr (std::same_as<Saturator, NoSaturator>) {
      return u_cmd;
    } else {
      return saturator.u(u_cmd);
    }
  }

  /**
   * @brief integrator_dynamics
   *
   * @param r (Reference as Output)
   * @param y (Output)
   * @param u_cmd (Commanded Input)
   * @param u_sat (Saturated Input)
   *
   * @return zdot = S(r - y) + Kaw (u_sat - u_cmd)
   */
  constexpr auto integrator_dynamics(const Output& r, const Output& y,
                                     const Input& u_cmd,
                                     const Input& u_sat) const {
    static_assert(!std::same_as<Integrator, NoIntegrator>,
                  "Integrator must be allocated for integrator_dynamics");

    auto zdot = integrator.z_dot(r, y);

    if constexpr (!std::same_as<AntiWindup, NoAntiWindup>) {
      zdot = zdot + anti_windup.z_correction(u_cmd, u_sat);
    }

    return zdot;
  }

  /**
   * @brief xhat_dot
   *
   * @param xhat (Estimated state)
   * @param u (Input)
   * @param y (Output)
   * 
   * @return xhat_dot
   */
  constexpr auto observer_dynamics(const State& xhat, const Input& u,
                                   const Output& y) const {
    static_assert(!std::same_as<Observer, NoObserver>,
                  "Observer must be allocated for observer_dynamics to run");
    return observer.xhat_dot(ss, xhat, u, y);
  }

  /** @brief member variables **/
  StateSpace ss;
  Controller controller{};
  Observer observer{};
  Reference reference{};
  Integrator integrator{};
  Saturator saturator{};
  AntiWindup anti_windup{};
};

}  // namespace control
