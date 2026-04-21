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
template <StateSpaceLike SS, typename Controller = NoController,
          typename Observer = NoObserver, typename Reference = NoReference,
          typename Integrator = NoIntegrator, typename Saturator = NoSaturator,
          typename AntiWindup = NoAntiWindup>
struct ClosedLoop {
  using StateSpace = linalg::remove_cvref_t<SS>;
  using T = ss_type_t<StateSpace>;
  using State = state_vec_t<StateSpace>;
  using Input = input_vec_t<StateSpace>;
  using Output = output_vec_t<StateSpace>;

  /**
   * @brief ObserverState
   *
   * @param xhat (Estimated state)
   */
  template <typename O = Observer>
  struct ObserverState {
    State xhat;  ///< Estimated state
  };

  /**
   * @brief IntegratorState
   *
   * @param z (Integrator state)
   */
  template <typename I = Integrator>
  struct IntegratorState {
    typename I::IntState z;  ///< Integrator state
  };

  /**
   * @brief RuntimeState
   *
   * @param xhat (Estimated state)
   * @param z (Integrator state)
   */
  template <typename O = Observer, typename I = Integrator>
  struct RuntimeState {
    State xhat;              ///< Estimated state
    typename I::IntState z;  ///< Integrator state
  };

  /**
   * @brief step (Base CL Feedback)
   *
   * x_next = x + dt (A x + B u)
   * u = -K x
   *
   * @param x (State)
   * @param dt (Time step)
   *
   * @return (x_next)
   */
  constexpr auto step(const State& x, const T& dt) const {
    const auto u_cmd = this->control_from_state(x);
    const auto u = this->saturate_input(u_cmd);
    return std::tuple(this->plant_next(x, u, dt));
  }

  /**
   * @brief step (Reference Tracking CL Feedback)
   *
   * x_next = x + dt (A x + B u)
   * u = -K x + Kr r
   *
   * @param x (State)
   * @param r (Reference as Output)
   * @param dt (Time step)
   *
   * @return (x_next)
   */
  constexpr auto step(const State& x, const Output& r, const T& dt) const {
    const auto u_cmd = this->control_from_state(x, r);
    const auto u = this->saturate_input(u_cmd);
    return std::tuple(this->plant_next(x, u, dt));
  }

  /**
   * @brief step (Reference Tracking + Integrate SS Error CL Feedback)
   *
   * x_next = x + dt (A x + B u)
   * z_next = z + dt z_delta
   * u = -K x + Kr r - Ki z
   *
   * @param x (State)
   * @param s (Integrator State)
   * @param r (Reference as Output)
   * @param dt (Time step)
   *
   * @return (x_next, z_next)
   */
  template <typename I = Integrator>
  constexpr auto step(const State& x, const IntegratorState<I>& s,
                      const Output& r, const T& dt) const {
    static_assert(!std::same_as<I, NoIntegrator>,
                  "Integrator must be allocated for this step overload");

    const auto u_cmd = this->control_from_state(x, s.z, r);
    const auto u = this->saturate_input(u_cmd);
    const auto y = this->plant_output(x, u);
    const auto z_delta = this->integrator_delta(r, y, u_cmd, u);
    return std::tuple(this->plant_next(x, u, dt),
                      IntegratorState<I>{this->integrator_next(s.z, z_delta, dt)});
  }

  /**
   * @brief step (Observer Feedback)
   *
   * x_next = x + dt (A x + B u)
   * xhat_next = xhat + dt xhat_delta
   *
   * @param x (State)
   * @param s (ObserverState)
   * @param dt (Time step)
   *
   * @return (x_next, xhat_next)
   */
  template <typename O = Observer>
  constexpr auto step(const State& x, const ObserverState<O>& s,
                      const T& dt) const {
    static_assert(!std::same_as<O, NoObserver>,
                  "Observer must be allocated for this step overload");

    const auto u_cmd = this->control_from_state(s.xhat);
    const auto u = this->saturate_input(u_cmd);
    const auto y = this->plant_output(x, u);
    const auto xhat_delta = this->observer_delta(s.xhat, u, y);
    return std::tuple(this->plant_next(x, u, dt),
                      ObserverState<O>{this->observer_next(s.xhat, xhat_delta, dt)});
  }

  /**
   * @brief step (Reference Tracking w/ Observer Feedback)
   *
   * x_next = x + dt (A x + B u)
   * xhat_next = xhat + dt xhat_delta
   *
   * @param x (State)
   * @param s (ObserverState)
   * @param r (Reference as Output)
   * @param dt (Time step)
   *
   * @return (x_next, xhat_next)
   */
  template <typename O = Observer>
  constexpr auto step(const State& x, const ObserverState<O>& s,
                      const Output& r, const T& dt) const {
    static_assert(!std::same_as<O, NoObserver>,
                  "Observer must be allocated for this step overload");

    const auto u_cmd = this->control_from_state(s.xhat, r);
    const auto u = this->saturate_input(u_cmd);
    const auto y = this->plant_output(x, u);
    const auto xhat_delta = this->observer_delta(s.xhat, u, y);
    return std::tuple(this->plant_next(x, u, dt),
                      ObserverState<O>{this->observer_next(s.xhat, xhat_delta, dt)});
  }

  /**
   * @brief step (Reference Tracking w/ Observer Feedback & Integrator on SS Error)
   *
   * x_next = x + dt (A x + B u)
   * xhat_next = xhat + dt xhat_delta
   * z_next = z + dt z_delta
   *
   * @param x (State)
   * @param s (RuntimeState - Includes observer & integrator states)
   * @param r (Reference as Output)
   * @param dt (Time step)
   *
   * @return (x_next, xhat_next, z_next)
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
    const auto xhat_delta = this->observer_delta(s.xhat, u, y);
    const auto z_delta = this->integrator_delta(r, y, u_cmd, u);
    return std::tuple(
        this->plant_next(x, u, dt),
        RuntimeState<O, I>{this->observer_next(s.xhat, xhat_delta, dt),
                           this->integrator_next(s.z, z_delta, dt)});
  }

  /**
   * @brief plant_next
   *
   * @param x (State)
   * @param u (Input)
   * @param dt (Time step)
   *
   * @return x_next
   */
  constexpr auto plant_next(const State& x, const Input& u, const T& dt) const {
    return ss.x_next(x, u, dt);
  }

  /**
   * @brief plant_output
   *
   * y = C x + D u
   *
   * @param x (State)
   * @param u (Input)
   *
   * @return y
   */
  constexpr auto plant_output(const State& x, const Input& u) const {
    return ss.y(x, u);
  }

  /**
   * @brief control_from_state
   *
   * u = -K x
   *
   * @param x_ctrl (State input for control)
   *
   * @return u_fb
   */
  constexpr auto control_from_state(const State& x_ctrl) const {
    static_assert(!std::same_as<Controller, NoController>,
                  "Controller must be allocated for state feedback");
    return controller.u_fb(x_ctrl);
  }

  /**
   * @brief control_from_state
   *
   * u = -K x + Kr r
   *
   * @param x_ctrl (State input for control)
   * @param r (Reference as Output)
   *
   * @return u_fb + u_ff
   */
  constexpr auto control_from_state(const State& x_ctrl,
                                    const Output& r) const {
    static_assert(!std::same_as<Reference, NoReference>,
                  "Reference block must be allocated for reference tracking");
    return this->control_from_state(x_ctrl) + reference.u_ff(r);
  }

  /**
   * @brief control_from_state
   *
   * u = -K x + Kr r - Ki z
   *
   * @param x_ctrl (State input for control)
   * @param z (Integrator state)
   * @param r (Reference as Output)
   *
   * @return u_fb + u_ff + u_i
   */
  template <typename I = Integrator>
  constexpr auto control_from_state(const State& x_ctrl,
                                    const typename I::IntState& z,
                                    const Output& r) const {
    static_assert(!std::same_as<I, NoIntegrator>,
                  "Integrator must be allocated for this control overload");
    return this->control_from_state(x_ctrl, r) + integrator.u_i(z);
  }

  /**
   * @brief saturate_input
   *
   * u = sat(u_cmd)
   *
   * @param u_cmd (Commanded input)
   *
   * @return u_sat
   */
  constexpr auto saturate_input(const Input& u_cmd) const {
    if constexpr (std::same_as<Saturator, NoSaturator>) {
      return u_cmd;
    } else {
      return saturator.u(u_cmd);
    }
  }

  /**
   * @brief observer_delta
   *
   * xhat_delta = A xhat + B u + L (y - yhat)
   *
   * @param xhat (Estimated state)
   * @param u (Input)
   * @param y (Output)
   *
   * @return xhat_delta
   */
  constexpr auto observer_delta(const State& xhat, const Input& u,
                                const Output& y) const {
    static_assert(!std::same_as<Observer, NoObserver>,
                  "Observer must be allocated for observer_delta");
    return observer.xhat_delta(ss, xhat, u, y);
  }

  /**
   * @brief observer_next
   *
   * @param xhat (Estimated state)
   * @param xhat_delta (Estimated state increment)
   * @param dt (Time step)
   *
   * @return xhat_next
   */
  constexpr auto observer_next(const State& xhat, const State& xhat_delta,
                               const T& dt) const {
    return xhat + dt * xhat_delta;
  }

  /**
   * @brief integrator_delta
   *
   * z_delta = S (r - y) + Kaw (u_sat - u_cmd)
   *
   * @param r (Reference as Output)
   * @param y (Output)
   * @param u_cmd (Commanded input)
   * @param u_sat (Saturated input)
   *
   * @return z_delta
   */
  template <typename I = Integrator>
  constexpr auto integrator_delta(const Output& r, const Output& y,
                                  const Input& u_cmd,
                                  const Input& u_sat) const {
    static_assert(!std::same_as<I, NoIntegrator>,
                  "Integrator must be allocated for integrator_delta");

    auto z_delta = integrator.z_delta(r, y);

    if constexpr (!std::same_as<AntiWindup, NoAntiWindup>) {
      z_delta = z_delta + anti_windup.z_correction(u_cmd, u_sat);
    }

    return z_delta;
  }

  /**
   * @brief integrator_next
   *
   * @param z (Integrator state)
   * @param z_delta (Integrator increment)
   * @param dt (Time step)
   *
   * @return z_next
   */
  template <typename I = Integrator>
  constexpr auto integrator_next(const typename I::IntState& z,
                                 const typename I::IntState& z_delta,
                                 const T& dt) const {
    static_assert(!std::same_as<I, NoIntegrator>,
                  "Integrator must be allocated for integrator_next");
    return z + dt * z_delta;
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
