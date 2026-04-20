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

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "control.hpp"

using namespace linalg;
using namespace control;
using Catch::Matchers::WithinAbs;

TEST_CASE("Closed-loop regulatory blocks compose and step coherently",
          "[closed_loop_test]") {
  using Space = Matrixd<1, 1>;
  using Vec = Vector<double, 1>;

  constexpr auto A = Space{2.0};
  constexpr auto B = Space{3.0};
  constexpr auto C = Space{1.0};
  constexpr auto ss =
      StateSpace<TimeDomainType::CONTINUOUS, Space, Space, Space>(A, B, C);

  constexpr auto K = Space{4.0};
  constexpr auto L = Space{7.0};
  constexpr auto Kr = Space{5.0};
  constexpr auto S = Space{1.0};
  constexpr auto Ki = Space{6.0};
  constexpr auto lower = Vec{-1.0};
  constexpr auto upper = Vec{1.0};
  constexpr auto Kaw = Space{8.0};

  const auto controller = state_feedback_block<decltype(ss)>(K);
  const auto observer = luenberger_observer_block<decltype(ss)>(L);
  const auto reference = dc_tracking_block<decltype(ss)>(Kr);
  const auto integrator = output_integrator<decltype(ss), 1>(S, Ki);
  const auto saturator = input_saturation_block<decltype(ss)>(lower, upper);
  const auto anti_windup = anti_windup_block<decltype(ss), 1>(Kaw);

  SECTION("Blocks apply their local operators") {
    constexpr auto x = Vec{0.5};
    constexpr auto z = Vec{0.25};
    constexpr auto y = Vec{1.0};
    constexpr auto r = Vec{2.0};
    constexpr auto u = Vec{6.5};

    REQUIRE_THAT(controller.u_fb(x)(0), WithinAbs(-2.0, 1e-12));
    REQUIRE_THAT(reference.u_ff(r)(0), WithinAbs(10.0, 1e-12));
    REQUIRE_THAT(integrator.u_i(z)(0), WithinAbs(-1.5, 1e-12));
    REQUIRE_THAT(integrator.z_dot(r, y)(0), WithinAbs(1.0, 1e-12));
    REQUIRE_THAT(observer.xhat_dot(ss, x, u, y)(0), WithinAbs(24.0, 1e-12));
    REQUIRE_THAT(saturator.u(Vec{2.5})(0), WithinAbs(1.0, 1e-12));
    REQUIRE_THAT(anti_windup.z_correction(Vec{2.5}, Vec{1.0})(0),
                 WithinAbs(-12.0, 1e-12));
  }

  SECTION("Builder wires controller, observer, reference, and integrator") {
    const auto cl = loop(ss)
                        .with_controller(controller)
                        .with_observer(observer)
                        .with_reference(reference)
                        .with_integrator(integrator)
                        .build();

    constexpr auto x = Vec{1.0};
    constexpr auto xhat = Vec{0.5};
    constexpr auto z = Vec{0.25};
    constexpr auto r = Vec{2.0};
    constexpr auto dt = 0.1;

    const auto [x_next, runtime_next] =
        cl.step(x, decltype(cl)::RuntimeState<>{xhat, z}, r, dt);
    REQUIRE_THAT(x_next(0), WithinAbs(3.15, 1e-12));
    REQUIRE_THAT(runtime_next.xhat(0), WithinAbs(2.9, 1e-12));
    REQUIRE_THAT(runtime_next.z(0), WithinAbs(0.35, 1e-12));
  }

  SECTION("Observer-free closed loop advances plant and integrator state") {
    const auto cl = loop(ss)
                        .with_controller(controller)
                        .with_reference(reference)
                        .with_integrator(integrator)
                        .build();

    constexpr auto x = Vec{1.0};
    constexpr auto z = Vec{0.25};
    constexpr auto r = Vec{2.0};
    constexpr auto dt = 0.1;

    const auto [x_next, integrator_next] =
        cl.step(x, decltype(cl)::IntegratorState<>{z}, r, dt);
    REQUIRE_THAT(x_next(0), WithinAbs(2.55, 1e-12));
    REQUIRE_THAT(integrator_next.z(0), WithinAbs(0.35, 1e-12));
  }

  SECTION("Integrator-free closed loop advances plant and observer state") {
    const auto cl = loop(ss)
                        .with_controller(controller)
                        .with_observer(observer)
                        .with_reference(reference)
                        .build();

    constexpr auto x = Vec{1.0};
    constexpr auto xhat = Vec{0.5};
    constexpr auto r = Vec{2.0};
    constexpr auto dt = 0.1;

    const auto [x_next, observer_next] =
        cl.step(x, decltype(cl)::ObserverState<>{xhat}, r, dt);
    REQUIRE_THAT(x_next(0), WithinAbs(3.6, 1e-12));
    REQUIRE_THAT(observer_next.xhat(0), WithinAbs(3.35, 1e-12));
  }

  SECTION("Saturation is applied before plant and observer dynamics") {
    const auto cl = loop(ss)
                        .with_controller(controller)
                        .with_observer(observer)
                        .with_reference(reference)
                        .with_saturator(saturator)
                        .build();

    constexpr auto x = Vec{1.0};
    constexpr auto xhat = Vec{0.5};
    constexpr auto r = Vec{2.0};
    constexpr auto dt = 0.1;

    const auto u_cmd = cl.control_from_state(xhat, r);
    const auto u_sat = cl.saturate_input(u_cmd);
    REQUIRE_THAT(u_cmd(0), WithinAbs(8.0, 1e-12));
    REQUIRE_THAT(u_sat(0), WithinAbs(1.0, 1e-12));

    const auto [x_next, observer_next] =
        cl.step(x, decltype(cl)::ObserverState<>{xhat}, r, dt);
    REQUIRE_THAT(x_next(0), WithinAbs(1.5, 1e-12));
    REQUIRE_THAT(observer_next.xhat(0), WithinAbs(1.25, 1e-12));
  }

  SECTION("Anti-windup feeds saturation residual into integrator dynamics") {
    const auto cl = loop(ss)
                        .with_controller(controller)
                        .with_reference(reference)
                        .with_integrator(integrator)
                        .with_saturator(saturator)
                        .with_anti_windup(anti_windup)
                        .build();

    constexpr auto x = Vec{1.0};
    constexpr auto z = Vec{0.25};
    constexpr auto r = Vec{2.0};
    constexpr auto dt = 0.1;

    const auto u_cmd = cl.control_from_state(x, z, r);
    const auto u_sat = cl.saturate_input(u_cmd);
    const auto y = cl.plant_output(x, u_sat);
    const auto zdot = cl.integrator_dynamics(r, y, u_cmd, u_sat);

    REQUIRE_THAT(u_cmd(0), WithinAbs(4.5, 1e-12));
    REQUIRE_THAT(u_sat(0), WithinAbs(1.0, 1e-12));
    REQUIRE_THAT(zdot(0), WithinAbs(-27.0, 1e-12));

    const auto [x_next, integrator_next] =
        cl.step(x, decltype(cl)::IntegratorState<>{z}, r, dt);
    REQUIRE_THAT(x_next(0), WithinAbs(1.5, 1e-12));
    REQUIRE_THAT(integrator_next.z(0), WithinAbs(-2.45, 1e-12));
  }
}
