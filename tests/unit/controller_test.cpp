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

#include "control/regulatory/controller.hpp"

#include <catch2/catch_test_macros.hpp>
#include <tuple>

#include "control/model/state_space.hpp"
#include "math.hpp"

using namespace control;
using namespace linalg;

TEST_CASE("Controller forward Euler step", "[controller_test]") {
  using Space = Matrixd<2, 2>;
  constexpr auto build = mat_traits<Space>();

  SECTION("Propagates state with ZOH Euler step and no feedthrough") {
    constexpr auto A = build(1.0, 2.0, -1.0, 0.5);
    constexpr auto B = build(0.0, 1.0, 3.0, -2.0);
    constexpr auto C = build(2.0, 0.0, 0.0, -1.0);
    constexpr auto ss =
        StateSpace<TimeDomainType::CONTINUOUS, Space, Space, Space>(A, B, C);

    constexpr auto dt = 0.1;
    constexpr auto controller = Controller(ss, dt);
    constexpr auto x = Vectord<2>{1.0, -2.0};
    constexpr auto u = Vectord<2>{0.5, 1.5};
    constexpr auto step = controller.step(x, u);
    constexpr auto x_next = std::get<0>(step);
    constexpr auto y = std::get<1>(step);

    constexpr auto xdot = A * x + B * u;
    constexpr auto expected_x_next = x + dt * xdot;
    constexpr auto expected_y = C * x;

    STATIC_REQUIRE(all_equal(x_next, expected_x_next));
    STATIC_REQUIRE(all_equal(y, expected_y));
  }

  SECTION("Includes feedthrough term in output while integrating state") {
    constexpr auto A = build(0.2, -0.4, 0.1, 0.3);
    constexpr auto B = build(1.0, 0.0, 0.0, 1.0);
    constexpr auto C = build(1.5, -0.5, 0.25, 2.0);
    constexpr auto D = build(0.5, 1.0, -1.0, 0.75);
    constexpr auto ss =
        StateSpace<TimeDomainType::CONTINUOUS, Space, Space, Space>(A, B, C, D);

    constexpr auto dt = 0.25;
    constexpr auto controller = Controller(ss, dt);
    constexpr auto x = Vectord<2>{-1.0, 2.0};
    constexpr auto u = Vectord<2>{3.0, -0.5};
    constexpr auto step = controller.step(x, u);
    constexpr auto x_next = std::get<0>(step);
    constexpr auto y = std::get<1>(step);

    constexpr auto xdot = A * x + B * u;
    constexpr auto expected_x_next = x + dt * xdot;
    constexpr auto expected_y = C * x + D * u;

    STATIC_REQUIRE(all_equal(x_next, expected_x_next));
    STATIC_REQUIRE(all_equal(y, expected_y));
  }
}
