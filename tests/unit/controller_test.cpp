/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#include "control/regulatory/controller.hpp"

#include <catch2/catch_test_macros.hpp>
#include <tuple>

#include "control/model/state_space.hpp"
#include "math.hpp"

using namespace control;
using namespace linalg;

TEST_CASE("Controller discrete step", "[controller_test]") {
  using Space = Matrixd<2, 2>;
  constexpr auto build = mat_traits<Space>();

  SECTION("Propagates state with discrete dynamics and no feedthrough") {
    constexpr auto A = build(1.0, 2.0, -1.0, 0.5);
    constexpr auto B = build(0.0, 1.0, 3.0, -2.0);
    constexpr auto C = build(2.0, 0.0, 0.0, -1.0);
    constexpr auto ss =
        StateSpace<TimeDomainType::DISCRETE, Space, Space, Space>(A, B, C);

    constexpr auto dt = 0.1;
    constexpr auto controller = Controller(ss, dt);
    constexpr auto x = Vectord<2>{1.0, -2.0};
    constexpr auto u = Vectord<2>{0.5, 1.5};
    constexpr auto step = controller.step(x, u);
    constexpr auto x_next = std::get<0>(step);
    constexpr auto y = std::get<1>(step);

    constexpr auto expected_x_next = A * x + B * u;
    constexpr auto expected_y = C * x;

    STATIC_REQUIRE(all_equal(x_next, expected_x_next));
    STATIC_REQUIRE(all_equal(y, expected_y));
  }

  SECTION("Includes feedthrough term in output while propagating state") {
    constexpr auto A = build(0.2, -0.4, 0.1, 0.3);
    constexpr auto B = build(1.0, 0.0, 0.0, 1.0);
    constexpr auto C = build(1.5, -0.5, 0.25, 2.0);
    constexpr auto D = build(0.5, 1.0, -1.0, 0.75);
    constexpr auto ss =
        StateSpace<TimeDomainType::DISCRETE, Space, Space, Space>(A, B, C, D);

    constexpr auto dt = 0.25;
    constexpr auto controller = Controller(ss, dt);
    constexpr auto x = Vectord<2>{-1.0, 2.0};
    constexpr auto u = Vectord<2>{3.0, -0.5};
    constexpr auto step = controller.step(x, u);
    constexpr auto x_next = std::get<0>(step);
    constexpr auto y = std::get<1>(step);

    constexpr auto expected_x_next = A * x + B * u;
    constexpr auto expected_y = C * x + D * u;

    STATIC_REQUIRE(all_equal(x_next, expected_x_next));
    STATIC_REQUIRE(all_equal(y, expected_y));
  }
}
