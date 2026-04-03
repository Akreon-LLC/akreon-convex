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

#include "control.hpp"
#include "math.hpp"
#include "test_helpers.hpp"

using namespace linalg;
using namespace convex;
using namespace control;
using namespace unit_test;
using namespace config;

TEST_CASE("Minimal LQR 2x2", "[lqr_test]") {
  using Space = Matrixd<2, 2>;
  constexpr auto P = Variable<PsdVarMeta<Space>>();
  constexpr auto prob_init = make_problem(P);

  constexpr auto CostMat = mat_traits<Space>();

  SECTION("LQR case 1") {
    constexpr auto Q = CostMat(1.0, 0.0, 0.0, 1.0);
    constexpr auto R = CostMat(1.0, 0.0, 0.0, 1.0);
    constexpr auto A = CostMat(1.0, 0.0, 0.0, 1.0);
    constexpr auto B = CostMat(1.0, 0.0, 0.0, 1.0);
    constexpr auto C = CostMat(1.0, 0.0, 0.0, 1.0);
    constexpr auto ss =
        StateSpace<TimeDomainType::CONTINUOUS, Space, Space, Space>(A, B, C);
    constexpr auto lqr = LQR<decltype(ss), decltype(Q), decltype(R)>();
    constexpr auto lmi = lqr(ss, Q, R, P);
    constexpr auto prob_constr = prob_init.in_set(lmi);
    constexpr auto prob_min = prob_constr.minimize(0.0);
    STATIC_REQUIRE(is_problem_set_v<decltype(prob_min)>);
    STATIC_REQUIRE(
        std::tuple_size<decltype(prob_min.constraint_cones())>::value == 2);
  }
}
