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

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <iostream>

#include "math.hpp"
#include "unit_test_helpers.hpp"

using namespace linalg;
using namespace convex;
using namespace affine;
using namespace conic;
using namespace unit_test;

TEST_CASE("Simple LPs", "[pd_ipm_test]") {
  SECTION("Minimal LP 2 vars") {
    using Space = Matrixd<1, 2>;
    using Codom = Matrixd<1, 3>;
    constexpr auto x = Variable<NonnegVarMeta<Space>>();
    constexpr auto prob_init = make_problem(x);

    constexpr auto A_mat = mat_traits<Matrixd<3, 2>>();
    constexpr auto A = A_mat(1.0, 1.0, 2.0, 1.0, -1.0, 2.0);
    constexpr auto b = Vectord<3>{4.0, 5.0, 2.0};
    constexpr Polyhedron<PolyhedronMeta<Space, Codom>, decltype(x)> poly(A, b,
                                                                         x);
    constexpr auto prob_constr = prob_init.in_set(poly);

    // TODO - add dot / weighted sum operator & sum squares / etc...
    constexpr auto prob_min = prob_constr.minimize(x(0).value());
    STATIC_REQUIRE(is_problem_set_v<decltype(prob_min)>);
    STATIC_REQUIRE(
        std::tuple_size<decltype(prob_min.constraint_cones())>::value == 3);
    // TODO - FIXME
    //constexpr PrimalDualIPM<decltype(prob_min)> solver;
    //constexpr auto res = solver.solve(prob_min);
  }
}
