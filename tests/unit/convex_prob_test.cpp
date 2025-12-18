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

#include "math.hpp"
#include "unit_test_helpers.hpp"

using namespace linalg;
using namespace convex;
using namespace affine;
using namespace conic;
using namespace unit_test;

TEST_CASE("Test dag setup", "[convex_dag]") {
  SECTION("Test dag one atom") {
    using Space = Matrixd<1, 1>;
    using BoolMatT = Matrix<bool, 1, 1>;
    constexpr auto abs_fn = AbsFn<Space>();
    constexpr BoolMatT adj{};
    constexpr auto dag = DAG<decltype(abs_fn)>(adj, abs_fn);
    STATIC_REQUIRE(std::tuple_size<decltype(dag.atoms)>::value == 1);
    STATIC_REQUIRE(adj(0)(0) == dag.adj(0)(0));
    STATIC_REQUIRE(!has_cycle(dag));
  }
}
