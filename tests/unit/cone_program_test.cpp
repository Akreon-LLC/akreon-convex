/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#include <algorithm>
#include <catch2/catch_test_macros.hpp>

#include "math.hpp"
#include "unit_test_helpers.hpp"

using namespace linalg;
using namespace convex;
using namespace unit_test;

TEST_CASE("Test Simple Cone Program") {
  SECTION("Simple Cone Program Construction") {
    using Space = Matrixd<1, 3>;

    constexpr auto norm = NormFn<Space, L1<Vectord<3>>>();
    constexpr auto epi_f = norm.epigraph();
    constexpr ConeProgram<Space, decltype(epi_f)> prog(epi_f);
    STATIC_REQUIRE(prog.soft_check());

    constexpr auto ellipsoid = Ellipsoid<Space, EllipsoidType::CE>(
        Vectord<3>{}, make_identity<double, 3>());
    constexpr auto cone = ellipsoid.cone_view();
    constexpr ConeProgram<Space, decltype(epi_f), decltype(cone)> prog2(epi_f,
                                                                        cone);
    STATIC_REQUIRE(prog2.soft_check());
  }
}
