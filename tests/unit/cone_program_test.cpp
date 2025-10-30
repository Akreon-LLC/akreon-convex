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
    ConeProgram<Space, decltype(epi_f)> prog(epi_f);
    STATIC_REQUIRE(prog.soft_check());
  }
}
