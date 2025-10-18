/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <iostream>

#include "math.hpp"
#include "unit_test_helpers.hpp"

using namespace linalg;
using namespace convex;
using namespace unit_test;

TEST_CASE("Test norm fn", "[convex_fn]") {
  SECTION("Test L1 fn") {
    using Space = Matrixd<1, 2>;
    constexpr auto norm = NormFn<Space, L1<Vectord<2>>>();
    STATIC_REQUIRE(norm.eval(Vectord<2>{}).has_value());
    STATIC_REQUIRE(norm.eval(Vectord<2>{}).value() <= eps);
    constexpr auto epi_f = norm.epigraph();
    STATIC_REQUIRE(epi_f.is_well_formed());
    STATIC_REQUIRE(epi_f.contains(Vectord<3>{}));
  }

  SECTION("Test L2 fn") {
    using Space = Matrixd<1, 2>;
    constexpr auto norm = NormFn<Space, L2<Vectord<2>>>();
    STATIC_REQUIRE(norm.eval(Vectord<2>{}).has_value());
    STATIC_REQUIRE(norm.eval(Vectord<2>{}).value() <= eps);
    constexpr auto epi_f = norm.epigraph();
    STATIC_REQUIRE(epi_f.is_well_formed());
    STATIC_REQUIRE(epi_f.contains(Vectord<3>{}));
  }

  SECTION("Test Linf fn") {
    using Space = Matrixd<1, 2>;
    constexpr auto norm = NormFn<Space, Linf<Vectord<2>>>();
    STATIC_REQUIRE(norm.eval(Vectord<2>{}).has_value());
    STATIC_REQUIRE(norm.eval(Vectord<2>{}).value() <= eps);
    constexpr auto epi_f = norm.epigraph();
    STATIC_REQUIRE(epi_f.is_well_formed());
    STATIC_REQUIRE(epi_f.contains(Vectord<3>{}));
  }
}
