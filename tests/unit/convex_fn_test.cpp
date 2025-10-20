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
    STATIC_REQUIRE(norm.eval(Vectord<2>{1.0, 2.0}).value() == 3.0);
    constexpr auto epi_f = norm.epigraph();
    STATIC_REQUIRE(epi_f.is_well_formed());
    STATIC_REQUIRE(epi_f.contains(Vectord<3>{}));
    STATIC_REQUIRE(epi_f.contains(Vectord<3>{1.0, 2.0, 3.0}));
    STATIC_REQUIRE(!epi_f.contains(Vectord<3>{1.0, 2.0, 1.0}));
  }

  SECTION("Test L2 fn") {
    using Space = Matrixd<1, 2>;
    constexpr auto norm = NormFn<Space, L2<Vectord<2>>>();
    STATIC_REQUIRE(norm.eval(Vectord<2>{}).has_value());
    STATIC_REQUIRE(norm.eval(Vectord<2>{}).value() <= eps);
    STATIC_REQUIRE(norm.eval(Vectord<2>{1.0, 2.0}).value() ==
                   constexpr_sqrt(5.0));
    constexpr auto epi_f = norm.epigraph();
    STATIC_REQUIRE(epi_f.is_well_formed());
    STATIC_REQUIRE(epi_f.contains(Vectord<3>{}));
    STATIC_REQUIRE(epi_f.contains(Vectord<3>{1.0, 2.0, constexpr_sqrt(5.0)}));
    STATIC_REQUIRE(!epi_f.contains(Vectord<3>{1.0, 2.0, 1.0}));
  }

  SECTION("Test Linf fn") {
    using Space = Matrixd<1, 2>;
    constexpr auto norm = NormFn<Space, Linf<Vectord<2>>>();
    STATIC_REQUIRE(norm.eval(Vectord<2>{}).has_value());
    STATIC_REQUIRE(norm.eval(Vectord<2>{}).value() <= eps);
    STATIC_REQUIRE(norm.eval(Vectord<2>{1.0, 2.0}).value() == 2.0);
    constexpr auto epi_f = norm.epigraph();
    STATIC_REQUIRE(epi_f.is_well_formed());
    STATIC_REQUIRE(epi_f.contains(Vectord<3>{}));
    STATIC_REQUIRE(epi_f.contains(Vectord<3>{1.0, 2.0, 2.0}));
    STATIC_REQUIRE(!epi_f.contains(Vectord<3>{1.0, 2.0, 1.0}));
  }
}

TEST_CASE("Test affine fn", "[convex_fn]") {
  SECTION("Test affine fn (trivial)") {
    using Space = Matrixd<1, 2>;
    constexpr auto x0 = Vectord<1>{};
    constexpr auto U = Space{};
    constexpr auto map = AffineMap<Space>(U, x0);
    constexpr auto affine_fn = AffineFn<Space>(map);
    STATIC_REQUIRE(affine_fn.eval(Vectord<2>{}).has_value());
    STATIC_REQUIRE(affine_fn.eval(Vectord<2>{}).value() == 0.0);
    constexpr auto epi_f = affine_fn.epigraph();
    STATIC_REQUIRE(epi_f.is_well_formed());
    STATIC_REQUIRE(epi_f.contains(Vectord<3>{}));
    STATIC_REQUIRE(!epi_f.contains(Vectord<3>{1.0, 2.0, -1.0}));
  }
  SECTION("Test affine fn") {
    using Space = Matrixd<1, 3>;
    constexpr auto x0 = Vectord<1>{2.0};
    constexpr auto U = Space{Vectord<3>{1.0, 5.0, 2.0}};
    constexpr auto map = AffineMap<Space>(U, x0);
    constexpr auto affine_fn = AffineFn<Space>(map);
    STATIC_REQUIRE(affine_fn.eval(Vectord<3>{}).has_value());
    STATIC_REQUIRE(affine_fn.eval(Vectord<3>{}).value() == 2.0);
    constexpr auto epi_f = affine_fn.epigraph();
    STATIC_REQUIRE(epi_f.is_well_formed());
    STATIC_REQUIRE(epi_f.contains(Vectord<4>{}));
    STATIC_REQUIRE(epi_f.contains(Vectord<4>{1.0, 1.0, 1.0, 4.0}));
    STATIC_REQUIRE(!epi_f.contains(Vectord<4>{1.0, 1.0, 0.0, 1.0}));
  }
}
