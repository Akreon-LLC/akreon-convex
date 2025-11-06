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
using namespace unit_test;

#if 0
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

TEST_CASE("Test Simple Convex Program") {
  SECTION("Simple Convex Program Construction") {
    using Space = Matrixd<1, 4>;

    constexpr auto norm = NormFn<Space, L2<Vectord<4>>>();
    constexpr auto qol = QuadOverLinearFn<Space>();
    constexpr ConvexProgram<Space, decltype(norm), decltype(qol)> prog(norm,
                                                                       qol);
    STATIC_REQUIRE(prog.solve());
    constexpr auto affine_sub = AffineSubspace<Space, Matrixd<4, 4>>(
        make_identity<double, 4>(), Vectord<4>{});
    constexpr ConvexProgram<Space, decltype(norm), decltype(qol),
                            decltype(affine_sub)>
        prog2(norm, qol, affine_sub);
    STATIC_REQUIRE(prog2.solve());
  }
}
#endif
