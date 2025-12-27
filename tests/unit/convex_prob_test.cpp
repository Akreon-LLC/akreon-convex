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

TEST_CASE("Test var setup (simple)", "[convex]") {
  SECTION("Free var") {
    using MatT = Matrixd<1, 1>;
    using ConvexDom = ConvexDomain<MatT, SignType::FREE>;
    using ConvexMeta =
        ConvexAtomMeta<ConvexDom, ConvexDom, CurvatureType::AFFINE>;

    constexpr auto x = Variable<ConvexMeta>();
    STATIC_REQUIRE(dom_sign_v<meta_dom_t<atom_meta_t<decltype(x)>>> ==
                   SignType::FREE);
    STATIC_REQUIRE(dom_sign_v<meta_codom_t<atom_meta_t<decltype(x)>>> ==
                   SignType::FREE);
    STATIC_REQUIRE(meta_curv_v<atom_meta_t<decltype(x)>> ==
                   CurvatureType::AFFINE);

    constexpr auto mat = mat_traits<MatT>();
    constexpr auto four_x = x.mul(mat(4.0));
    STATIC_REQUIRE(four_x.map.linear_part() == mat(4.0));
    constexpr auto four_x_plus_two = x.add(Vectord<1>{2.0});
    STATIC_REQUIRE(four_x_plus_two.map.translation() == Vectord<1>{2.0});

    constexpr auto x0 = x(0).value();
    STATIC_REQUIRE(dom_sign_v<meta_dom_t<atom_meta_t<decltype(x0)>>> ==
                   SignType::FREE);
    STATIC_REQUIRE(dom_sign_v<meta_codom_t<atom_meta_t<decltype(x0)>>> ==
                   SignType::FREE);
    STATIC_REQUIRE(meta_curv_v<atom_meta_t<decltype(x0)>> ==
                   CurvatureType::AFFINE);

    STATIC_REQUIRE(std::tuple_size<decltype(x0.children)>::value == 1);
    STATIC_REQUIRE(!x(1).has_value());

    constexpr auto two_x0 = x0.mul(2.0);
    STATIC_REQUIRE(two_x0.map.linear_part() == mat(2.0));

    constexpr auto two_x0_plus_one = two_x0.add(1.0);
    STATIC_REQUIRE(two_x0_plus_one.map.linear_part() == mat(2.0));
    STATIC_REQUIRE(two_x0_plus_one.map.translation() == Vectord<1>{1.0});
  }
}
