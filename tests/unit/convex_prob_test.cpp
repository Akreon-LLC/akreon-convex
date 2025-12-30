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

TEST_CASE("Test var setup (simple)", "[convex]") {
  SECTION("Free var") {
    using MatT = Matrixd<1, 1>;
    constexpr auto x = Variable<FreeVarMeta<MatT>>();
    STATIC_REQUIRE(dom_sign_v<meta_dom_t<atom_meta_t<decltype(x)>>> ==
                   SignType::FREE);
    STATIC_REQUIRE(dom_sign_v<meta_codom_t<atom_meta_t<decltype(x)>>> ==
                   SignType::FREE);
    STATIC_REQUIRE(meta_curv_v<atom_meta_t<decltype(x)>> ==
                   CurvatureType::AFFINE);
    STATIC_REQUIRE(is_var_type_v<decltype(x)>);

    constexpr auto mat = mat_traits<MatT>();
    constexpr auto four_x = x.mul(mat(4.0));
    STATIC_REQUIRE(four_x.map.linear_part() == mat(4.0));

    constexpr auto four_x_plus_two = four_x.add(Vectord<1>{2.0});
    STATIC_REQUIRE(four_x_plus_two.map.translation() == Vectord<1>{2.0});

    constexpr auto builder = ConeMapBuilder<MatT>();
    constexpr auto canon1 = four_x_plus_two.canonicalize(builder);
    constexpr auto canon1_map = std::get<0>(canon1);
    constexpr auto canon1_cb = std::get<1>(canon1);

    STATIC_REQUIRE(canon1_map.linear_part() == mat(4.0));
    STATIC_REQUIRE(canon1_map.translation() == Vectord<1>{2.0});
    STATIC_REQUIRE(std::tuple_size<decltype(canon1_cb.cones)>::value == 0);

    constexpr auto x0 = x(0).value();
    STATIC_REQUIRE(is_var_type_v<decltype(x0)>);
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

    constexpr auto canon2 = two_x0_plus_one.canonicalize(builder);
    constexpr auto canon2_map = std::get<0>(canon2);
    constexpr auto canon2_cb = std::get<1>(canon2);
    STATIC_REQUIRE(canon2_map.linear_part() == mat(2.0));
    STATIC_REQUIRE(canon2_map.translation() == Vectord<1>{1.0});
    STATIC_REQUIRE(std::tuple_size<decltype(canon2_cb.cones)>::value == 0);

#if 0
    using Mat2T = Matrixd<1, 2>;
    constexpr auto sum =
        Sum<SumConvexMeta<Mat2T, MatT>, decltype(x0), decltype(x0)>(x0, x0);
    STATIC_REQUIRE(std::tuple_size<decltype(sum.children)>::value == 2);
#endif
  }

#if 0
  SECTION("Sum of abs") {
    using MatT = Matrixd<1, 2>;
    constexpr auto x = Variable<FreeVarMeta<MatT>>();
    constexpr auto x0 = x(0).value();
    constexpr auto x1 = x(1).value();

    using Space = Matrixd<1, 1>;
    constexpr auto abs_x0 = Abs<AbsMeta<Space>, decltype(x0)>(x0);
    constexpr auto abs_x1 = Abs<AbsMeta<Space>, decltype(x1)>(x1);
    constexpr auto sum =
        Sum<SumConvexMeta<MatT, Space>, decltype(abs_x0), decltype(abs_x1)>(
            abs_x0, abs_x1);
    STATIC_REQUIRE(std::tuple_size<decltype(sum.children)>::value == 2);
  }
#endif
}
