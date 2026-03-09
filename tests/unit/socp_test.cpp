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
using namespace config;

TEST_CASE("Minimal SOCP 2 Vars", "[socp_test]") {
  using Space = Matrixd<1, 2>;
  using Singleton = Matrixd<1, 1>;
  constexpr auto x = Variable<VarMeta<Space>>();
  constexpr auto x_trans = x.add(Vectord<2>{-1.0, 2.0});
  constexpr auto prob_init = make_problem(x_trans);
  constexpr auto norm =
      Norm<NormMeta<Space, Singleton>, decltype(x_trans), L2<Vectord<2>>>(
          x_trans);
  constexpr auto square = Square<SquareMeta<Singleton>, decltype(norm)>(norm);
  constexpr auto half_square = square.mul<0.5>();
  constexpr auto inv_sqrt2 = 1.0 / constexpr_sqrt_safe(2.0);

  constexpr auto F_mat = mat_traits<Matrixd<6, 4>>();
  constexpr auto F_exp =
      F_mat(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -inv_sqrt2, 0.0, 0.0, 0.0,
            inv_sqrt2);
  constexpr Vectord<6> g_exp{-1.0, 2.0, 0.0, 0.0, 0.5 * inv_sqrt2,
                             0.5 * inv_sqrt2};

  SECTION("Unconstrained SOCP") {
    constexpr auto prob_min = prob_init.minimize(half_square);

    STATIC_REQUIRE(is_problem_set_v<decltype(prob_min)>);
    STATIC_REQUIRE(
        std::tuple_size<decltype(prob_min.constraint_cones())>::value == 2);

    constexpr auto F = prob_min.constraint_map().linear_part();
    STATIC_REQUIRE(mat_rows_v<decltype(F)> == 6);
    STATIC_REQUIRE(mat_cols_v<decltype(F)> == 4);
    STATIC_REQUIRE(all_equal(F, F_exp));

    constexpr auto g = prob_min.constraint_map().translation();
    STATIC_REQUIRE(vec_size_v<decltype(g)> == 6);
    STATIC_REQUIRE(g == g_exp);

    constexpr auto c = prob_min.linear_objective();
    STATIC_REQUIRE(c == Vectord<4>{0.0, 0.0, 0.0, 0.5});

    constexpr PrimalDualIPM solver(prob_min);
    constexpr auto res = solver.res;
    constexpr auto trace = solver.trace;

    log(res, trace);
  }

#if 0
  SECTION("L2 Ball Constrained SOCP") {
    constexpr auto r = 1.0;
    constexpr auto ball =
        NormBall<NormBallMeta<Space>, decltype(x), NormBallType::L2>(
            Vectord<2>{}, r, x);
    constexpr auto prob = prob_init.in_set(ball);
    constexpr auto prob_min = prob.minimize(half_square);

    STATIC_REQUIRE(is_problem_set_v<decltype(prob_min)>);

    constexpr auto c = prob_min.linear_objective();
    STATIC_REQUIRE(c == Vectord<4>{0.0, 0.0, 0.0, 0.5});

    constexpr PrimalDualIPM solver(prob_min);
    constexpr auto res = solver.res;
    constexpr auto trace = solver.trace;
    log(res, trace);
  }
#endif
}
