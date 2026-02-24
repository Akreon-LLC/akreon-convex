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

TEST_CASE("Minimal LP 2 Vars", "[pd_ipm_test]") {
  using Space = Matrixd<1, 2>;
  using Codom = Matrixd<1, 3>;
  constexpr auto x = Variable<NonnegVarMeta<Space>>();
  constexpr auto prob_init = make_problem(x);

  constexpr auto A_mat = mat_traits<Matrixd<3, 2>>();
  constexpr auto A = A_mat(1.0, 1.0, 2.0, 1.0, -1.0, 2.0);
  constexpr auto b = Vectord<3>{4.0, 5.0, 2.0};
  constexpr Polyhedron<PolyhedronMeta<Space, Codom>, decltype(x)> poly(A, b, x);
  constexpr auto prob_constr = prob_init.in_set(poly);

  constexpr auto F_mat = mat_traits<Matrixd<5, 2>>();
  constexpr auto F_exp =
      F_mat(1.0, 0.0, 0.0, 1.0, -1.0, -1.0, -2.0, -1.0, 1.0, -2.0);

  constexpr auto g_exp = Vectord<5>{0.0, 0.0, 4.0, 5.0, 2.0};

  SECTION("Min x0") {
    constexpr auto prob_min = prob_constr.minimize(x(0).value());
    STATIC_REQUIRE(is_problem_set_v<decltype(prob_min)>);
    STATIC_REQUIRE(
        std::tuple_size<decltype(prob_min.constraint_cones())>::value == 2);
    constexpr auto F = prob_min.constraint_map().linear_part();
    STATIC_REQUIRE(mat_rows_v<decltype(F)> == 5);
    STATIC_REQUIRE(mat_cols_v<decltype(F)> == 2);

    STATIC_REQUIRE(all_equal(F_exp, F));

    constexpr auto g = prob_min.constraint_map().translation();
    STATIC_REQUIRE(g == g_exp);

    constexpr auto c = prob_min.linear_objective();
    STATIC_REQUIRE(c == Vectord<2>{1.0, 0.0});

    PrimalDualIPM solver(prob_min);
    auto res = solver.res;
    auto trace = solver.trace;
#if 0
    STATIC_REQUIRE(constexpr_abs(res.z(0)) < SolverTraits<double>::DUALITY_GAP);
    STATIC_REQUIRE(res.tag == SolverResult::OPTIMAL);
    STATIC_REQUIRE(res.num_iter > 0 &&
                   res.num_iter < SolverTraits<double>::STEPS);
    STATIC_REQUIRE(trace[res.num_iter - 1].flags == 0x4);
#endif

    char buf[16384];
    trace_to_cstr(res.num_iter, trace, buf, sizeof(buf));
    std::cout << buf << std::endl;
    std::cout << res.z(0) << " " << res.z(1) << std::endl;
  }

  SECTION("Min x1") {
    constexpr auto prob_min = prob_constr.minimize(x(1).value());
    STATIC_REQUIRE(is_problem_set_v<decltype(prob_min)>);
    STATIC_REQUIRE(
        std::tuple_size<decltype(prob_min.constraint_cones())>::value == 2);
    constexpr auto F = prob_min.constraint_map().linear_part();
    STATIC_REQUIRE(mat_rows_v<decltype(F)> == 5);
    STATIC_REQUIRE(mat_cols_v<decltype(F)> == 2);

    STATIC_REQUIRE(all_equal(F_exp, F));

    constexpr auto g = prob_min.constraint_map().translation();
    STATIC_REQUIRE(g == g_exp);

    constexpr auto c = prob_min.linear_objective();
    STATIC_REQUIRE(c == Vectord<2>{0.0, 1.0});

    PrimalDualIPM solver(prob_min);
    auto res = solver.res;
    auto trace = solver.trace;
#if 0
    STATIC_REQUIRE(constexpr_abs(res.z(1)) < SolverTraits<double>::DUALITY_GAP);
    STATIC_REQUIRE(res.tag == SolverResult::OPTIMAL);
    STATIC_REQUIRE(res.num_iter > 0 &&
                   res.num_iter < SolverTraits<double>::STEPS);
    STATIC_REQUIRE(trace[res.num_iter - 1].flags == 0x4);
#endif

    char buf[16384];
    trace_to_cstr(res.num_iter, trace, buf, sizeof(buf));
    std::cout << buf << std::endl;
    std::cout << res.z(0) << " " << res.z(1) << std::endl;
  }

  SECTION("Min sum(x.split())") {
    static constexpr auto x_tup = x.split();
    constexpr auto x_sum =
        Sum<SumMeta<Space, Matrixd<1, 1>>, decltype(std::get<0>(x_tup)),
            decltype(std::get<1>(x_tup))>(std::get<0>(x_tup),
                                          std::get<1>(x_tup));

    constexpr auto prob_min = prob_constr.minimize(x_sum);
    STATIC_REQUIRE(is_problem_set_v<decltype(prob_min)>);
    STATIC_REQUIRE(
        std::tuple_size<decltype(prob_min.constraint_cones())>::value == 2);
    constexpr auto F = prob_min.constraint_map().linear_part();
    STATIC_REQUIRE(mat_rows_v<decltype(F)> == 5);
    STATIC_REQUIRE(mat_cols_v<decltype(F)> == 2);

    STATIC_REQUIRE(all_equal(F_exp, F));

    constexpr auto g = prob_min.constraint_map().translation();
    STATIC_REQUIRE(g == g_exp);

    constexpr auto c = prob_min.linear_objective();
    STATIC_REQUIRE(c == Vectord<2>{1.0, 1.0});

    const PrimalDualIPM solver(prob_min);
    const auto res = solver.res;
    const auto trace = solver.trace;

    char buf[64000];
    trace_to_cstr(res.num_iter, trace, buf, sizeof(buf));
    std::cout << buf << std::endl;
    std::cout << res.z(0) << " " << res.z(1) << std::endl;
  }
}
