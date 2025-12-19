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

template <size_t N>
constexpr bool adj_mat_equal(const linalg::Matrix<bool, N, N>& adj_one,
                             const linalg::Matrix<bool, N, N>& adj_two) {
  for (size_t i = 0; i < N; ++i)
    for (size_t j = 0; j < N; ++j)
      if (adj_one(i)(j) != adj_two(i)(j)) return false;
  return true;
}

TEST_CASE("Test dag setup (simple)", "[convex_dag]") {
  using Space = Matrixd<1, 1>;
  constexpr auto abs_fn = AbsFn<Space>();

  SECTION("Test dag one atom") {
    using BoolMatT = Matrix<bool, 1, 1>;
    constexpr BoolMatT adj{};
    constexpr auto dag = DAG<decltype(abs_fn)>(adj, abs_fn);
    STATIC_REQUIRE(std::tuple_size<decltype(dag.atoms)>::value == 1);
    STATIC_REQUIRE(adj(0)(0) == dag.adj(0)(0));
    STATIC_REQUIRE(!has_cycle(dag));
  }

  SECTION("Test dag two atoms (unlinked)") {
    using BoolMatT = Matrix<bool, 2, 2>;
    constexpr BoolMatT adj{};
    constexpr auto dag = DAG(adj, abs_fn, abs_fn);
    STATIC_REQUIRE(std::tuple_size<decltype(dag.atoms)>::value == 2);
    STATIC_REQUIRE(adj_mat_equal(adj, dag.adj));
    STATIC_REQUIRE(!has_cycle(dag));
  }

  SECTION("Test dag two atoms (linked)") {
    using BoolMatT = Matrix<bool, 2, 2>;
    constexpr auto mat = mat_traits<BoolMatT>();
    constexpr auto adj = mat(0, 1, 0, 0);
    constexpr auto dag = DAG(adj, abs_fn, abs_fn);
    STATIC_REQUIRE(std::tuple_size<decltype(dag.atoms)>::value == 2);
    STATIC_REQUIRE(adj_mat_equal(adj, dag.adj));
    STATIC_REQUIRE(!has_cycle(dag));
  }

  SECTION("Test dag two atoms (linked w/ cycle)") {
    using BoolMatT = Matrix<bool, 2, 2>;
    constexpr auto mat = mat_traits<BoolMatT>();
    constexpr auto adj = mat(0, 1, 1, 0);
    constexpr auto dag = DAG(adj, abs_fn, abs_fn);
    STATIC_REQUIRE(std::tuple_size<decltype(dag.atoms)>::value == 2);
    STATIC_REQUIRE(adj_mat_equal(adj, dag.adj));
    STATIC_REQUIRE(has_cycle(dag));
  }

  SECTION("Test dag three atoms (linked and simple)") {
    using BoolMatT = Matrix<bool, 3, 3>;
    constexpr auto mat = mat_traits<BoolMatT>();
    constexpr auto adj = mat(0, 1, 1, 0, 0, 0, 0, 0, 0);
    constexpr auto dag = DAG(adj, abs_fn, abs_fn, abs_fn);
    STATIC_REQUIRE(std::tuple_size<decltype(dag.atoms)>::value == 3);
    STATIC_REQUIRE(adj_mat_equal(adj, dag.adj));
    STATIC_REQUIRE(!has_cycle(dag));
  }

  SECTION("Test dag three atoms (linked w/ cycle)") {
    using BoolMatT = Matrix<bool, 3, 3>;
    constexpr auto mat = mat_traits<BoolMatT>();
    constexpr auto adj = mat(0, 1, 1, 0, 0, 0, 1, 0, 0);
    constexpr auto dag = DAG(adj, abs_fn, abs_fn, abs_fn);
    STATIC_REQUIRE(std::tuple_size<decltype(dag.atoms)>::value == 3);
    STATIC_REQUIRE(adj_mat_equal(adj, dag.adj));
    STATIC_REQUIRE(has_cycle(dag));
  }

  SECTION("Test dag three atoms (linked no cycle)") {
    using BoolMatT = Matrix<bool, 3, 3>;
    constexpr auto mat = mat_traits<BoolMatT>();
    constexpr auto adj = mat(0, 1, 1, 0, 0, 0, 0, 1, 0);
    constexpr auto dag = DAG(adj, abs_fn, abs_fn, abs_fn);
    STATIC_REQUIRE(std::tuple_size<decltype(dag.atoms)>::value == 3);
    STATIC_REQUIRE(adj_mat_equal(adj, dag.adj));
    STATIC_REQUIRE(!has_cycle(dag));
  }
}
