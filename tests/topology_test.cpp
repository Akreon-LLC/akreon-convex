/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#include <algorithm>
#include <catch2/catch_test_macros.hpp>

#include "math.hpp"

using namespace topology;
using namespace algebra;

TEST_CASE("Test open set properties", "[topology]") {
  using T = OpenSet<int, 2>;
  using U = OpenSet<int, 4>;

  constexpr T A{1, 2};
  constexpr T B{2, 3};

  SECTION("Union check") {
    constexpr auto C = set_union(A, B);
    constexpr auto D = A | B;
    STATIC_REQUIRE(C == D);
    STATIC_REQUIRE(C.size() == 3);
    STATIC_REQUIRE(C == U{1, 2, 3});
    STATIC_REQUIRE(C.is_open(5));
    STATIC_REQUIRE(!C.is_open(3));

    constexpr auto E = set_union<int, 2, 2, 3>(A, B);
    STATIC_REQUIRE(E.size() == 3);
  }

  SECTION("Intersection check") {
    constexpr auto C = set_intersection(A, B);
    constexpr auto D = A & B;
    STATIC_REQUIRE(C == D);
    STATIC_REQUIRE(C.size() == 1);
    STATIC_REQUIRE(C.count(2));
    STATIC_REQUIRE(C.is_open(3));
    STATIC_REQUIRE(!C.is_open(2));
  }

  SECTION("Difference check") {
    constexpr auto C = set_difference(A, B);
    constexpr auto D = A - B;
    STATIC_REQUIRE(C == D);
    STATIC_REQUIRE(C.size() == 2);
    STATIC_REQUIRE(C == U{1, 3});
    STATIC_REQUIRE(C.is_open(4));
    STATIC_REQUIRE(!C.is_open(3));
  }

  SECTION("Complement check") {
    constexpr U universe{1, 2, 3, 4};
    constexpr auto C = set_complement(universe, A);
    constexpr auto D = set_complement(universe, B);
    STATIC_REQUIRE(C.size() == 2);
    STATIC_REQUIRE(D.size() == 2);
    constexpr auto E = set_intersection(C, D);
    STATIC_REQUIRE(E.size() == 1);
    STATIC_REQUIRE(E.count(4));
    STATIC_REQUIRE(E.s_size == 6);
  }

  SECTION("Insert check") {
    constexpr auto test_insert = []() {
      U C;
      C.insert(1);
      return C.count(1);
    };
    STATIC_REQUIRE(test_insert());
  }
}

TEST_CASE("Test topologcal space construction", "[topology]") {
  using T = OpenSet<int, 2>;
  SECTION("Runtime checks") {
    Topology<T, 3> tau{T{}};

    tau.open_sets.insert(T{1, 2});
    REQUIRE(!tau.open_sets.insert(T{1, 2}));
    tau.open_sets.insert(T{3, 4});
    tau.open_sets.insert(T{1, 5});

    REQUIRE(!tau.open_sets.insert(T{99, 100}));

    REQUIRE(tau.open_sets.size() == 3);
    REQUIRE(tau.open_sets.is_open(6));
    REQUIRE(!tau.open_sets.is_open(1));
  }

  constexpr Topology<T, 3> tau{T{1, 2}, T{3, 4}, T{1, 5}};
  SECTION("Static checks") {
    STATIC_REQUIRE(TopologicalSpace<T, 4>);
    STATIC_REQUIRE(tau.open_sets.size() == 3);
  }

  SECTION("Property checks") {
    constexpr auto int_X = interior(tau, T{2, 3});
    STATIC_REQUIRE(int_X.size() == 0);

    constexpr auto int_X2 = interior(tau, T{1, 5});
    STATIC_REQUIRE(int_X2.size() == 2);

    constexpr auto ext_X = exterior(tau, T{2, 3});
    STATIC_REQUIRE(ext_X.size() == 2);
    STATIC_REQUIRE(ext_X == T{1, 5});

    constexpr auto cl_X = closure(tau, T{2, 3});
    STATIC_REQUIRE(cl_X.size() == 5);
    STATIC_REQUIRE(cl_X.count(1));
    STATIC_REQUIRE(cl_X.count(2));
    STATIC_REQUIRE(cl_X.count(3));
    STATIC_REQUIRE(cl_X.count(4));
    STATIC_REQUIRE(cl_X.count(5));

    constexpr auto bd_X = boundary(tau, T{2, 3});
    STATIC_REQUIRE(bd_X.size() == 5);
    STATIC_REQUIRE(bd_X.count(1));
    STATIC_REQUIRE(bd_X.count(2));
    STATIC_REQUIRE(bd_X.count(3));
    STATIC_REQUIRE(bd_X.count(4));
    STATIC_REQUIRE(bd_X.count(5));

    STATIC_REQUIRE(!is_closed_under_union(tau));

    STATIC_REQUIRE(!is_closed_under_finite_intersection(tau));

    // topology above formed by open sets is not disjoint
    STATIC_REQUIRE(!is_hausdorff(tau));

    STATIC_REQUIRE(is_second_countable(tau));
  }
}

TEST_CASE("Test metric space", "[topology]") {
  using T = OpenSet<double, 2>;
  constexpr Topology<T, 3> tau{T{}, T{3.f, 4.f}, T{1.f, 5.f}};

  auto d = [](const double& x, const double& y) { return (x - y) * (x - y); };

  STATIC_REQUIRE(MetricSpace<double, 3, decltype(d)>);
  STATIC_REQUIRE(distance_metric_is_valid_on_space(tau, d));
}
