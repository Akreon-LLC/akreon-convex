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
    constexpr U universe{1,2,3,4};
    constexpr auto C = set_complement(universe, A);
    constexpr auto D = set_complement(universe, B);
    STATIC_REQUIRE(C.size() == 2);
    STATIC_REQUIRE(D.size() == 2);
    constexpr auto E = set_intersection(C, D);
    STATIC_REQUIRE(E.size() == 1);
    STATIC_REQUIRE(E.count(4));
    STATIC_REQUIRE(E.s_size == 6);
  }
}

TEST_CASE("Test topologcal space construction", "[topology]") {
  using T = OpenSet<int, 2>;
  SetTopology<T, 3> tau;

  tau.open_sets.insert(T{1, 2});
  REQUIRE(!tau.open_sets.insert(T{1, 2}));
  tau.open_sets.insert(T{3, 4});
  tau.open_sets.insert(T{1, 5});

  REQUIRE(!tau.open_sets.insert(T{99, 100}));

  SECTION("Static checks") {
    STATIC_REQUIRE(TopologicalSpace<SetTopology<T, 4>, T>);
  }

  REQUIRE(tau.open_sets.is_open(6));
  REQUIRE(!tau.open_sets.is_open(1));

  SECTION("Property checks") {
    auto int_X = tau.interior(T{2, 3});
    REQUIRE(int_X.size() == 0);

    int_X = tau.interior(T{1, 5});
    REQUIRE(int_X.size() == 2);

    // auto ext_X = tau.exterior(T{2, 3});
    // REQUIRE(ext_X.size() == 0);

    // auto cl_X = tau.closure(X);
    // REQUIRE(cl_X.size() == 3);
    // REQUIRE(cl_X == Set<int>{1, 2, 3});

    // auto bd_X = tau.boundary(X);
    // REQUIRE(bd_X.size() == 2);
    // REQUIRE(bd_X == Set<int>{2, 3});

    // REQUIRE(tau.is_closed_under_union());

    // REQUIRE(tau.is_closed_under_finite_intersection());

    //// topology above formed by open sets is not disjoint
    // REQUIRE(!tau.is_hausdorff());

    // REQUIRE(tau.is_second_countable());
  }

#if 0

  SECTION("Metric space checks") {
    auto d = [](const int& x, const int& y) { return std::abs(x - y); };

    STATIC_REQUIRE(MetricSpace<SetTopology<T>, T, decltype(d)>);
    REQUIRE(distance_metric_is_valid_on_space(tau, d));

    using Td = Set<double>;
    SetTopology<Td> dtau;
    auto d2 = [](const double& x, const double& y) { return std::abs(x - y); };

    STATIC_REQUIRE(MetricSpace<decltype(dtau), Td, decltype(d2)>);
    REQUIRE(distance_metric_is_valid_on_space(dtau, d2));
  }
#endif
}
