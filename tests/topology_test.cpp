/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#include "topology/topology.hpp"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>

using namespace topology;

TEST_CASE("Test topologcal space construction", "[topology]") {
  using T = int;
  Topology<T> tau;

  tau.open_sets = {{}, {1}, {2}, {1, 2}};

  Set<T> X = {1, 2};

  STATIC_REQUIRE(TopologicalSpace<int>);
  STATIC_REQUIRE(TopologicalSpace<unsigned>);
  STATIC_REQUIRE(TopologicalSpace<float>);
  STATIC_REQUIRE(TopologicalSpace<double>);
}

TEST_CASE("Test properties of topology", "[topology]") {
  Topology<int> tau;
  Set<int> X = {1, 3};
  tau.open_sets = {{}, {1}, {1, 2}, {1, 2, 3}};
  auto int_X = tau.interior(X);
  REQUIRE(int_X.size() == 1);
  REQUIRE(int_X == Set<int>{1});

  auto ext_X = tau.exterior(X);
  REQUIRE(ext_X.size() == 0);
  REQUIRE(ext_X == Set<int>{});

  auto cl_X = tau.closure(X);
  REQUIRE(cl_X.size() == 3);
  REQUIRE(cl_X == Set<int>{1, 2, 3});

  auto bd_X = tau.boundary(X);
  REQUIRE(bd_X.size() == 2);
  REQUIRE(bd_X == Set<int>{2, 3});
}

TEST_CASE("Test metric space construction", "[metric_space]") {
  SECTION("Norm is valid") {
    Topology<double> tau;
    auto d = [](const double& x, const double& y) { return std::abs(x - y); };

    MetricSpace<double, decltype(d)> ms{.tau = tau};
    REQUIRE(ms.distance_metric_is_valid());
  }

  SECTION("Euclidean checks") {
    STATIC_REQUIRE(EuclideanType<double>);
    STATIC_REQUIRE(!EuclideanType<int>);
  }
}

TEST_CASE("Test topological manifold construction", "[topology]") {
  STATIC_REQUIRE(!TopologicalManifold<int>);
  STATIC_REQUIRE(TopologicalManifold<double>);
}
