/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#include "topology/topology.hpp"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>

#include "algebra/field.hpp"
#include "topology/metric_space.hpp"

using namespace topology;
using namespace algebra;

TEST_CASE("Test topologcal space construction", "[topology]") {
  using T = Set<int>;
  SetTopology<T> tau;
  tau.open_sets = {{}, {1}, {1, 2}, {1, 2, 3}};

  T X = {1, 3};

  SECTION("Static checks") {
    STATIC_REQUIRE(TopologicalSpace<SetTopology<T>, T>);
  }

  SECTION("Property checks") {
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

    REQUIRE(tau.is_closed_under_union());

    REQUIRE(tau.is_closed_under_finite_intersection());

    // topology above formed by open sets is not disjoint
    REQUIRE(!tau.is_hausdorff());

    REQUIRE(tau.is_second_countable());
  }

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
}

TEST_CASE("Test field types", "[field]") {
  STATIC_REQUIRE(FloatingPoint<double>);
  STATIC_REQUIRE(FloatingPoint<std::vector<double>>);
  STATIC_REQUIRE(FloatingPoint<Eigen::Vector3d>);
  STATIC_REQUIRE(FloatingPoint<Eigen::Matrix3d>);
  STATIC_REQUIRE(FloatingPoint<std::vector<Eigen::Matrix3d>>);
}
