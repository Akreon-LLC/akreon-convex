/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#include "math.hpp"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>

using namespace topology;
using namespace chart;
using namespace atlas;
using namespace smooth_manifold;
using namespace smooth_map;
using namespace vector_space;

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

  auto ext_X = tau.exterior(X);
  REQUIRE(ext_X.size() == 0);

  auto cl_X = tau.closure(X);
  REQUIRE(cl_X.size() == 3);

  auto bd_X = tau.boundary(X);
  REQUIRE(bd_X.size() == 2);
}

TEST_CASE("Test metric space construction", "[metric_space]") {
  auto d = [](const double& x, const double& y) { return std::abs(x - y); };
  MetricSpace<double> ms{.tau = Topology<double>(), .d = d};
  REQUIRE_NOTHROW(ms);

  STATIC_REQUIRE(EuclideanType<double>);
  STATIC_REQUIRE(!EuclideanType<int>);
}

TEST_CASE("Test topological manifold construction", "[topology]") {
  STATIC_REQUIRE(!TopologicalManifold<int>);
  STATIC_REQUIRE(TopologicalManifold<double>);
}

TEST_CASE("Test construction of smooth manifolds", "[smooth_manifold]") {
  using Td = double;               // manifold points
  using Rn = std::vector<double>;  // points \in E

  // Forward map: double -> vector<double>
  auto phi = [](Td x) -> Rn { return {x}; };

  // Inverse map: vector<double> -> double
  auto inv_phi = [](const Rn& y) -> Td { return y[0]; };

  SECTION("Chart") {
    STATIC_REQUIRE(ValidChart<Td, Rn, decltype(phi), decltype(inv_phi)>);

    Chart<Td, Rn, decltype(phi), decltype(inv_phi)> chart{
        .U = {1.f, 2.f}, .phi = phi, .inv_phi = inv_phi};
    REQUIRE(is_homeomorphism(chart));
  }

  SECTION("Atlas") {
    STATIC_REQUIRE(ValidAtlas<Td, Rn, decltype(phi), decltype(inv_phi)>);
    STATIC_REQUIRE(SmoothAtlas<Td, Rn, decltype(phi), decltype(inv_phi)>);
  }

  SmoothManifold<Td, Rn, decltype(phi), decltype(inv_phi)> smooth_manifold;
  SECTION("Manifold") { REQUIRE_NOTHROW(smooth_manifold); }

  // TODO write test for smooth map
  SECTION("Map") {
    STATIC_REQUIRE(
        ValidSmoothMap<Td, Td, Rn, Rn, decltype(phi), decltype(inv_phi),
                       decltype(phi), decltype(inv_phi)>);
  }
}

TEST_CASE("Test construction of Rn space", "[vector_space]") {
  STATIC_REQUIRE(RnSpace<Eigen::Matrix<double, 3, 1>>);
  STATIC_REQUIRE(RnSpace<Eigen::Matrix<std::complex<double>, 3, 1>>);

  SECTION("Vector space to Metric space") {
    Eigen::Matrix<double, 3, 1> vec;
    STATIC_REQUIRE(std::is_same_v<decltype(to_metric_space(vec)),
                                  MetricSpace<Eigen::Matrix<double, 3, 1>>>);
  }
}
