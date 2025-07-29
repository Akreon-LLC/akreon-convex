/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#include <algorithm>
#include <catch2/catch_test_macros.hpp>

#include "math.hpp"

using namespace topology;
using namespace algebra;
using namespace geometry;

TEST_CASE("BLAH", "[blah]") {
  REQUIRE(true);
}

#if 0

TEST_CASE("Test topological manifold construction", "[topology]") {
  STATIC_REQUIRE(!TopologicalManifold<int>);
  STATIC_REQUIRE(!TopologicalManifold<Set<double>>);
  STATIC_REQUIRE(TopologicalManifold<double>);
}

TEST_CASE("Test construction of smooth manifolds", "[smooth_manifold]") {
  using M = Rn;
  // Forward map: M -> Rn
  auto phi = [](const M& x) -> Rn { return x; };

  // Inverse map: Rn -> M
  auto inv_phi = [](const Rn& y) -> M { return y; };

  SECTION("Chart") {
    STATIC_REQUIRE(ValidChart<M, Rn, decltype(phi), decltype(inv_phi)>);

    Chart<M, Rn, decltype(phi), decltype(inv_phi)> chart{
        .U = {1.f, 2.f}, .phi = phi, .inv_phi = inv_phi};
    REQUIRE(is_homeomorphism(chart));
  }

  SECTION("Atlas") {
    STATIC_REQUIRE(ValidAtlas<Td, Rn, decltype(phi), decltype(inv_phi)>);
    STATIC_REQUIRE(SmoothAtlas<Td, Rn, decltype(phi), decltype(inv_phi)>);

    // TODO (dakre) finish this test cast
    Atlas<Td, Rn, decltype(phi), decltype(inv_phi)> atlas;
    Topology<Td> tau;

    REQUIRE(!atlas.covers(tau));
    REQUIRE(atlas.is_compatible());
  }

  SmoothManifold<Td, Rn, decltype(phi), decltype(inv_phi)> smooth_manifold;
  SECTION("Manifold") { REQUIRE_NOTHROW(smooth_manifold); }

  // TODO (dakre) finish this test cast
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
    STATIC_REQUIRE(
        std::is_same_v<decltype(to_euclidean_space(vec)),
                       MetricSpace<Eigen::Matrix<double, 3, 1>,
                                   L2<Eigen::Matrix<double, 3, 1>>>>);
  }
}

#endif
