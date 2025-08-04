/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#include <algorithm>
#include <catch2/catch_test_macros.hpp>

#include "math.hpp"

using namespace topology;
using namespace algebra;
using namespace spaces;

template <typename T>
concept DefaultConstructible = requires { T{}; };

TEST_CASE("Test euclidean space", "[E]") {
  using OSD2 = OpenSet<double, 2>;
  using Derived = Topology<OSD2, 2>;

  STATIC_REQUIRE(DefaultConstructible<EuclideanSpace<Derived>>);
}
