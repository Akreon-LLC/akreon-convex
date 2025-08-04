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

  using VD5 = Vectord<5>;
  using Derived2 = Topology<VD5, 9>;
  STATIC_REQUIRE(DefaultConstructible<EuclideanSpace<Derived2>>);
}

TEST_CASE("Test linear system", "[y=Ax]") {
  constexpr auto x = Vectord<3>{3.f, -2.f, 1.f};
  constexpr auto A = Vector<Vectord<3>, 3>{x, x, x};
  constexpr auto y = A*x;
  STATIC_REQUIRE(y(0) == 14.f);
  STATIC_REQUIRE(y(1) == 14.f);
  STATIC_REQUIRE(y(2) == 14.f);
}
