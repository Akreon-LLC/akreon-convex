/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#include <algorithm>
#include <iostream>
#include <random>

#include "math.hpp"

#define TRIALS 10000

using namespace linalg;
using namespace convex;

int main() {
  using Space = Matrixd<1, 2>;
  constexpr auto norm = NormFn<Space, L1<Vectord<2>>>();
  constexpr auto norm_eval = norm.eval(Vectord<2>{});
  return norm_eval.value();
}
