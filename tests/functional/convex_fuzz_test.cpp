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
#include <iostream>
#include <random>

#include "math.hpp"

#define TRIALS 10000

using namespace linalg;
using namespace convex;

int main() {
  static constexpr size_t N = 100;
  using Space = Matrixd<1, N>;
  constexpr auto l1 = NormFn<Space, L1<Vectord<N>>>();
  constexpr auto l1_eval = l1.eval(Vectord<N>{});
  constexpr auto l2 = NormFn<Space, L2<Vectord<N>>>();
  constexpr auto l2_eval = l2.eval(Vectord<N>{});
  return l1_eval.value() + l2_eval.value();
}
