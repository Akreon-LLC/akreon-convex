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
  using Space = Matrixd<1, 2>;
  constexpr auto norm = NormFn<Space, L1<Vectord<2>>>();
  constexpr auto norm_eval = norm.eval(Vectord<2>{});
  return norm_eval.value();
}
