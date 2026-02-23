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

#include "math.hpp"

using namespace linalg;
using namespace convex;

int main() {
  static constexpr size_t N = 100;
  using Dom = Matrixd<1, N>;
  using Codom = Matrixd<1, 1>;
  constexpr auto var = Variable<FreeVarMeta<Dom>>();
  constexpr auto prob = make_problem(var);

  constexpr auto norm =
      Norm<NormMeta<Dom, Codom>, decltype(var), L2<Vectord<N>>>(var);
  constexpr auto prob2 = prob.minimize(norm);
  return prob2.validate() == true;
}
