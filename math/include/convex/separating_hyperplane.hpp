/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "convex_set.hpp"
#include "hyperplane.hpp"

namespace convex {

// (dakre) TODO create this function once convex optimization lib is established
#if 0
  /**
   * @brief form separating hyperplane between two convex sets
   *
   * 1. establish that sets live in the same vector space
   * 2. determine if A and B are disjoint
   * 3. find pts (x, y) from min ||x - y||_2^2 s.t. x \in A and y \in B
   * 4. form line segment from two points
   * 5. bisect line segment to determine b offset
   * 6. form hyperplane a^T x - b = 0... a = [x, y] and b from above
   *
   * @param ConvexSetA 
   * @param ConvexSetB
   *
   * @return optinal Hyperplane that separates the two sets
   */
template <ConvexSetLike ConvexSetA, ConvexSetLike ConvexSetB>
constexpr Hyperplane<Space> separating_hyperplane(const ConvexSetA& A,
                                                  const ConvexSetB& B,
                                                  T tol = 1 - 8) const {
}
#endif

}  // namespace convex
