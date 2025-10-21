/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "functions/convex_fn.hpp"
#include "sets/convex_set.hpp"
#include "sets/hyperplane.hpp"

namespace convex {

// (dakre) TODO create these function once convex optimization lib is
// established

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
  requires(std::same_as<set_space_t<ConvexSetA>, set_space_t<ConvexSetB>>)
constexpr std::optional<Hyperplane<set_space_t<ConvexSetA>>>
separating_hyperplane(const ConvexSetA& A, const ConvexSetB& B,
                      set_type_t<ConvexSetA> tol = 1e-8) {
  (void)A, (void)B, (void)tol;
  return std::nullopt;
}

/**
 * @brief form supporting hyperplane in direction u given a convex set
 *
 * x^* = arg max u^T x s.t. x \in C, h(u) = u^T x^*, H(u) = { x | u^T x = h(u) }
 *
 * @param ConvexSet
 * @param directional vector u spanning ambient space
 *
 * @return supporting hyperplane
 */
template <ConvexSetLike ConvexSet, linalg::VectorLike Vec>
  requires(set_dim_v<ConvexSet> == linalg::vec_size_v<Vec>)
constexpr std::optional<Hyperplane<set_space_t<ConvexSet>>>
supporting_hyperplane(const ConvexSet& A, const Vec& u,
                      set_type_t<ConvexSet> tol = 1e-8) {
  (void)A, (void)u, (void)tol;
  return std::nullopt;
}

}  // namespace convex
