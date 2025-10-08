/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "convex_cone.hpp"
#include "convex_set.hpp"

namespace convex {

template <ConvexSetLike ConvexSet, ConeLike Cone>
  requires(set_space_t<ConvexSet> == set_space_t<Cone>)
class ConicPerspective {};

}  // namespace convex
