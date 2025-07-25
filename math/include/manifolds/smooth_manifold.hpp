/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "atlas.hpp"
#include "topology.hpp"

namespace smooth_manifold {

template <typename T, typename Rn, typename MapToRn, typename MapFromRn>
  requires topology::TopologicalManifold<T> &&
           atlas::SmoothAtlas<T, Rn, MapToRn, MapFromRn>
struct SmoothManifold {
  topology::Topology<T> M;
  atlas::Atlas<T, Rn, MapToRn, MapFromRn> atlas;
};

}  // namespace smooth_manifold
