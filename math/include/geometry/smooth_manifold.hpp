/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "atlas.hpp"

namespace geometry {

template <typename T, typename Rn, typename MapToRn, typename MapFromRn>
  requires topology::TopologicalManifold<T> &&
           SmoothAtlas<T, Rn, MapToRn, MapFromRn>
struct SmoothManifold {
  topology::Topology<T> M;
  Atlas<T, Rn, MapToRn, MapFromRn> atlas;
};

}  // namespace geometry
