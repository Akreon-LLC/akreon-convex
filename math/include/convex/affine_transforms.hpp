/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "affine_map.hpp"

namespace convex {

template <typename T>
  requires(FieldLike<T>)
using AffineMap2d = AffineMap<T, 2>;

template <typename T>
  requires(FieldLike<T>)
using AffineMap3d = AffineMap<T, 3>;

}  // namespace convex
