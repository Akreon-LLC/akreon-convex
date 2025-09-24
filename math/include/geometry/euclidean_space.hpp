/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "../linalg/inner_product_space.hpp"

namespace geometry {

template <typename T>
  requires(linalg::FieldLike<T>)
using Point2d = linalg::Vector<T, 2>;

template <typename T>
  requires(linalg::FieldLike<T>)
using Point3d = linalg::Vector<T, 3>;

}  // namespace geometry
