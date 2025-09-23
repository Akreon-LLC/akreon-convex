/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "affine_map.hpp"

namespace convex {

template <typename T>
  requires(linalg::FieldLike<T> && !linalg::is_complex_v<T>)
struct Rotation2d {
  constexpr auto operator()(const T& phi) const {
    return linalg::Matrix<T, 2, 2>{
      linalg::Vector<T, 2>{linalg::constexpr_cos(phi),
                     -1 * linalg::constexpr_sin(phi)},
        linalg::Vector<T, 2>{linalg::constexpr_sin(phi), linalg::constexpr_cos(phi)}};
  }
};

#if 0
template <typename T>
  requires(linalg::FieldLike<T> && !linalg::is_complex_v<T>)
class AffineTransform2d : public AffineMap2d<T> {
};
#endif

}  // namespace convex
