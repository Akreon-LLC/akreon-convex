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

#pragma once

#include "algebra/algebra_traits.hpp"

namespace affine {

/// @brief forward declaration of affine map
template <linalg::MatrixLike Mat>
class AffineMap;

/// @brief forward declaration of parametric affine set
template <linalg::MatrixLike Mat>
class ParametricAffineSet;

/// @brief meta type traits for affine map
template <class Mat>
struct affine_traits;

/// @brief derived type traits for affine map
template <linalg::MatrixLike Mat>
struct affine_traits<AffineMap<Mat>> {
  using AffineMat = Mat;
};

/// @brief derived type traits for affine map
template <linalg::MatrixLike Mat>
struct affine_traits<ParametricAffineSet<Mat>> {
  using AffineMat = Mat;
};

/// @brief alias to extract matrix type for affine map
template <class AffineMap>
using affine_mat_t =
    typename affine_traits<linalg::remove_cvref_t<AffineMap>>::AffineMat;

/// @brief concept for affine map
template <class Map>
concept AffineMapLike = requires(const Map& map) {
  { map.linear_part() };
  { map.translation() };
};

}  // namespace affine
