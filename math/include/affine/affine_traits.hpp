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

}  // namespace affine
