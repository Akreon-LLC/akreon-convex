/* Copyright © 2025 Akreon LLC. All rights reserved.
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

#include <concepts>

#include "linalg/matrix/matrix_concepts.hpp"
#include "linalg/scalar/scalar_concepts.hpp"
#include "linalg/vector/vector_concepts.hpp"
#include "coordinate_concepts.hpp"
#include "space_traits.hpp"

namespace linalg {

template <class Space>
concept VectorSpace =
    requires(const space_point_t<Space>& x, const space_point_t<Space>& y,
             const space_type_t<Space>& alpha) {
      { x + y } -> std::convertible_to<space_point_t<Space>>;
      { x - y } -> std::convertible_to<space_point_t<Space>>;
      { alpha * x } -> std::convertible_to<space_point_t<Space>>;
      { alpha * (x + y) } -> std::convertible_to<space_point_t<Space>>;
      { alpha * x + alpha * y } -> std::convertible_to<space_point_t<Space>>;
      {
        space_type_t<Space>(1) * x
      } -> std::convertible_to<space_point_t<Space>>;
    };

template <class Space>
concept RnSpace = VectorSpace<Space> && CoordinatizedSpace<Space> &&
                  Real<space_type_t<Space>>;

} // namespace linalg
