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

/**
 * @brief Legacy vector-space expression contract for a registered space.
 *
 * The registered `Point` type must support addition, subtraction, and scalar
 * multiplication by the registered `value_type`. The distributive expressions
 * are checked for type compatibility. As with other C++ concepts, this verifies
 * syntax and result types rather than proving the vector-space axioms.
 *
 * New element-coordinate modeling should use `Space` and
 * `CoordinatizedSpace`; this concept remains for APIs expressed through the
 * legacy `space_point_t`/`space_type_t` vocabulary.
 */
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

/**
 * @brief Real, coordinatized space satisfying the vector-space expressions.
 *
 * This combines the legacy arithmetic contract with the explicit modern
 * coordinate conversion contract.
 */
template <class Space>
concept RnSpace = VectorSpace<Space> && CoordinatizedSpace<Space> &&
                  Real<space_type_t<Space>>;

} // namespace linalg
