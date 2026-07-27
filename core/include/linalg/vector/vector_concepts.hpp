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

#include "linalg/scalar/scalar_concepts.hpp"
#include "vector_traits.hpp"

namespace linalg {

template <class Vec>
concept VectorContainer = requires {
  typename vector_traits_t<Vec>::element_type;
  { vector_dim_v<Vec> } -> std::convertible_to<std::size_t>;
};

template <class Vec>
concept FieldVector = VectorContainer<Vec> && Field<vector_element_t<Vec>>;

template <class Vec>
concept RealVector = FieldVector<Vec> && Real<vector_element_t<Vec>>;

template <class Vec>
concept ComplexVector = FieldVector<Vec> && Complex<vector_element_t<Vec>>;

template <class VecA, class VecB>
concept CompatibleVectors =
    VectorContainer<VecA> && VectorContainer<VecB> &&
    std::same_as<vector_element_t<VecA>, vector_element_t<VecB>> &&
    vector_dim_v<VecA> == vector_dim_v<VecB>;

} // namespace linalg
