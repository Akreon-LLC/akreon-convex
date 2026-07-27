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

#include "linalg/scalar/scalar_concepts.hpp"

namespace linalg {

template <Field T>
inline constexpr T sqrt2_v{T(1.414213562373095L)};

template <Field T>
inline constexpr T inv_sqrt2_v{T(1) / sqrt2_v<T>};

template <Field T>
inline constexpr T ln2_v{T(0.693147180559945309417232121458176568L)};

} // namespace linalg
