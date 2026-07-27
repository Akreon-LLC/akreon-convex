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

#include "scalar_traits.hpp"

namespace linalg {

template <typename T>
concept Scalar = requires {
  typename scalar_traits_t<T>::scalar_type;
  typename scalar_traits_t<T>::real_type;
};

template <typename T>
concept Field = Scalar<T> && requires(const remove_cvref_t<T>& a,
                                      const remove_cvref_t<T>& b) {
  { a + b } -> std::convertible_to<remove_cvref_t<T>>;
  { a - b } -> std::convertible_to<remove_cvref_t<T>>;
  { a * b } -> std::convertible_to<remove_cvref_t<T>>;
  { a / b } -> std::convertible_to<remove_cvref_t<T>>;
};

template <typename T>
concept Real = Field<T> && is_real_v<T>;

template <typename T>
concept Complex = Field<T> && is_complex_v<T>;

} // namespace linalg
