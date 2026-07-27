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

namespace linalg {

/// @brief forward declaration for L2 norm
template <FieldVector Vec>
struct L2;

/// @brief forward declaration for L1 norm
template <FieldVector Vec>
struct L1;

/// @brief forward declaration for Linf norm
template <FieldVector Vec>
struct Linf;

/// @brief declaring norm traits
template <class Norm>
struct norm_traits : std::false_type {};

/// @brief declaring norm type traits - L2
template <FieldVector Vec>
struct norm_traits<L2<Vec>> : std::true_type {
  using T = vector_element_t<Vec>;
  static constexpr std::size_t Size = vector_dim_v<Vec>;
};

/// @brief declaring norm type traits - L1
template <FieldVector Vec>
struct norm_traits<L1<Vec>> : std::true_type {
  using T = vector_element_t<Vec>;
  static constexpr std::size_t Size = vector_dim_v<Vec>;
};

/// @brief declaring norm type traits - Linf
template <FieldVector Vec>
struct norm_traits<Linf<Vec>> : std::true_type {
  using T = vector_element_t<Vec>;
  static constexpr std::size_t Size = vector_dim_v<Vec>;
};

/// @brief alias for extracting scalar type from norm
template <class Norm>
using norm_type_t = typename norm_traits<remove_cvref_t<Norm>>::T;

/// @brief helper for extracting size of norm
template <class Norm>
inline constexpr std::size_t norm_size_v =
    norm_traits<remove_cvref_t<Norm>>::Size;

/// @brief helper for is norm type
template <class Norm>
inline constexpr bool is_norm_v = norm_traits<remove_cvref_t<Norm>>{};

/// @brief concept for norm like
template <class Norm>
concept NormLike = is_norm_v<Norm>;

} // namespace linalg
