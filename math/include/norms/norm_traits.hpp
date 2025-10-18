/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

namespace linalg {

/// @brief forward declaration for L2 norm
template <FieldLike T, size_t N>
struct L2;

/// @brief forward declaration for L1 norm
template <FieldLike T, size_t N>
struct L1;

/// @brief forward declaration for Linf norm
template <FieldLike T, size_t N>
struct Linf;

/// @brief declaring norm traits
template <class Norm>
struct norm_traits : std::false_type {};

/// @brief declaring norm type traits - L2
template <FieldLike T, size_t N>
struct norm_traits<L2<T, N>> : std::true_type {
  using ScalarT = scalar_type_t<T>;
  static constexpr size_t Size = N;
};

/// @brief declaring norm type traits - L1
template <FieldLike T, size_t N>
struct norm_traits<L1<T, N>> : std::true_type {
  using ScalarT = scalar_type_t<T>;
  static constexpr size_t Size = N;
};

/// @brief declaring norm type traits - Linf
template <FieldLike T, size_t N>
struct norm_traits<Linf<T, N>> : std::true_type {
  using ScalarT = scalar_type_t<T>;
  static constexpr size_t Size = N;
};

/// @brief alias for extracting scalar type from norm
template <class Norm>
using norm_type_t = typename norm_traits<Norm>::ScalarT;

/// @brief helper for extracting size of norm
template <class Norm>
inline constexpr size_t norm_size_v = norm_traits<Norm>::Size;

/// @brief helper for is norm type
template <class Norm>
inline constexpr bool is_norm_v = norm_traits<Norm>{};

/// @brief concept for norm like
template <class Norm>
concept NormLike = is_norm_v<Norm>;

}  // namespace linalg
