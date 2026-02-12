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

#include "affine/affine_traits.hpp"
#include "convex_traits.hpp"

namespace conic {

/// @brief cone base class
template <linalg::VectorLike Vec, typename Derived>
struct Cone;

/// @brief forward decl for zero order cone
template <linalg::VectorLike Vec>
struct ZeroOrderCone;

/// @brief forward decl for nonneg orthant cone
template <linalg::VectorLike Vec>
struct NonnegOrthantCone;

/// @brief forward decl for soc
template <linalg::VectorLike Vec>
struct SecondOrderCone;

/// @brief forward decl for psd
template <linalg::VectorLike Vec, size_t N>
struct PositiveSemidefiniteCone;

/// @brief forward decl for exp
template <linalg::VectorLike Vec>
struct ExpCone;

/// @brief forward decl for power
template <linalg::VectorLike Vec>
struct PowerCone;

/// @brief forward decl for rsoc
template <linalg::VectorLike Vec>
struct RotatedSecondOrderCone;

/// @brief forward decl for log det cone
template <linalg::VectorLike Vec>
struct LogDetCone;

/// @brief cone traits
template <typename Cone>
struct cone_traits;

/// @brief cone traits derived for zoc
template <linalg::VectorLike Vec>
struct cone_traits<ZeroOrderCone<Vec>> : std::true_type {
  using SlackT = Vec;
  using T = linalg::vec_type_t<Vec>;
  static constexpr size_t N = linalg::vec_size_v<Vec>;
};

/// @brief cone traits derived for rplus
template <linalg::VectorLike Vec>
struct cone_traits<NonnegOrthantCone<Vec>> : std::false_type {
  using SlackT = Vec;
  using T = linalg::vec_type_t<Vec>;
  static constexpr size_t N = linalg::vec_size_v<Vec>;
};

/// @brief cone traits derived for soc
template <linalg::VectorLike Vec>
struct cone_traits<SecondOrderCone<Vec>> : std::false_type {
  using SlackT = Vec;
  using T = linalg::vec_type_t<Vec>;
  static constexpr size_t N = linalg::vec_size_v<Vec>;
};

/// @brief cone traits derived for psd
template <linalg::VectorLike Vec, size_t Dim>
struct cone_traits<PositiveSemidefiniteCone<Vec, Dim>> : std::false_type {
  using SlackT = Vec;
  using T = linalg::vec_type_t<Vec>;
  static constexpr size_t N = linalg::vec_size_v<Vec>;
};

/// @brief cone traits derived for exp
template <linalg::VectorLike Vec>
struct cone_traits<ExpCone<Vec>> : std::false_type {
  using SlackT = Vec;
  using T = linalg::vec_type_t<Vec>;
  static constexpr size_t N = linalg::vec_size_v<Vec>;
};

/// @brief cone traits derived for power cone
template <linalg::VectorLike Vec>
struct cone_traits<PowerCone<Vec>> : std::false_type {
  using SlackT = Vec;
  using T = linalg::vec_type_t<Vec>;
  static constexpr size_t N = linalg::vec_size_v<Vec>;
};

/// @brief cone traits derived for rsoc
template <linalg::VectorLike Vec>
struct cone_traits<RotatedSecondOrderCone<Vec>> : std::false_type {
  using SlackT = Vec;
  using T = linalg::vec_type_t<Vec>;
  static constexpr size_t N = linalg::vec_size_v<Vec>;
};

/// @brief alias for scalar type in cone
template <class Cone>
using cone_scalar_t = typename cone_traits<linalg::remove_cvref_t<Cone>>::T;

/// @brief alias for slack type in cone
template <class Cone>
using cone_slack_t = typename cone_traits<linalg::remove_cvref_t<Cone>>::SlackT;

/// @brief helper to get N from cone
template <class Cone>
inline constexpr size_t cone_dim_t =
    cone_traits<linalg::remove_cvref_t<Cone>>::N;

/// @brief helper for checking if its zero order cone
template <class Cone>
inline constexpr bool is_zoc_v = cone_traits<linalg::remove_cvref_t<Cone>>{};

/// @brief cone block forward decl
template <affine::AffineMapLike Map, typename... Cs>
struct ConeBlock;

/// @brief cone block traits
template <typename ConeBlock>
struct cone_block_traits;

/// @brief cone clobk type traits
template <affine::AffineMapLike Map, typename... Cs>
struct cone_block_traits<ConeBlock<Map, Cs...>> {
  using FMap = Map;
  static constexpr size_t NumCones = sizeof...(Cs);
};

/// @brief alias for extracting F map from cone block
template <typename CB>
using cone_block_map_t =
    typename cone_block_traits<linalg::remove_cvref_t<CB>>::FMap;

/// @brief helper to extract number of cones from cone block
template <typename CB>
inline constexpr size_t cone_block_num_cones_v =
    cone_block_traits<linalg::remove_cvref_t<CB>>::NumCones;

/// @brief concept for cone block like used in canonical form
template <class Cblk>
concept ConeBlockLike = requires(const Cblk& cblk) {
  { cblk.map };
  { cblk.cones };
};

}  // namespace conic
