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

namespace conic {

/// @brief cone type
enum struct ConeType : uint8_t { SYMMETRIC = 0, NONSYMMETRIC };

/// @brief cone base class
template <linalg::VectorLike Vec, typename Derived, ConeType Type>
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

/// @brief Base cone traits.
template <typename Cone>
struct cone_traits;

/// @brief Cone traits for ZeroOrderCone.
template <linalg::VectorLike Vec>
struct cone_traits<ZeroOrderCone<Vec>> : std::true_type {
  using SlackT = Vec;
  using T = linalg::vec_type_t<Vec>;
  static constexpr ConeType CT = ConeType::SYMMETRIC;
  static constexpr size_t N = linalg::vec_size_v<Vec>;
};

/// @brief Cone traits for NonnegOrthantCone.
template <linalg::VectorLike Vec>
struct cone_traits<NonnegOrthantCone<Vec>> : std::false_type {
  using SlackT = Vec;
  using T = linalg::vec_type_t<Vec>;
  static constexpr ConeType CT = ConeType::SYMMETRIC;
  static constexpr size_t N = linalg::vec_size_v<Vec>;
};

/// @brief Cone traits for SecondOrderCone.
template <linalg::VectorLike Vec>
struct cone_traits<SecondOrderCone<Vec>> : std::false_type {
  using SlackT = Vec;
  using T = linalg::vec_type_t<Vec>;
  static constexpr ConeType CT = ConeType::SYMMETRIC;
  static constexpr size_t N = linalg::vec_size_v<Vec>;
};

/// @brief Cone traits for PositiveSemidefiniteCone.
template <linalg::VectorLike Vec, size_t Dim>
struct cone_traits<PositiveSemidefiniteCone<Vec, Dim>> : std::false_type {
  using SlackT = Vec;
  using T = linalg::vec_type_t<Vec>;
  static constexpr ConeType CT = ConeType::SYMMETRIC;
  static constexpr size_t N = linalg::vec_size_v<Vec>;
};

/// @brief Cone traits for ExpCone.
template <linalg::VectorLike Vec>
struct cone_traits<ExpCone<Vec>> : std::false_type {
  using SlackT = Vec;
  using T = linalg::vec_type_t<Vec>;
  static constexpr ConeType CT = ConeType::NONSYMMETRIC;
  static constexpr size_t N = linalg::vec_size_v<Vec>;
};

/// @brief Cone traits for PowerCone.
template <linalg::VectorLike Vec>
struct cone_traits<PowerCone<Vec>> : std::false_type {
  using SlackT = Vec;
  using T = linalg::vec_type_t<Vec>;
  static constexpr ConeType CT = ConeType::NONSYMMETRIC;
  static constexpr size_t N = linalg::vec_size_v<Vec>;
};

/// @brief Cone traits for RotatedSecondOrderCone.
template <linalg::VectorLike Vec>
struct cone_traits<RotatedSecondOrderCone<Vec>> : std::false_type {
  using SlackT = Vec;
  using T = linalg::vec_type_t<Vec>;
  static constexpr ConeType CT = ConeType::SYMMETRIC;
  static constexpr size_t N = linalg::vec_size_v<Vec>;
};

/// @brief alias for scalar type in cone
template <class Cone>
using cone_scalar_t = typename cone_traits<linalg::remove_cvref_t<Cone>>::T;

/// @brief alias for slack type in cone
template <class Cone>
using cone_slack_t = typename cone_traits<linalg::remove_cvref_t<Cone>>::SlackT;

/// @brief helper to see if cone is symmetric
template <class Cone>
inline constexpr bool is_cone_symmetric_v =
    cone_traits<linalg::remove_cvref_t<Cone>>::CT == ConeType::SYMMETRIC;

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

/// @brief Base cone block traits.
template <typename ConeBlock>
struct cone_block_traits;

/// @brief Cone block traits specialization.
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

/// @brief Concept for canonicalization builder objects.
template <typename Builder>
concept BuilderLike = requires(const Builder& builder) {
  { builder.coord_map };
  { builder.cb };
};

/// @brief forward declaration of cone block builder
template <linalg::RnSpace DecVarDom, affine::AffineMapLike CoordMap,
          conic::ConeBlockLike CB>
struct ConeBlockBuilder;

/// @brief Base builder traits.
template <typename Builder>
struct builder_traits;

/// @brief builder type traits of cone block builder
template <linalg::RnSpace DecVarDom, affine::AffineMapLike CoordMap,
          conic::ConeBlockLike CB>
struct builder_traits<ConeBlockBuilder<DecVarDom, CoordMap, CB>> {
  using Dom = DecVarDom;
  using Map = CoordMap;
  using ConeBlock = CB;
};

/// @brief builder domain
template <typename Builder>
using builder_dom_t =
    typename builder_traits<linalg::remove_cvref_t<Builder>>::Dom;

/// @brief builder map
template <typename Builder>
using builder_map_t =
    typename builder_traits<linalg::remove_cvref_t<Builder>>::Map;

/// @brief builder cone block
template <typename Builder>
using builder_cb_t =
    typename builder_traits<linalg::remove_cvref_t<Builder>>::ConeBlock;

}  // namespace conic
