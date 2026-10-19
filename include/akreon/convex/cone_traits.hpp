/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <tuple>

#include "akreon/linalg/affine/affine_traits.hpp"

namespace conic {

/// @brief cone type
enum struct ConeType : uint8_t { SYMMETRIC = 0, ASYMMETRIC };

/// @brief cone base class
template <linalg::VectorContainer Vec, typename Derived, ConeType Type>
struct Cone;

/// @brief forward decl for nonneg orthant cone
template <linalg::VectorContainer Vec>
struct NonnegOrthantCone;

/// @brief forward decl for soc
template <linalg::VectorContainer Vec>
struct SecondOrderCone;

/// @brief forward decl for psd
template <linalg::VectorContainer Vec, std::size_t N>
struct PositiveSemidefiniteCone;

/// @brief forward decl for exp
template <linalg::VectorContainer Vec>
struct ExpCone;

/// @brief forward decl for power
template <linalg::VectorContainer Vec>
struct PowerCone;

/// @brief Base cone traits.
template <typename Cone>
struct cone_traits;

/// @brief Cone traits for NonnegOrthantCone.
template <linalg::VectorContainer Vec>
struct cone_traits<NonnegOrthantCone<Vec>> : std::false_type {
  using SlackT = Vec;
  using T = linalg::vector_element_t<Vec>;
  static constexpr ConeType CT = ConeType::SYMMETRIC;
  static constexpr std::size_t N = linalg::vector_dim_v<Vec>;
};

/// @brief Cone traits for SecondOrderCone.
template <linalg::VectorContainer Vec>
struct cone_traits<SecondOrderCone<Vec>> : std::false_type {
  using SlackT = Vec;
  using T = linalg::vector_element_t<Vec>;
  static constexpr ConeType CT = ConeType::SYMMETRIC;
  static constexpr std::size_t N = linalg::vector_dim_v<Vec>;
};

/// @brief Cone traits for PositiveSemidefiniteCone.
template <linalg::VectorContainer Vec, std::size_t Dim>
struct cone_traits<PositiveSemidefiniteCone<Vec, Dim>> : std::false_type {
  using SlackT = Vec;
  using T = linalg::vector_element_t<Vec>;
  static constexpr ConeType CT = ConeType::SYMMETRIC;
  static constexpr std::size_t N = linalg::vector_dim_v<Vec>;
};

/// @brief Cone traits for ExpCone.
template <linalg::VectorContainer Vec>
struct cone_traits<ExpCone<Vec>> : std::false_type {
  using SlackT = Vec;
  using T = linalg::vector_element_t<Vec>;
  static constexpr ConeType CT = ConeType::ASYMMETRIC;
  static constexpr std::size_t N = linalg::vector_dim_v<Vec>;
};

/// @brief Cone traits for PowerCone.
template <linalg::VectorContainer Vec>
struct cone_traits<PowerCone<Vec>> : std::false_type {
  using SlackT = Vec;
  using T = linalg::vector_element_t<Vec>;
  static constexpr ConeType CT = ConeType::ASYMMETRIC;
  static constexpr std::size_t N = linalg::vector_dim_v<Vec>;
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
inline constexpr std::size_t cone_dim_t =
    cone_traits<linalg::remove_cvref_t<Cone>>::N;

/// @brief Base traits for tuples of cones.
template <typename ConeTuple>
struct cone_tuple_traits;

/// @brief Cone tuple traits specialization.
template <typename... Cs>
struct cone_tuple_traits<std::tuple<Cs...>> {
  static constexpr std::size_t NumCones = sizeof...(Cs);
  static constexpr std::size_t LocalConeDim =
      (std::size_t{0} + ... + cone_dim_t<Cs>);
};

/// @brief Number of cones in a fixed cone tuple.
template <typename ConeTuple>
inline constexpr std::size_t cone_tuple_num_cones_v =
    cone_tuple_traits<linalg::remove_cvref_t<ConeTuple>>::NumCones;

/// @brief Total cone coordinate dimension represented by a fixed cone tuple.
template <typename ConeTuple>
inline constexpr std::size_t cone_tuple_local_cone_dim_v =
    cone_tuple_traits<linalg::remove_cvref_t<ConeTuple>>::LocalConeDim;

/// @brief Compatibility alias for total local cone coordinate dimension.
template <typename ConeTuple>
inline constexpr std::size_t cone_tuple_dim_v =
    cone_tuple_local_cone_dim_v<ConeTuple>;

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
  using MapT = affine::affine_mat_t<Map>;
  using Cones = std::tuple<Cs...>;

  static constexpr std::size_t NumCones = sizeof...(Cs);
  static constexpr std::size_t LocalConeDim =
      cone_tuple_local_cone_dim_v<Cones>;
  static constexpr std::size_t MapRows = linalg::mat_rows_v<MapT>;
  static constexpr std::size_t MapCols = linalg::mat_cols_v<MapT>;
};

/// @brief alias for extracting F map from cone block
template <typename CB>
using cone_block_map_t =
    typename cone_block_traits<linalg::remove_cvref_t<CB>>::FMap;

/// @brief helper to extract number of cones from cone block
template <typename CB>
inline constexpr std::size_t cone_block_num_cones_v =
    cone_block_traits<linalg::remove_cvref_t<CB>>::NumCones;

/// @brief total coordinate dimension spanned by the cone tuple
template <typename CB>
inline constexpr std::size_t cone_block_local_cone_dim_v =
    cone_block_traits<linalg::remove_cvref_t<CB>>::LocalConeDim;

/// @brief Compatibility alias for local cone coordinate dimension.
template <typename CB>
inline constexpr std::size_t cone_block_cone_dim_v =
    cone_block_local_cone_dim_v<CB>;

/// @brief number of rows in the cone block affine map
template <typename CB>
inline constexpr std::size_t cone_block_map_rows_v =
    cone_block_traits<linalg::remove_cvref_t<CB>>::MapRows;

/// @brief number of columns in the cone block affine map
template <typename CB>
inline constexpr std::size_t cone_block_map_cols_v =
    cone_block_traits<linalg::remove_cvref_t<CB>>::MapCols;

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
  { builder.eq_map };
};

/// @brief forward declaration of cone block builder
template <linalg::RnSpace DecVarDom, affine::AffineMapLike CoordMap,
          conic::ConeBlockLike CB, affine::AffineMapLike EqMap,
          typename CanonicalTrace>
struct CanonicalFormBuilder;

/// @brief Base builder traits.
template <typename Builder>
struct builder_traits;

/// @brief builder type traits of cone block builder
template <linalg::RnSpace DecVarDom, affine::AffineMapLike CoordMap,
          conic::ConeBlockLike CB, affine::AffineMapLike EqMap,
          typename CanonicalTrace>
struct builder_traits<
    CanonicalFormBuilder<DecVarDom, CoordMap, CB, EqMap, CanonicalTrace>> {
  using Dom = DecVarDom;
  using Map = CoordMap;
  using CoordinateMap = CoordMap;
  using ConeBlock = CB;
  using EqualityMap = EqMap;
  using CanonicalTraceT = CanonicalTrace;

  using CoordMapT = affine::affine_mat_t<CoordMap>;
  using EqMapT = affine::affine_mat_t<EqMap>;

  static constexpr std::size_t DecDim = linalg::space_dim_v<DecVarDom>;
  static constexpr std::size_t CoordRows = linalg::mat_rows_v<CoordMapT>;
  static constexpr std::size_t CoordCols = linalg::mat_cols_v<CoordMapT>;
  static constexpr std::size_t CoordAuxDim =
      CoordCols > DecDim ? CoordCols - DecDim : 0;

  static constexpr std::size_t ConeRows = cone_block_map_rows_v<CB>;
  static constexpr std::size_t ConeCols = cone_block_map_cols_v<CB>;
  static constexpr std::size_t ConeAuxDim =
      ConeCols > DecDim ? ConeCols - DecDim : 0;
  static constexpr std::size_t ConeCount = cone_block_num_cones_v<CB>;
  static constexpr std::size_t LocalConeDim = cone_block_local_cone_dim_v<CB>;

  static constexpr std::size_t EqRows = linalg::mat_rows_v<EqMapT>;
  static constexpr std::size_t EqCols = linalg::mat_cols_v<EqMapT>;
  static constexpr std::size_t EqAuxDim = EqCols > DecDim ? EqCols - DecDim : 0;

  static constexpr std::size_t FullDim = ConeCols > EqCols ? ConeCols : EqCols;
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

/// @brief builder equality map
template <typename Builder>
using builder_eq_t =
    typename builder_traits<linalg::remove_cvref_t<Builder>>::EqualityMap;

/// @brief builder canonical trace
template <typename Builder>
using builder_canonical_trace_t =
    typename builder_traits<linalg::remove_cvref_t<Builder>>::CanonicalTraceT;

/// @brief original decision-variable dimension tracked by a builder
template <typename Builder>
inline constexpr std::size_t builder_dec_dim_v =
    builder_traits<linalg::remove_cvref_t<Builder>>::DecDim;

/// @brief current coordinate selector row count
template <typename Builder>
inline constexpr std::size_t builder_coord_rows_v =
    builder_traits<linalg::remove_cvref_t<Builder>>::CoordRows;

/// @brief current coordinate selector column count
template <typename Builder>
inline constexpr std::size_t builder_coord_cols_v =
    builder_traits<linalg::remove_cvref_t<Builder>>::CoordCols;

/// @brief auxiliary columns visible to the current coordinate selector
template <typename Builder>
inline constexpr std::size_t builder_coord_aux_dim_v =
    builder_traits<linalg::remove_cvref_t<Builder>>::CoordAuxDim;

/// @brief current cone affine-map row count
template <typename Builder>
inline constexpr std::size_t builder_cone_rows_v =
    builder_traits<linalg::remove_cvref_t<Builder>>::ConeRows;

/// @brief current cone affine-map column count
template <typename Builder>
inline constexpr std::size_t builder_cone_cols_v =
    builder_traits<linalg::remove_cvref_t<Builder>>::ConeCols;

/// @brief auxiliary columns visible to the current cone map
template <typename Builder>
inline constexpr std::size_t builder_cone_aux_dim_v =
    builder_traits<linalg::remove_cvref_t<Builder>>::ConeAuxDim;

/// @brief number of cones currently accumulated by a builder
template <typename Builder>
inline constexpr std::size_t builder_cone_count_v =
    builder_traits<linalg::remove_cvref_t<Builder>>::ConeCount;

/// @brief total local cone coordinate dimension currently realized by a builder
template <typename Builder>
inline constexpr std::size_t builder_local_cone_dim_v =
    builder_traits<linalg::remove_cvref_t<Builder>>::LocalConeDim;

/// @brief Compatibility alias for realized local cone coordinate dimension.
template <typename Builder>
inline constexpr std::size_t builder_cone_dim_v =
    builder_local_cone_dim_v<Builder>;

/// @brief current equality affine-map row count
template <typename Builder>
inline constexpr std::size_t builder_eq_rows_v =
    builder_traits<linalg::remove_cvref_t<Builder>>::EqRows;

/// @brief current equality affine-map column count
template <typename Builder>
inline constexpr std::size_t builder_eq_cols_v =
    builder_traits<linalg::remove_cvref_t<Builder>>::EqCols;

/// @brief auxiliary columns visible to the current equality map
template <typename Builder>
inline constexpr std::size_t builder_eq_aux_dim_v =
    builder_traits<linalg::remove_cvref_t<Builder>>::EqAuxDim;

/// @brief widest canonical domain currently visible to cone or equality rows
template <typename Builder>
inline constexpr std::size_t builder_full_dim_v =
    builder_traits<linalg::remove_cvref_t<Builder>>::FullDim;

} // namespace conic
