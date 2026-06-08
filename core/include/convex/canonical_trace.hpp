/*
 * Copyright © 2026 Akreon LLC. All rights reserved.
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

#include <cstddef>
#include <cstdint>
#include <limits>
#include <tuple>
#include <type_traits>

#include "convex/cone_traits.hpp"
#include "convex/convex_traits.hpp"

namespace convex {

/// @brief Record category in the structural canonicalization trace.
enum struct CanonicalRecordKind : uint8_t { NODE = 0, CONE = 1, EQUALITY = 2 };

/// @brief Cone family emitted into the global cone block.
enum struct CanonicalConeKind : uint8_t {
  NONNEG_ORTHANT = 0,
  SECOND_ORDER = 1,
  POSITIVE_SEMIDEFINITE = 2,
  EXP = 3,
  POWER = 4,
  UNKNOWN = 255
};

/// @brief Sentinel used when a canonical node does not have a parent node.
inline constexpr size_t canonical_no_parent =
    std::numeric_limits<size_t>::max();

/**
 * @brief Structural atom emitted during canonicalization.
 *
 * A node represents one canonicalization emission site, usually a convex atom
 * or set operation. Cone and equality records reference `node_id` so downstream
 * layers can group solver slices by the atom/set that produced them.
 */
struct CanonicalNodeRecord {
  size_t node_id{}; ///< Stable index of this trace node.

  /// Parent trace node in a future richer lineage DAG, or
  /// `canonical_no_parent` for root/flat emissions.
  size_t parent_id{canonical_no_parent};

  /// Canonicalization role that produced this node.
  AtomType atom_type{};

  /// Index of this node within its parent's children. Currently flat emission
  /// sites use zero; populated parent links can use this to preserve child
  /// ordering without relying on names.
  size_t child_index{};
};

/**
 * @brief Cone slice emitted by one canonical trace node.
 *
 * `cone_offset` and `cone_dim` locate the slice inside the final cone primal,
 * dual, and residual vectors returned by the PD-IPM solver.
 */
struct CanonicalConeRecord {
  size_t node_id{};           ///< Node that emitted this cone.
  size_t local_cone_index{};  ///< Cone index within the emitting node.
  size_t global_cone_index{}; ///< Cone index in the accumulated cone block.
  CanonicalConeKind cone_kind{CanonicalConeKind::UNKNOWN};
  size_t cone_offset{}; ///< Starting row in the global cone vectors.
  size_t cone_dim{};    ///< Number of rows occupied by this cone.
};

/**
 * @brief Equality-row slice emitted by one canonical trace node.
 *
 * `row_offset` and `row_count` locate the slice inside the final equality dual
 * and equality residual vectors returned by the PD-IPM solver.
 */
struct CanonicalEqualityRecord {
  size_t node_id{};        ///< Node that emitted these equality rows.
  size_t local_eq_index{}; ///< Equality block index within the emitting node.
  size_t row_offset{};     ///< Starting row in the global equality vectors.
  size_t row_count{};      ///< Number of equality rows in this slice.
};

/**
 * @brief Compile-time canonicalization trace.
 *
 * The trace is tuple-backed to preserve constexpr/no-heap behavior while
 * carrying enough structural metadata for runtime bindings to map final
 * primal-dual values back to canonical atoms, cones, and equalities.
 */
template <typename NodeRecords = std::tuple<>,
          typename ConeRecords = std::tuple<>,
          typename EqualityRecords = std::tuple<>>
struct CanonicalTrace {
  NodeRecords nodes{};          ///< Canonical node records.
  ConeRecords cones{};          ///< Cone slice records.
  EqualityRecords equalities{}; ///< Equality slice records.
};

/// @brief EmptyCanonicalTrace alias to empty CanonicalTrace
using EmptyCanonicalTrace = CanonicalTrace<>;

/// @brief forward decl for canonical_trace_traits
template <typename Trace>
struct canonical_trace_traits;

/// @brief canonical_trace_traits type traits
template <typename NodeTuple, typename ConeTuple, typename EqualityTuple>
struct canonical_trace_traits<
    CanonicalTrace<NodeTuple, ConeTuple, EqualityTuple>> {
  using Nodes = NodeTuple;
  using Cones = ConeTuple;
  using Equalities = EqualityTuple;

  static constexpr size_t NumNodes = std::tuple_size_v<NodeTuple>;
  static constexpr size_t NumCones = std::tuple_size_v<ConeTuple>;
  static constexpr size_t NumEqualities = std::tuple_size_v<EqualityTuple>;
};

template <typename Trace>
using canonical_trace_nodes_t =
    typename canonical_trace_traits<std::remove_cvref_t<Trace>>::Nodes;

template <typename Trace>
using canonical_trace_cones_t =
    typename canonical_trace_traits<std::remove_cvref_t<Trace>>::Cones;

template <typename Trace>
using canonical_trace_equalities_t =
    typename canonical_trace_traits<std::remove_cvref_t<Trace>>::Equalities;

template <typename Trace>
inline constexpr size_t canonical_trace_num_nodes_v =
    canonical_trace_traits<std::remove_cvref_t<Trace>>::NumNodes;

template <typename Trace>
inline constexpr size_t canonical_trace_num_cones_v =
    canonical_trace_traits<std::remove_cvref_t<Trace>>::NumCones;

template <typename Trace>
inline constexpr size_t canonical_trace_num_equalities_v =
    canonical_trace_traits<std::remove_cvref_t<Trace>>::NumEqualities;

template <typename Cone>
struct canonical_cone_kind {
  static constexpr CanonicalConeKind value = CanonicalConeKind::UNKNOWN;
};

template <linalg::VectorLike Vec>
struct canonical_cone_kind<conic::NonnegOrthantCone<Vec>> {
  static constexpr CanonicalConeKind value = CanonicalConeKind::NONNEG_ORTHANT;
};

template <linalg::VectorLike Vec>
struct canonical_cone_kind<conic::SecondOrderCone<Vec>> {
  static constexpr CanonicalConeKind value = CanonicalConeKind::SECOND_ORDER;
};

template <linalg::VectorLike Vec, size_t Dim>
struct canonical_cone_kind<conic::PositiveSemidefiniteCone<Vec, Dim>> {
  static constexpr CanonicalConeKind value =
      CanonicalConeKind::POSITIVE_SEMIDEFINITE;
};

template <linalg::VectorLike Vec>
struct canonical_cone_kind<conic::ExpCone<Vec>> {
  static constexpr CanonicalConeKind value = CanonicalConeKind::EXP;
};

template <linalg::VectorLike Vec>
struct canonical_cone_kind<conic::PowerCone<Vec>> {
  static constexpr CanonicalConeKind value = CanonicalConeKind::POWER;
};

template <typename Cone>
inline constexpr CanonicalConeKind canonical_cone_kind_v =
    canonical_cone_kind<std::remove_cvref_t<Cone>>::value;

/**
 * @brief Extends canonical trace by tuple concat
 *
 * @param trace (Trace)
 * @param nodes (NodeRecords)
 * @param cones (ConeRecords)
 * @param equalities (EqualityRecords)
 *
 * @return CanonicalTrace
 */
template <typename Trace, typename NodeRecords, typename ConeRecords,
          typename EqualityRecords>
constexpr auto extend_canonical_trace(const Trace& trace,
                                      const NodeRecords& nodes,
                                      const ConeRecords& cones,
                                      const EqualityRecords& equalities) {
  return CanonicalTrace<decltype(std::tuple_cat(trace.nodes, nodes)),
                        decltype(std::tuple_cat(trace.cones, cones)),
                        decltype(std::tuple_cat(trace.equalities, equalities))>{
      std::tuple_cat(trace.nodes, nodes), std::tuple_cat(trace.cones, cones),
      std::tuple_cat(trace.equalities, equalities)};
}

/**
 * @brief Make canonical cone records
 *
 * @param node_id (size_t)
 * @param global_cone_start (size_t)
 * @param cone_offset_start (size_t)
 * @param index_sequence<Is...>
 *
 * @return tuple of records
 */
template <typename ConeTuple>
constexpr auto make_canonical_cone_records(const size_t& node_id,
                                           const size_t& global_cone_start,
                                           const size_t& cone_offset_start) {
  using Tuple = std::remove_cvref_t<ConeTuple>;

  auto prior_cone_tuple_offset = [&]<size_t Idx>(auto&& self) -> size_t {
    if constexpr (Idx == 0) {
      return 0;
    } else {
      using PrevCone = std::tuple_element_t<Idx - 1, Tuple>;

      return self.template operator()<Idx - 1>(self) +
             conic::cone_dim_t<PrevCone>;
    }
  };

  auto make_records = [&]<size_t... Is>(std::index_sequence<Is...>) {
    return std::make_tuple(CanonicalConeRecord{
        node_id, Is, global_cone_start + Is,
        canonical_cone_kind_v<std::tuple_element_t<Is, Tuple>>,
        cone_offset_start + prior_cone_tuple_offset.template operator()<Is>(
                                prior_cone_tuple_offset),
        conic::cone_dim_t<std::tuple_element_t<Is, Tuple>>}...);
  };

  return make_records(std::make_index_sequence<std::tuple_size_v<Tuple>>{});
}

/**
 * @brief Write a canonical trace JSON object through a caller-provided
 * appender.
 *
 * The trace owns the structural schema, while the caller owns the concrete
 * character buffer and truncation policy. `append` must accept printf-style
 * format strings and arguments and return `true` when the write fit.
 *
 * @param trace canonical trace to serialize
 * @param append callable used to append formatted text
 *
 * @return true when the entire trace was written
 */
template <typename Trace, typename Append>
inline bool append_canonical_trace_json(const Trace& trace, Append&& append) {
  if (!append("\"canonical_trace\":{")) return false;

  if (!append("\"nodes\":[")) return false;
  bool first = true;
  bool ok = true;
  std::apply(
      [&](const auto&... records) {
        (((ok = ok &&
                append("%s{\"node_id\":%zu,\"parent_id\":%zu,"
                       "\"atom_type\":%u,\"child_index\":%zu}",
                       first ? "" : ",", records.node_id, records.parent_id,
                       static_cast<unsigned>(records.atom_type),
                       records.child_index)),
          first = false),
         ...);
      },
      trace.nodes);
  if (!ok) return false;

  if (!append("],\"cones\":[")) return false;
  first = true;
  ok = true;
  std::apply(
      [&](const auto&... records) {
        (((ok =
               ok && append("%s{\"node_id\":%zu,\"local_cone_index\":%zu,"
                            "\"global_cone_index\":%zu,\"cone_kind\":%u,"
                            "\"cone_offset\":%zu,\"cone_dim\":%zu}",
                            first ? "" : ",", records.node_id,
                            records.local_cone_index, records.global_cone_index,
                            static_cast<unsigned>(records.cone_kind),
                            records.cone_offset, records.cone_dim)),
          first = false),
         ...);
      },
      trace.cones);
  if (!ok) return false;

  if (!append("],\"equalities\":[")) return false;
  first = true;
  ok = true;
  std::apply(
      [&](const auto&... records) {
        (((ok = ok && append("%s{\"node_id\":%zu,\"local_eq_index\":%zu,"
                             "\"row_offset\":%zu,\"row_count\":%zu}",
                             first ? "" : ",", records.node_id,
                             records.local_eq_index, records.row_offset,
                             records.row_count)),
          first = false),
         ...);
      },
      trace.equalities);
  if (!ok) return false;
  return append("]}");
}

} // namespace convex
